import os
import logging
import random
import torch
import torch.nn.functional as F
import traceback
from typing import Dict, Any, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from ...models import FineTuningJob, DocumentCollection, EvaluationDataset, DocumentChunk
from .callbacks import ProgressCallback, PauseCallback, PauseRequested
from .dataset_sources import collect_instruction_pairs, collect_document_collections
from .lora import build_lora_config, apply_lora_plus

logger = logging.getLogger(__name__)

GENERAL_INSTRUCTIONS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain quantum computing in simple terms",
     "Quantum computing uses qubits, which can be in superposition, to solve certain problems faster."),
    ("Write a Python function that reverses a string",
     "def reverse(s): return s[::-1]"),
    ("Summarize what a REST API is",
     "A REST API is an HTTP interface where resources are addressed by URLs and manipulated with standard methods."),
    ("Convert 25 degrees Celsius to Fahrenheit",
     "25 °C is 77 °F: F = C × 9/5 + 32."),
    ("Give three tips for writing clean code",
     "Use clear names, keep functions small, and follow the project's style guide."),
    ("What causes seasons on Earth?",
     "Seasons are caused by the tilt of Earth's rotational axis relative to its orbit around the Sun."),
    ("Explain the difference between a list and a tuple in Python",
     "Lists are mutable; tuples are immutable and can be dictionary keys."),
    ("How does photosynthesis work?",
     "Plants convert CO2 and water into glucose and oxygen using light energy and chlorophyll."),
    ("Rewrite this sentence politely: 'Send me the file'",
     "Could you please send me the file when you have a moment?"),
]


class RAFTTrainer:
    """
    RAFT (Retrieval-Augmented Fine-Tuning) training.
    Completion-only masking: loss over the answer (+EOS), prompt and context set to -100.
    """

    MAX_DISTRACTORS = 7
    HEATMAP_BUCKETS = 24

    def __init__(self, job: FineTuningJob, device=None, resume: bool = False):
        self.job = job
        self.resume = resume
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.base_model = None  # KL teacher
        logger.info(f"Initializing RAFTTrainer for job {job.id} on {self.device}")

    def load_general_instructions(self) -> List[Tuple[str, str]]:
        return GENERAL_INSTRUCTIONS

    def get_distractor_chunks(self, count: int, exclude_collections: List[int]) -> List[str]:
        chunks = list(DocumentChunk.objects.filter(
            is_active=True
        ).exclude(
            document__collection_id__in=exclude_collections
        ).order_by('?').values_list('content', flat=True)[:count * 3])
        return chunks[:count]

    def _truncate_context(self, context_parts: List[str], tokenizer, max_tokens: int) -> List[str]:
        result = []
        total_tokens = 0
        for part in context_parts:
            tokens = len(tokenizer.encode(part, add_special_tokens=False))
            if total_tokens + tokens > max_tokens:
                break
            result.append(part)
            total_tokens += tokens
        return result

    def _rank_chunks_for_question(self, question: str, chunks: List[str], top_n: int) -> List[str]:
        """Poor-man's retrieval: rank chunks by word overlap with the question.

        The QA datasets carry no source-chunk link, so round-robin pairing trained
        on unrelated context; lexical overlap is the honest cheap fix.
        """
        q_words = set(w.lower() for w in question.split() if len(w) > 3)
        scored = []
        for c in chunks:
            c_words = set(w.lower() for w in c.split()[:400] if len(w) > 3)
            overlap = len(q_words & c_words)
            scored.append((overlap, c))
        scored.sort(key=lambda t: -t[0])
        return [c for _, c in scored[:top_n]]

    def prepare_raft_dataset(self) -> Dataset:
        logger.info("Preparing RAFT dataset...")
        self.job.add_log("🎯 Preparing RAFT dataset...")

        examples = []

        # 1. Load instructions
        # Audit fix: the create form stores dataset ids in training_config —
        # the M2M relation is never populated, so UI-created jobs always
        # failed with "No instruction examples found"
        instructions = []
        answers = []
        for inst, ans in collect_instruction_pairs(self.job):
            instructions.append(inst)
            answers.append(ans)

        if not instructions:
            raise ValueError("No instruction examples found")

        # 2. Relevant documents
        relevant_docs = []
        doc_collections = collect_document_collections(self.job)
        for collection in doc_collections:
            chunks = list(DocumentChunk.objects.filter(
                document__collection=collection,
                is_active=True
            ).order_by('?').values_list('content', flat=True)[:200])
            relevant_docs.extend(chunks)

        if not relevant_docs:
            raise ValueError("No relevant documents found")

        # 3. Distractors
        exclude_ids = [c.id for c in doc_collections]
        distractors = self.get_distractor_chunks(
            count=len(instructions) * self.MAX_DISTRACTORS,
            exclude_collections=exclude_ids
        )

        # 4. Tokenizer for truncation
        tokenizer = AutoTokenizer.from_pretrained(
            self.job.base_model.local_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_context_tokens = self.job.max_seq_length // 2

        # 5. Build examples
        for i, (inst, ans) in enumerate(zip(instructions, answers)):
            # lexical-overlap ranking instead of round-robin: the oracle doc
            # must actually relate to the question
            ranked = self._rank_chunks_for_question(inst, relevant_docs, top_n=1 + self.MAX_DISTRACTORS)
            if not ranked:
                continue
            rel_doc = ranked[0]
            other_ranked = ranked[1:]

            # Oracle
            if random.random() < self.job.oracle_ratio:
                context_parts = self._truncate_context([rel_doc], tokenizer, max_context_tokens)
                if context_parts:
                    context = "\n\n".join(context_parts)
                    prompt = f"### Context:\n{context}\n\n### Instruction:\n{inst}\n\n### Response:"
                    examples.append({'prompt': prompt, 'response': ans, 'type': 'oracle'})

            # With distractors
            if random.random() < self.job.distractor_ratio:
                for k in [7, 5, 3]:
                    if len(other_ranked) + len(distractors) < k:
                        continue
                    need = k - min(k, len(other_ranked))
                    dist_samples = random.sample(distractors, min(need, len(distractors))) + \
                        other_ranked[:k - need]
                    context_parts = [rel_doc] + dist_samples[:k]
                    context_parts = self._truncate_context(context_parts, tokenizer, max_context_tokens)
                    if len(context_parts) >= 2:
                        context = "\n\n".join(context_parts)
                        prompt = f"### Context:\n{context}\n\n### Instruction:\n{inst}\n\n### Response:"
                        examples.append({
                            'prompt': prompt,
                            'response': ans,
                            'type': f'with_distractors_{k}'
                        })
                        break

            # Without context
            if random.random() < self.job.no_context_ratio:
                prompt = f"### Instruction:\n{inst}\n\n### Response:"
                examples.append({'prompt': prompt, 'response': ans, 'type': 'no_context'})

        # 6. General instructions
        general_instructions = self.load_general_instructions()
        n_general = int(len(examples) * self.job.general_instructions_ratio)
        for inst, ans in random.sample(general_instructions, min(n_general, len(general_instructions))):
            prompt = f"### Instruction:\n{inst}\n\n### Response:"
            examples.append({'prompt': prompt, 'response': ans, 'type': 'general'})

        if not examples:
            raise ValueError("RAFT produced zero examples (check ratios and datasets)")

        # 7. Completion-only tokenization: labels over the response (+EOS) only
        rows = {'input_ids': [], 'attention_mask': [], 'labels': []}
        max_len = self.job.max_seq_length
        for ex in examples:
            full = ex['prompt'] + ' ' + ex['response'] + tokenizer.eos_token
            full_ids = tokenizer(full, truncation=True, max_length=max_len).input_ids
            prompt_ids = tokenizer(ex['prompt'], truncation=True, max_length=len(full_ids)).input_ids
            p_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * p_len + full_ids[p_len:]
            rows['input_ids'].append(full_ids)
            rows['attention_mask'].append([1] * len(full_ids))
            rows['labels'].append(labels)
        dataset = Dataset.from_dict(rows)

        logger.info(f"RAFT dataset prepared with {len(dataset)} examples")
        self.job.add_log(f"✅ RAFT dataset prepared with {len(dataset)} examples")
        return dataset

    def setup_model_with_kl(self):
        logger.info(f"Setting up model on {self.device}...")
        self.job.add_log(f"⚙️ Setting up model on {self.device}...")

        quant = getattr(self.job, 'quantization', None)
        if quant == 'none' or not quant:
            quant = None

        if quant in ('int4', 'int8') and self.device.type == 'cpu':
            raise ValueError(
                f"CPU fallback does not support quantization='{quant}' "
                f"(bitsandbytes requires CUDA). Set the job quantization to "
                f"'float16' or 'none' to train on CPU.")

        if self.device.type == 'cuda':
            # GPU – bitsandbytes
            if quant in ('int4', 'int8'):
                from transformers import BitsAndBytesConfig
                if quant == 'int4':
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                else:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                quantization_config = bnb_config
                torch_dtype = torch.float16
                device_map = 'auto'
            else:
                quantization_config = None
                if quant == 'float16':
                    torch_dtype = torch.float16
                elif quant == 'bfloat16':
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
                device_map = 'auto'

            model = AutoModelForCausalLM.from_pretrained(
                self.job.base_model.local_path,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # KL teacher: a second unquantized copy — on 4GB cards this does not
            # fit next to the nf4 student; disable with a clear log line
            if self.job.use_kl_regularization:
                free_bytes, total_bytes = torch.cuda.mem_get_info() if \
                    torch.cuda.is_available() else (0, 1)
                if total_bytes and total_bytes < 8 * 2**30:
                    logger.warning('KL teacher disabled: GPU < 8GB cannot fit '
                                   'the student + unquantized teacher')
                    self.job.add_log('⚠️ KL regularization disabled: not enough VRAM '
                                     'for the teacher model on this GPU')
                    self.job.use_kl_regularization = False
                else:
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        self.job.base_model.local_path,
                        torch_dtype=torch.float16,
                        device_map='auto',
                        trust_remote_code=True
                    )
                    self.base_model.eval()
                    for param in self.base_model.parameters():
                        param.requires_grad = False

            # use_reentrant=False: the reentrant path silently zeroes LoRA grads
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
            if quant in ('int4', 'int8'):
                model = prepare_model_for_kbit_training(
                    model, gradient_checkpointing_kwargs={'use_reentrant': False})

        else:
            # CPU
            if torch.cuda.is_available():
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.cuda.empty_cache()

            model = AutoModelForCausalLM.from_pretrained(
                self.job.base_model.local_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None,
            )
            model = model.to('cpu')
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

            if self.job.use_kl_regularization:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.job.base_model.local_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to('cpu')
                self.base_model.eval()
                for param in self.base_model.parameters():
                    param.requires_grad = False

        # Log the model placement
        if hasattr(model, 'hf_device_map'):
            logger.info(f"📌 Device map: {model.hf_device_map}")
            devices = set(model.hf_device_map.values())
            if 'cpu' in devices:
                logger.warning("⚠️ Some layers are on CPU, training may be slower.")
        else:
            logger.info(f"📌 Model device: {next(model.parameters()).device}")

        # LoRA via the shared Tier-1 helper
        lora_config = build_lora_config(self.job, self.job.base_model.local_path)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, lora_config

    def compute_kl_loss(self, student_logits, teacher_logits, attention_mask=None):
        """Per-token KL: sum over vocabulary, mean over ATTENDED tokens
        (batchmean on [B,S,V] inflates the weight by sequence length)."""
        logp = F.log_softmax(student_logits, dim=-1)
        p = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(logp, p, reduction='none').sum(-1)  # [B, S]
        if attention_mask is not None:
            mask = attention_mask.to(kl.device)
            denom = mask.sum().clamp(min=1)
            return (kl * mask).sum() / denom
        return kl.mean()

    def train(self, resume: bool = False):
        try:
            self.job.status = 'PREPARING'
            self.job.save()
            self.job.add_log("📦 Starting RAFT fine-tuning...")

            dataset = self.prepare_raft_dataset()
            if len(dataset) >= 20:
                split_dataset = dataset.train_test_split(test_size=0.1)
                self.job.add_log(f"Train/val split: {len(split_dataset['train'])} / {len(split_dataset['test'])}")
                eval_dataset = split_dataset['test']
            else:
                split_dataset = {'train': dataset}
                eval_dataset = None
                self.job.add_log(f"Train split: {len(dataset)} (no val split — small dataset)")

            self.job.status = 'TRAINING'
            self.job.save()
            self.job.add_log("🔧 Setting up model with KL regularization...")

            model, lora_config = self.setup_model_with_kl()
            tokenizer = AutoTokenizer.from_pretrained(
                self.job.base_model.local_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                label_pad_token_id=-100,
            )

            output_dir = self.job.output_model.local_path
            if not output_dir:
                raise ValueError("Output model path is not set")
            os.makedirs(output_dir, exist_ok=True)

            train_dataset = split_dataset['train']
            total_steps = len(train_dataset) * self.job.num_epochs // (
                self.job.batch_size * self.job.gradient_accumulation_steps
            )
            self.job.add_log(f"Total training steps: {total_steps}")

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.job.num_epochs,
                per_device_train_batch_size=self.job.batch_size,
                gradient_accumulation_steps=self.job.gradient_accumulation_steps,
                learning_rate=self.job.learning_rate,
                fp16=self.device.type == 'cuda',
                optim="adamw_8bit" if self.device.type == 'cuda' else "adamw_torch",
                logging_steps=1,
                save_steps=500,
                save_total_limit=3,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=500,
                load_best_model_at_end=bool(eval_dataset),
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",
                use_cpu=self.device.type == 'cpu',
                remove_unused_columns=False,
                dataloader_num_workers=0,
                gradient_checkpointing_kwargs={'use_reentrant': False},
                **({'neftune_noise_alpha': self.job.neftune_alpha}
                   if self.job.neftune_alpha else {}),
            )

            progress_callback = ProgressCallback(
                job=self.job,
                total_steps=total_steps,
                save_metrics_every=10,
                histogram_interval=50
            )

            # Custom Trainer with KL loss (v5 signature; KL helper inside the class)
            class RAFTTrainer(Trainer):
                def __init__(self, base_model=None, kl_lambda=0.1, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.base_model = base_model
                    self.kl_lambda = kl_lambda

                def compute_kl_loss(self, student_logits, teacher_logits, attention_mask=None):
                    logp = F.log_softmax(student_logits, dim=-1)
                    p = F.softmax(teacher_logits, dim=-1)
                    kl = F.kl_div(logp, p, reduction='none').sum(-1)
                    if attention_mask is not None:
                        mask = attention_mask.to(kl.device)
                        return (kl * mask).sum() / mask.sum().clamp(min=1)
                    return kl.mean()

                def compute_loss(self, model, inputs, return_outputs=False,
                                 num_items_in_batch=None, **kwargs):
                    outputs = model(**inputs)
                    ce_loss = outputs.loss
                    if self.base_model is not None and self.kl_lambda > 0:
                        with torch.no_grad():
                            base_outputs = self.base_model(**inputs)
                        kl_loss = self.compute_kl_loss(
                            outputs.logits, base_outputs.logits,
                            inputs.get('attention_mask'))
                        loss = ce_loss + self.kl_lambda * kl_loss
                    else:
                        loss = ce_loss
                    return (loss, outputs) if return_outputs else loss

            trainer = RAFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                base_model=self.base_model,
                kl_lambda=self.job.kl_lambda if self.job.use_kl_regularization else 0.0,
                callbacks=[progress_callback, PauseCallback(self.job)],
            )
            apply_lora_plus(trainer, self.job.lora_plus_lr_ratio)

            # guarded resume: only an explicitly resumed run picks up old checkpoints
            checkpoint = None
            if resume and os.path.exists(output_dir) and \
                    any(f.startswith('checkpoint-') for f in os.listdir(output_dir)):
                checkpoint = True
                self.job.add_log("🔄 Resuming from existing checkpoint...")

            self.job.add_log("🚀 Starting training loop..." + (" (resuming)" if checkpoint else ""))
            trainer.train(resume_from_checkpoint=checkpoint)

            self.job.add_log("💾 Saving model and tokenizer...")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            output_model = self.job.output_model
            output_model.status = 'DOWNLOADED'
            output_model.save()

            self.job.add_log(f"✅ RAFT training completed. Model saved to {output_dir}")
            return output_model

        except PauseRequested:
            self.job.status = 'PAUSED'
            self.job.save()
            self.job.add_log("⏸ Обучение приостановлено — прогресс в чекпоинте")
            return None

        except Exception as e:
            logger.error(f"Error in RAFT training: {e}", exc_info=True)
            self.job.add_error(str(e), traceback.format_exc())
            self.job.fail(str(e), {'traceback': traceback.format_exc()})
            raise
