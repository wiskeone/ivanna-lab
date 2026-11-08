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
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from ...models import FineTuningJob, DocumentCollection, EvaluationDataset, DocumentChunk
from .callbacks import ProgressCallback, PauseCallback, PauseRequested

logger = logging.getLogger(__name__)


class RAFTTrainer:
    """
    RAFT (Retrieval-Augmented Fine-Tuning) обучение
    """
    MAX_DISTRACTORS = 7

    def __init__(self, job: FineTuningJob, device=None):
        self.job = job
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.base_model = None  # Для KL-дивергенции
        logger.info(f"Initializing RAFTTrainer for job {job.id} on {self.device}")

    def load_general_instructions(self) -> List[Tuple[str, str]]:
        # Заглушка – можно загружать из файла
        return [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Explain quantum computing in simple terms",
             "Quantum computing uses quantum mechanics to solve problems..."),
        ]

    def get_distractor_chunks(self, count: int, exclude_collections: List[int]) -> List[str]:
        chunks = DocumentChunk.objects.filter(
            is_active=True
        ).exclude(
            document__collection_id__in=exclude_collections
        ).values_list('content', flat=True)[:count * 10]
        return random.sample(list(chunks), min(count, len(chunks)))

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

    def prepare_raft_dataset(self) -> Dataset:
        logger.info("Preparing RAFT dataset...")
        self.job.add_log("🎯 Preparing RAFT dataset...")

        examples = []

        # 1. Загружаем инструкции
        instructions = []
        answers = []
        for dataset in self.job.instruction_datasets.all():
            for q in dataset.questions:
                if isinstance(q, dict):
                    if 'answer' in q:
                        instructions.append(q.get('text', ''))
                        answers.append(q['answer'])
                    elif 'output' in q:
                        instructions.append(q.get('instruction', q.get('text', '')))
                        answers.append(q['output'])

        if not instructions:
            raise ValueError("No instruction examples found")

        # 2. Релевантные документы
        relevant_docs = []
        doc_collections = list(self.job.document_collections.all())
        for collection in doc_collections:
            chunks = DocumentChunk.objects.filter(
                document__collection=collection,
                is_active=True
            ).values_list('content', flat=True)[:100]
            relevant_docs.extend(chunks)

        if not relevant_docs:
            raise ValueError("No relevant documents found")

        # 3. Дистракторы
        exclude_ids = [c.id for c in doc_collections]
        distractors = self.get_distractor_chunks(
            count=len(instructions) * self.MAX_DISTRACTORS,
            exclude_collections=exclude_ids
        )

        # 4. Токенизатор для обрезки
        tokenizer = AutoTokenizer.from_pretrained(
            self.job.base_model.local_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_context_tokens = self.job.max_seq_length // 2

        # 5. Создаём примеры
        for i, (inst, ans) in enumerate(zip(instructions, answers)):
            rel_doc = relevant_docs[i % len(relevant_docs)]

            # Oracle
            if random.random() < self.job.oracle_ratio:
                context_parts = [rel_doc]
                context_parts = self._truncate_context(context_parts, tokenizer, max_context_tokens)
                if context_parts:
                    context = "\n\n".join(context_parts)
                    prompt = f"### Context:\n{context}\n\n### Instruction:\n{inst}\n\n### Response:"
                    examples.append({'prompt': prompt, 'response': ans, 'type': 'oracle'})

            # С дистракторами
            if random.random() < self.job.distractor_ratio:
                for k in [7, 5, 3]:
                    if len(distractors) < k:
                        continue
                    dist_samples = random.sample(distractors, k)
                    context_parts = [rel_doc] + dist_samples
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

            # Без контекста
            if random.random() < self.job.no_context_ratio:
                prompt = f"### Instruction:\n{inst}\n\n### Response:"
                examples.append({'prompt': prompt, 'response': ans, 'type': 'no_context'})

        # 6. Общие инструкции
        general_instructions = self.load_general_instructions()
        n_general = int(len(examples) * self.job.general_instructions_ratio)
        for inst, ans in random.sample(general_instructions, min(n_general, len(general_instructions))):
            prompt = f"### Instruction:\n{inst}\n\n### Response:"
            examples.append({'prompt': prompt, 'response': ans, 'type': 'general'})

        # 7. Создаём датасет
        texts = [f"{ex['prompt']} {ex['response']}" for ex in examples]
        dataset = Dataset.from_dict({'text': texts, 'metadata': examples})

        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.job.max_seq_length
            )

        dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        logger.info(f"RAFT dataset prepared with {len(dataset)} examples")
        self.job.add_log(f"✅ RAFT dataset prepared with {len(dataset)} examples")
        return dataset

    def setup_model_with_kl(self):
        logger.info(f"Setting up model on {self.device}...")
        self.job.add_log(f"⚙️ Setting up model on {self.device}...")

        quant = getattr(self.job, 'quantization', None)
        if quant == 'none' or not quant:
            quant = None

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

            if self.job.use_kl_regularization:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.job.base_model.local_path,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map='auto' if self.device.type == 'cuda' else None,
                    trust_remote_code=True
                )
                self.base_model.eval()
                for param in self.base_model.parameters():
                    param.requires_grad = False

            model.gradient_checkpointing_enable()
            if quant in ('int4', 'int8'):
                model = prepare_model_for_kbit_training(model)

        else:
            # CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.cuda.empty_cache()

            if quant in ('int4', 'int8'):
                try:
                    import quanto
                    model = AutoModelForCausalLM.from_pretrained(
                        self.job.base_model.local_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map=None,
                    )
                    if quant == 'int4':
                        quanto.quantize(model, weights=quanto.qint4, activations=None)
                    else:
                        quanto.quantize(model, weights=quanto.qint8, activations=None)
                    quanto.freeze(model)
                except ImportError:
                    logger.warning("quanto not installed, falling back to float32")
                    quant = None
                    model = AutoModelForCausalLM.from_pretrained(
                        self.job.base_model.local_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map=None,
                    )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.job.base_model.local_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=None,
                )
            model = model.to('cpu')
            model.gradient_checkpointing_enable()

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

        # Логируем размещение модели
        if hasattr(model, 'hf_device_map'):
            logger.info(f"📌 Device map: {model.hf_device_map}")
            devices = set(model.hf_device_map.values())
            if 'cpu' in devices:
                logger.warning("⚠️ Some layers are on CPU, training may be slower.")
        else:
            logger.info(f"📌 Model device: {next(model.parameters()).device}")

        # LoRA
        lora_config = LoraConfig(
            r=self.job.lora_r,
            lora_alpha=self.job.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.job.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, lora_config

    def compute_kl_loss(self, student_logits, teacher_logits):
        student_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        return kl_loss

    def train(self):
        try:
            # Включаем оптимизации CuDNN
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            self.job.status = 'PREPARING'
            self.job.save()
            self.job.add_log("📦 Starting RAFT fine-tuning...")

            dataset = self.prepare_raft_dataset()
            split_dataset = dataset.train_test_split(test_size=0.1)
            self.job.add_log(f"Train/val split: {len(split_dataset['train'])} / {len(split_dataset['test'])}")

            self.job.status = 'TRAINING'
            self.job.save()
            self.job.add_log("🔧 Setting up model with KL regularization...")

            model, lora_config = self.setup_model_with_kl()
            tokenizer = AutoTokenizer.from_pretrained(
                self.job.base_model.local_path,
                trust_remote_code=True
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True
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
                logging_steps=1,                          # логируем каждый шаг в консоль
                save_steps=500,                            # чекпоинты каждые 500 шагов
                save_total_limit=3,
                eval_strategy="steps",
                eval_steps=500,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",
                no_cuda=self.device.type == 'cpu',
                remove_unused_columns=False,
                dataloader_num_workers=0,                  # отключаем дополнительные процессы
            )

            progress_callback = ProgressCallback(
                job=self.job,
                total_steps=total_steps,
                save_metrics_every=10,
                histogram_interval=50
            )

            # Кастомный Trainer с KL-loss
            class RAFTTrainer(Trainer):
                def __init__(self, base_model=None, kl_lambda=0.1, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.base_model = base_model
                    self.kl_lambda = kl_lambda

                def compute_loss(self, model, inputs, return_outputs=False):
                    outputs = model(**inputs)
                    ce_loss = outputs.loss
                    if self.base_model is not None and self.kl_lambda > 0:
                        with torch.no_grad():
                            base_outputs = self.base_model(**inputs)
                        kl_loss = self.compute_kl_loss(outputs.logits, base_outputs.logits)
                        loss = ce_loss + self.kl_lambda * kl_loss
                    else:
                        loss = ce_loss
                    return (loss, outputs) if return_outputs else loss

            trainer = RAFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=split_dataset['test'],
                data_collator=data_collator,
                base_model=self.base_model,
                kl_lambda=self.job.kl_lambda if self.job.use_kl_regularization else 0.0,
                callbacks=[progress_callback, PauseCallback(self.job)],
            )

            checkpoint = None
            if os.path.exists(output_dir) and any(f.startswith('checkpoint-') for f in os.listdir(output_dir)):
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