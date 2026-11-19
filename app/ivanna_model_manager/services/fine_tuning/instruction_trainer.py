import os
import logging
import torch
import traceback
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from ...models import FineTuningJob, EvaluationDataset
from .callbacks import ProgressCallback, PauseCallback, PauseRequested
from .lora import build_lora_config, apply_lora_plus

logger = logging.getLogger(__name__)


class InstructionTrainer:
    """
    Обучение модели на инструкциях (формат вопрос-ответ).
    Completion-only маскирование: loss считается только по ответу (+EOS).
    """

    def __init__(self, job: FineTuningJob, device=None, resume: bool = False):
        self.job = job
        self.resume = resume
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Initializing InstructionTrainer for job {job.id} on {self.device}")

    def prepare_dataset(self) -> Dataset:
        logger.info("Preparing instruction dataset...")
        self.job.add_log("📋 Preparing instruction dataset...")

        instructions = []
        answers = []

        datasets = self.job.instruction_datasets.all()

        for dataset in datasets:
            logger.info(f"Loading dataset: {dataset.name}")
            self.job.add_log(f"  - Loading dataset: {dataset.name}")
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

        tokenizer = AutoTokenizer.from_pretrained(
            self.job.base_model.local_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rows = {'input_ids': [], 'attention_mask': [], 'labels': []}
        max_len = self.job.max_seq_length
        for inst, ans in zip(instructions, answers):
            prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
            full = prompt + ans + tokenizer.eos_token
            full_ids = tokenizer(full, truncation=True, max_length=max_len).input_ids
            # completion-only masking: the prompt part teaches nothing
            prompt_ids = tokenizer(prompt, truncation=True, max_length=len(full_ids)).input_ids
            p_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * p_len + full_ids[p_len:]
            rows['input_ids'].append(full_ids)
            rows['attention_mask'].append([1] * len(full_ids))
            rows['labels'].append(labels)

        dataset = Dataset.from_dict(rows)

        logger.info(f"Total instruction examples: {len(dataset)}")
        self.job.add_log(f"✅ Loaded {len(dataset)} instruction examples")
        return dataset

    def setup_model(self):
        logger.info(f"Setting up model on {self.device}...")
        self.job.add_log(f"⚙️ Setting up model on {self.device}...")

        quant = getattr(self.job, 'quantization', None)
        if quant == 'none' or not quant:
            quant = None

        if self.device.type == 'cuda':
            if quant in ('int4', 'int8'):
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError:
                    raise ImportError("bitsandbytes not installed")
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
            elif quant in ('float16', 'bfloat16', None):
                quantization_config = None
                if quant == 'float16':
                    torch_dtype = torch.float16
                elif quant == 'bfloat16':
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
                device_map = 'auto'
            else:
                logger.warning(f"Unknown quantization '{quant}' for GPU, using float16")
                quantization_config = None
                torch_dtype = torch.float16
                device_map = 'auto'

            model = AutoModelForCausalLM.from_pretrained(
                self.job.base_model.local_path,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            # use_reentrant=False: the reentrant path silently zeroes LoRA grads
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
            if quant in ('int4', 'int8'):
                model = prepare_model_for_kbit_training(
                    model, gradient_checkpointing_kwargs={'use_reentrant': False})

        else:
            # CPU: int4/int8 quantized training is bnb-only (CUDA) — a clear error
            # instead of a silent quanto/fp32 swamp
            if quant in ('int4', 'int8'):
                raise ValueError(
                    f"CPU fallback does not support quantization='{quant}' "
                    f"(bitsandbytes requires CUDA). Set the job quantization to "
                    f"'float16' or 'none' to train on CPU.")

            if torch.cuda.is_available():
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.cuda.empty_cache()

            logger.info("Loading model in float32 on CPU")
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

        # Логируем размещение модели
        if hasattr(model, 'hf_device_map'):
            logger.info(f"📌 Device map: {model.hf_device_map}")
            devices = set(model.hf_device_map.values())
            if 'cpu' in devices:
                logger.warning("⚠️ Some layers are on CPU, training may be slower.")
        else:
            logger.info(f"📌 Model device: {next(model.parameters()).device}")

        # LoRA via the shared Tier-1 helper (DoRA / rsLoRA / target presets)
        lora_config = build_lora_config(self.job, self.job.base_model.local_path)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, lora_config

    def train(self):
        try:
            self.job.status = 'PREPARING'
            self.job.save()
            self.job.add_log("📦 Starting instruction fine-tuning...")

            dataset = self.prepare_dataset()
            # train_test_split raises when the test split rounds to zero
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
            self.job.add_log("🔧 Setting up model...")

            model, lora_config = self.setup_model()
            tokenizer = AutoTokenizer.from_pretrained(
                self.job.base_model.local_path,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

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

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[progress_callback, PauseCallback(self.job)],
            )
            # LoRA+: separate lr for lora_B (skips the stock optimizer)
            apply_lora_plus(trainer, self.job.lora_plus_lr_ratio)

            # guarded resume: only an explicitly resumed run picks up old checkpoints
            checkpoint = None
            if self.resume and os.path.exists(output_dir) and \
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

            self.job.add_log(f"✅ Instruction training completed. Model saved to {output_dir}")
            return output_model

        except PauseRequested:
            self.job.status = 'PAUSED'
            self.job.save()
            self.job.add_log("⏸ Обучение приостановлено пользователем — прогресс в чекпоинте")
            return None

        except Exception as e:
            logger.error(f"Error in instruction training: {e}", exc_info=True)
            self.job.add_error(str(e), traceback.format_exc())
            self.job.fail(str(e), {'traceback': traceback.format_exc()})
            raise
