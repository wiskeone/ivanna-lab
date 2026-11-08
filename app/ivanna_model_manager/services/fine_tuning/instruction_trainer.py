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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from ...models import FineTuningJob, EvaluationDataset
from .callbacks import ProgressCallback

logger = logging.getLogger(__name__)


class InstructionTrainer:
    """
    Обучение модели на инструкциях (формат вопрос-ответ)
    """

    def __init__(self, job: FineTuningJob, device=None):
        self.job = job
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

        texts = []
        for inst, ans in zip(instructions, answers):
            prompt = f"### Instruction:\n{inst}\n\n### Response:\n{ans}"
            texts.append(prompt)

        logger.info(f"Total instruction examples: {len(texts)}")
        self.job.add_log(f"✅ Loaded {len(texts)} instruction examples")

        dataset = Dataset.from_dict({'text': texts})

        tokenizer = AutoTokenizer.from_pretrained(
            self.job.base_model.local_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.job.max_seq_length
            )

        dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        logger.info(f"Dataset prepared with {len(dataset)} examples")
        self.job.add_log(f"✅ Dataset prepared with {len(dataset)} examples")
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
                    logger.info(f"Loading model with quanto {quant} on CPU")
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
                    logger.info(f"Model quantized with quanto {quant} and frozen")
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
                logger.info("Loading model in float32 on CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    self.job.base_model.local_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=None,
                )
            model = model.to('cpu')
            model.gradient_checkpointing_enable()

        # Логируем размещение модели
        if hasattr(model, 'hf_device_map'):
            logger.info(f"📌 Device map: {model.hf_device_map}")
            devices = set(model.hf_device_map.values())
            if 'cpu' in devices:
                logger.warning("⚠️ Some layers are on CPU, training may be slower.")
        else:
            logger.info(f"📌 Model device: {next(model.parameters()).device}")

        # LoRA (для инструкций можно использовать те же модули)
        lora_config = LoraConfig(
            r=self.job.lora_r,
            lora_alpha=self.job.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.job.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, lora_config

    def train(self):
        try:
            # Включаем оптимизации CuDNN
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            self.job.status = 'PREPARING'
            self.job.save()
            self.job.add_log("📦 Starting instruction fine-tuning...")

            dataset = self.prepare_dataset()
            split_dataset = dataset.train_test_split(test_size=0.1)
            self.job.add_log(f"Train/val split: {len(split_dataset['train'])} / {len(split_dataset['test'])}")

            self.job.status = 'TRAINING'
            self.job.save()
            self.job.add_log("🔧 Setting up model...")

            model, lora_config = self.setup_model()
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

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=split_dataset['test'],
                data_collator=data_collator,
                callbacks=[progress_callback],
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

            self.job.add_log(f"✅ Instruction training completed. Model saved to {output_dir}")
            return output_model

        except Exception as e:
            logger.error(f"Error in instruction training: {e}", exc_info=True)
            self.job.add_error(str(e), traceback.format_exc())
            self.job.fail(str(e), {'traceback': traceback.format_exc()})
            raise