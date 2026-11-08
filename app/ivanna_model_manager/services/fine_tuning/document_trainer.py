import os
import logging
import torch
import traceback
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from ...models import FineTuningJob, DocumentChunk
from .callbacks import ProgressCallback   # ваш существующий callback

logger = logging.getLogger(__name__)


class DocumentTrainer:
    """
    Обучение модели на документах (языковое моделирование) с поддержкой квантизации,
    автоматическим распределением модели (device_map='auto') и чекпоинтами на ключевых этапах.
    """

    def __init__(self, job: FineTuningJob, device: Optional[str] = None):
        self.job = job
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Initializing DocumentTrainer for job {job.id} on {self.device}")

    def prepare_dataset(self) -> Dataset:
        """Загружает тексты из коллекций и подготавливает датасет."""
        logger.info("Preparing document dataset...")
        self.job.add_log("📄 Preparing document dataset...")

        texts = []
        collections = self.job.document_collections.all()

        for collection in collections:
            logger.info(f"Loading documents from collection: {collection.name}")
            self.job.add_log(f"  - Loading from collection: {collection.name}")
            chunks = DocumentChunk.objects.filter(
                document__collection=collection,
                is_active=True
            ).values_list('content', flat=True)[:10000]  # лимит для теста
            texts.extend(chunks)

        if not texts:
            raise ValueError("No documents found in selected collections")

        logger.info(f"Total text chunks loaded: {len(texts)}")
        self.job.add_log(f"✅ Loaded {len(texts)} text chunks")

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
        """
        Настройка модели с автоматическим распределением (device_map='auto').
        Для GPU используется bitsandbytes (int4/int8), для CPU – quanto (если доступен).
        """
        logger.info(f"Setting up model on {self.device}...")
        self.job.add_log(f"⚙️ Setting up model on {self.device}...")

        quant = getattr(self.job, 'quantization', None)
        if quant == 'none' or not quant:
            quant = None

        if self.device.type == 'cuda':
            # GPU
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
                else:  # int8
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

    def train(self):
        """Запуск обучения с чекпоинтами и отслеживанием прогресса."""
        try:
            # Включаем оптимизации CuDNN
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            self.job.status = 'PREPARING'
            self.job.save()
            self.job.add_log("📦 Starting document fine-tuning...")

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

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )

            output_dir = self.job.output_model.local_path
            if not output_dir:
                raise ValueError("Output model path is not set")
            os.makedirs(output_dir, exist_ok=True)

            effective_batch_size = self.job.batch_size * self.job.gradient_accumulation_steps
            total_steps = len(split_dataset['train']) * self.job.num_epochs // effective_batch_size
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
                save_steps=500,                            # сохраняем реже (каждые 500 шагов)
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

            progress_callback = ProgressCallback(self.job, total_steps)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=split_dataset['train'],
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

            self.job.add_log("💾 Saving final model and tokenizer...")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            output_model = self.job.output_model
            output_model.status = 'DOWNLOADED'
            output_model.save()

            self.job.add_log(f"✅ Training completed, model saved to {output_dir}")
            return output_model

        except Exception as e:
            logger.error(f"Error in document training: {e}", exc_info=True)
            self.job.fail(str(e), {'traceback': traceback.format_exc()})
            raise