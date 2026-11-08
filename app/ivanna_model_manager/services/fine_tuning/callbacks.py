import time
import numpy as np
from transformers import TrainerCallback
from ...models import TrainingMetric
import logging

logger = logging.getLogger(__name__)


class PauseRequested(Exception):
    """v5.1: брошен PauseCallback-ом, когда пользователь запросил паузу.

    Ловится ВНЕ trainer.train(): чекпоинт к этому моменту уже сохранён
    HF Trainer-ом (save_steps), resume подхватит его через
    resume_from_checkpoint."""

from ...models import TrainingMetric
import logging

logger = logging.getLogger(__name__)


class ProgressCallback(TrainerCallback):
    """
    Callback для сохранения пошаговых метрик и гистограмм во время fine-tuning,
    а также логирования каждого шага в консоль и каждых 10 шагов в лог задачи.
    """
    def __init__(self, job, total_steps, save_metrics_every=10, histogram_interval=50):
        self.job = job
        self.total_steps = total_steps
        self.save_metrics_every = save_metrics_every
        self.histogram_interval = histogram_interval
        self.start_time = time.time()
        self.target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    def _collect_histograms(self, model):
        """Собирает гистограммы весов и градиентов для целевых слоёв."""
        weights_hists = {}
        grad_hists = {}
        if model is None:
            return weights_hists, grad_hists

        for name, param in model.named_parameters():
            if any(layer in name for layer in self.target_layers) and 'lora' in name:
                data = param.detach().cpu().numpy().flatten()
                if data.size > 0:
                    hist, _ = np.histogram(data, bins=50)
                    weights_hists[name] = hist.tolist()

                if param.grad is not None:
                    grad_data = param.grad.detach().cpu().numpy().flatten()
                    if grad_data.size > 0:
                        grad_hist, _ = np.histogram(grad_data, bins=50)
                        grad_hists[name] = grad_hist.tolist()
        return weights_hists, grad_hists

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step

        # Логируем каждый шаг в консоль
        logger.info(f"Step {step}/{self.total_steps} completed")

        # Каждые 10 шагов добавляем запись в лог задачи
        if step % 10 == 0:
            self.job.add_log(f"Step {step}/{self.total_steps} completed")

        # Обновляем прогресс в задаче
        if self.total_steps:
            progress = 100.0 * step / self.total_steps
            self.job.update_progress(progress, f"Step {step}/{self.total_steps}")

        # Сохраняем метрики и гистограммы с заданной периодичностью
        if step % self.save_metrics_every != 0:
            return

        log = state.log_history[-1] if state.log_history else {}
        train_loss = log.get('loss')
        grad_norm = log.get('grad_norm')
        learning_rate = log.get('learning_rate')
        epoch = log.get('epoch')
        time_elapsed = time.time() - self.start_time

        weights_hists = {}
        grad_hists = {}
        if step % self.histogram_interval == 0:
            weights_hists, grad_hists = self._collect_histograms(model)

        if not TrainingMetric.objects.filter(job=self.job, step=step).exists():
            TrainingMetric.objects.create(
                job=self.job,
                step=step,
                epoch=epoch,
                train_loss=train_loss,
                grad_norm=grad_norm,
                learning_rate=learning_rate,
                weights_histograms=weights_hists,
                grad_histograms=grad_hists,
                time_elapsed=time_elapsed,
            )
        else:
            logger.debug(f"Metric for step {step} already exists, skipping")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            TrainingMetric.objects.create(
                job=self.job,
                step=state.global_step,
                epoch=state.epoch,
                eval_loss=metrics['eval_loss'],
                time_elapsed=time.time() - self.start_time,
            )

class PauseCallback(TrainerCallback):
    """
    v5.1: пауза fine-tuning по флагу в БД.

    Каждый шаг обновляет job из БД; если пользователь нажал «Пауза»
    (pause_requested=True) — бросает PauseRequested. HF Trainer к этому
    моменту уже сохранил чекпоинт (save_steps), поэтому резюм подхватит
    прогресс через resume_from_checkpoint без потери шагов.
    """

    def __init__(self, job, check_every: int = 5):
        self.job = job
        self.check_every = max(1, check_every)
        self._counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._counter += 1
        if self._counter % self.check_every:
            return
        # свежий флаг из БД (не из кэшированного объекта)
        from ...models import FineTuningJob
        if FineTuningJob.objects.filter(
            pk=self.job.pk, pause_requested=True
        ).exists():
            control.should_training_stop = True
            raise PauseRequested(f"Job {self.job.pk} paused at step {state.global_step}")
