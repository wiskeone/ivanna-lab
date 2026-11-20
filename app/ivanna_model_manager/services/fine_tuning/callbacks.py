import time
import numpy as np
from transformers import TrainerCallback
from ...models import TrainingMetric
import logging

logger = logging.getLogger(__name__)


class PauseRequested(Exception):
    """v5.1: raised by PauseCallback when the user requested a pause.

    Caught OUTSIDE trainer.train(): by that point the checkpoint has already
    been saved by the HF Trainer (save_steps); resume picks it up via
    resume_from_checkpoint."""

from ...models import TrainingMetric
import logging

logger = logging.getLogger(__name__)


class ProgressCallback(TrainerCallback):
    """
    Callback that stores per-step metrics and histograms during fine-tuning,
    logs every step to the console and every 10 steps to the job log.
    """
    def __init__(self, job, total_steps, save_metrics_every=10, histogram_interval=50):
        self.job = job
        self.total_steps = total_steps
        self.save_metrics_every = save_metrics_every
        self.histogram_interval = histogram_interval
        self.start_time = time.time()
        self.target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    def _collect_histograms(self, model):
        """Collects weight and gradient histograms for the target layers."""
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

        # Log every step to the console
        logger.info(f"Step {step}/{self.total_steps} completed")

        # Every 10 steps, append an entry to the job log
        if step % 10 == 0:
            self.job.add_log(f"Step {step}/{self.total_steps} completed")

        # Update the job progress
        if self.total_steps:
            progress = 100.0 * step / self.total_steps
            self.job.update_progress(progress, f"Step {step}/{self.total_steps}")

        # Save metrics and histograms at the configured interval
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
    v5.1: fine-tuning pause driven by a DB flag.

    Each step refreshes the job from the DB; if the user pressed "Pause"
    (pause_requested=True) — raises PauseRequested. By that point the HF
    Trainer has already saved the checkpoint (save_steps), so the resume
    picks up progress via resume_from_checkpoint without losing steps.
    """

    def __init__(self, job, check_every: int = 5):
        self.job = job
        self.check_every = max(1, check_every)
        self._counter = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._counter += 1
        if self._counter % self.check_every:
            return
        # fresh flag from the DB (not from the cached object)
        from ...models import FineTuningJob
        if FineTuningJob.objects.filter(
            pk=self.job.pk, pause_requested=True
        ).exists():
            control.should_training_stop = True
            raise PauseRequested(f"Job {self.job.pk} paused at step {state.global_step}")
