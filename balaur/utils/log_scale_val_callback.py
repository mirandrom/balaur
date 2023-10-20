import pytorch_lightning as pl
import math

from typing import Any, Optional


class LogScaleValCheckCallback(pl.Callback):
    def __init__(self, accumulate_grad_batches: int = 1):
        self.accumulate_grad_batches = accumulate_grad_batches

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if math.log10(trainer.global_step+1) % 1 == 0:
            trainer.val_check_batch = (trainer.global_step + 1) * self.accumulate_grad_batches
