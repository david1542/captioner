from typing import Any

import pytorch_lightning as pl
from torch import nn

from torch.optim import Optimizer


class PLWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    # PyTorch Lightning methods
    def training_step(self, batch, *args):
        loss = self.model.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, *args):
        loss = self.model.compute_loss(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, *args):
        loss = self.model.compute_loss(batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
