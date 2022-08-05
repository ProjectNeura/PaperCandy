from typing import Any
from torch.nn import Module as _Module

from papercandy import network as _network
from papercandy.universal import train as _train


class Trainer(_train.Trainer):
    def _train_one_batch(self, epoch: int, network: _Module, loss_function: _Module, optimizer: _Module,
                         data: _network.DataCompound) -> [Any, float]:
        optimizer.zero_grad()
        output = network(data.data)
        current_loss = loss_function(output, data.target)
        current_loss.backward()
        optimizer.step()
        return output, current_loss.item()


class TrainingMonitor(_train.TrainingMonitor):
    def on_updated(self, trainer: Trainer, epoch: int, loss: float, result: _network.ResultCompound): pass

    def on_batch_finished(self, trainer: Trainer, epoch: int): pass

    def on_finished(self, trainer: Trainer, epoch: int): pass


class TrainerUtils(_train.TrainerUtils):
    pass
