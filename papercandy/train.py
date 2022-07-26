from typing import Any
from torch.nn import Module as _Module

from papercandy import network as _network
from papercandy.universal import train as _train


TrainingMonitor = _train.TrainingMonitor


class Trainer(_train.Trainer):
    def train_single(self, epoch: int, network: _Module, loss_function: _Module, optimizer: _Module,
                     data: _network.DataCompound) -> [Any, float]:
        optimizer.zero_grad()
        output = network(data.data)
        current_loss = loss_function(output, data.target)
        current_loss.backward()
        optimizer.step()
        return output, current_loss.item()
