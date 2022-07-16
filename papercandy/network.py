from os import PathLike
from typing import Union

from typing_extensions import Self

from torch.nn import Module as _Module
from torch import Tensor as _Tensor, save as _save


from _universal import network as _network


class DataCompound(_network.DataCompound):
    def __init__(self, data: _Tensor, target: _Tensor):
        super(DataCompound, self).__init__(data, target, d_type=_Tensor)


class NetworkC(_network.NetworkC):
    def __init__(self, network: _Module):
        self._network: _Module = network

    def cuda(self) -> Self:
        self._network = self._network.cuda()
        return self

    def get(self) -> _Module:
        return self._network

    def save(self, path: Union[str, PathLike]):
        _save(self.get(), path)


class LossFunctionC(_network.LossFunctionC):
    def __init__(self, loss_function: _Module):
        self._loss_function: _Module = loss_function

    def cuda(self) -> Self:
        self._loss_function = self._loss_function.cuda()
        return self

    def get(self) -> _Module:
        return self._loss_function

    def save(self, path: Union[str, PathLike]):
        _save(self.get(), path)


class OptimizerC(_network.OptimizerC):
    def __init__(self, optimizer: _Module):
        self._optimizer: _Module = optimizer

    def cuda(self) -> Self:
        self._optimizer = self._optimizer.cuda()
        return self

    def get(self) -> _Module:
        return self._optimizer

    def save(self, path: Union[str, PathLike]):
        _save(self.get(), path)
