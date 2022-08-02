from os import PathLike
from typing import Union
from copy import copy as _copy
from typing_extensions import Self
from torch.nn import Module as _Module
from torch.nn import modules as _modules
from torch.optim import Optimizer as _Optimizer
from torch import Tensor as _Tensor, save as _save

from papercandy.universal import network as _network


class DataCompound(_network.DataCompound):
    def __init__(self, data: _Tensor, target: _Tensor):
        super(DataCompound, self).__init__(data, target, d_type=_Tensor)

    def unpack(self) -> [_Tensor, _Tensor]:
        return super(DataCompound, self).unpack()

    def cuda(self) -> Self:
        o = _copy(self)
        o.data, o.target = self.data.cuda(), self.target.cuda()


LayerInfo = _network.LayerInfo
LayerInfoList = _network.LayerInfoList


class NetworkC(_network.NetworkC):
    def __init__(self, network: _Module):
        self._network: _Module = network

    def cuda(self) -> Self:
        o = _copy(self)
        o._network = self._network.cuda()
        return self

    def get(self) -> _Module:
        return self._network

    def save(self, path: Union[str, PathLike]):
        _save(self.get(), path)

    def structure(self) -> LayerInfoList[LayerInfo]:
        lil = LayerInfoList()
        for val in self._network.__dict__["_modules"].values():
            if isinstance(val, _modules.Sequential):
                lil += self.reflect_sequential(val)
            layer_info = self.layer2layer_info(val)
            if layer_info is not None:
                lil.append(layer_info)
        return lil

    def reflect_sequential(self, seq: _modules.Sequential) -> LayerInfoList[LayerInfo]:
        lil = LayerInfoList()
        for layer in seq:
            if isinstance(layer, _modules.Sequential):
                lil += self.reflect_sequential(layer)
            layer_info = self.layer2layer_info(layer)
            if layer_info is not None:
                lil.append(layer_info)
        return lil

    @staticmethod
    def layer2layer_info(layer: _Module) -> Union[LayerInfo, None]:
        angle = 23
        if isinstance(layer, _modules.conv._ConvNd):
            return LayerInfo(1200, 1200, angle, "Conv", f"{layer.in_channels}(in)x{layer.out_channels}(out)")
        if isinstance(layer, _modules.pooling._MaxPoolNd):
            return LayerInfo(800, 800, angle, "Pooling")
        if isinstance(layer, _modules.dropout._DropoutNd):
            return LayerInfo(800, 800, angle, "Dropout")
        if isinstance(layer, _modules.batchnorm._BatchNorm):
            return LayerInfo(800, 800, angle, "BatchNorm")
        if isinstance(layer, _modules.ReLU):
            return LayerInfo(400, 400, angle, "Relu")
        return None


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
    def __init__(self, optimizer: _Optimizer):
        self._optimizer: _Optimizer = optimizer

    def cuda(self) -> Self:
        return self

    def get(self) -> _Optimizer:
        return self._optimizer

    def save(self, path: Union[str, PathLike]):
        _save(self.get(), path)
