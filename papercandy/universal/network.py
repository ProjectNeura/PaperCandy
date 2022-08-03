from os import PathLike
from typing_extensions import Self
from abc import abstractmethod, ABC as _ABC
from numpy import sin as _sin, cos as _cos
from typing import Any, Union, Sequence, Iterable

from papercandy.universal import utils as _utils


class DataCompound(object):
    def __init__(self, data: Any, target: Any, d_type: type = Any):
        self.data: d_type = data
        self.target: d_type = target

    @abstractmethod
    def gpu(self) -> Self:
        raise NotImplementedError

    def unpack(self) -> [Any, Any]:
        return self.data, self.target


class Container(object):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def gpu(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Union[str, PathLike]):
        raise NotImplementedError


class LayerInfo(object):
    def __init__(self, width: int, height: int, angle: int, name: str, description: str = ""):
        self.width: int = width
        self.g_width: int = round(_cos(_utils.angle2radian(angle)) * width)
        self.height: int = height
        self.angle: int = angle
        self.name: str = name
        self.description: str = description

    def get(self) -> (int, int, int, int, str, str):
        return self.width, self.g_width, self.height, self.angle, self.name, self.description

    def parse_interval(self, interval: Union[int, float]) -> int:
        if interval >= 1:
            return interval
        return interval * self.g_width


class LayerInfoList(Sequence):
    def __init__(self, *layers: LayerInfo):
        self._layers: list[LayerInfo] = list(layers)

    def __iter__(self) -> Iterable[LayerInfo]:
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, item: Union[int, slice]) -> Union[LayerInfo, list[LayerInfo]]:
        return self._layers[item]

    def __call__(self, interval: Union[int, float]) -> (int, int):
        if interval < 0:
            raise ValueError("`interval` cannot be negative.")
        if len(self._layers) < 1:
            raise IndexError("There must be at least one layer.")
        canvas_width = 0
        canvas_height = 0
        display_interval = None
        for layer in self._layers:
            display_interval = layer.parse_interval(interval)
            canvas_width += layer.g_width + display_interval
            graph_height = layer.width * _sin(_utils.angle2radian(layer.angle)) + layer.height
            if graph_height > canvas_height:
                canvas_height = graph_height
        canvas_width -= display_interval
        return round(canvas_width), round(canvas_height)

    def __add__(self, other) -> Self:
        if not isinstance(other, LayerInfoList):
            raise TypeError("LayerInfoList can only be added to another LayerInfoList.")
        return LayerInfoList(*(self._layers + other.get_layers()))

    def reverse(self) -> Self:
        self._layers.reverse()
        return self

    def get_layers(self) -> list[LayerInfo]:
        return self._layers

    def append(self, layer_info: LayerInfo) -> Self:
        self._layers.append(layer_info)
        return self


class NetworkC(Container, _ABC):
    def __len__(self) -> int:
        return len(self.structure())

    @abstractmethod
    def structure(self) -> LayerInfoList[LayerInfo]:
        raise NotImplementedError


class LossFunctionC(Container, _ABC):
    pass


class OptimizerC(Container, _ABC):
    pass
