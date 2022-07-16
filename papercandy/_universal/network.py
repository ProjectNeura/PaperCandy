from os import PathLike
from typing import Any, Union
from abc import abstractmethod, ABC
from typing_extensions import Self


class DataCompound(object):
    def __init__(self, data: Any, target: Any, d_type: type = Any):
        self.data: d_type = data
        self.target: d_type = target


class Container(object):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def cuda(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Union[str, PathLike]):
        raise NotImplementedError


class NetworkC(Container, ABC):
    pass


class LossFunctionC(Container, ABC):
    pass


class OptimizerC(Container, ABC):
    pass
