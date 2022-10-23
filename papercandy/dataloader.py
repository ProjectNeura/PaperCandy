from typing import Union
from abc import ABCMeta
from typing_extensions import Self
from torch import Tensor as _Tensor
from os import listdir as _listdir, PathLike


from papercandy import network as _network
from papercandy.core import dataloader as _dataloader


Dataset = _dataloader.Dataset


class Dataloader(_dataloader.Dataloader):
    @staticmethod
    def combine_batch(data_batch: list[_network.DataCompound]) -> _network.DataCompound:
        d_list, t_list = [], []
        for dc in data_batch:
            d, t = dc.unpack()
            d, t = d.tolist(), t.tolist()
            d_list.append(d)
            t_list.append(t)
        return _network.DataCompound(_Tensor(d_list), _Tensor(t_list))


class PreprocessedDataloader(_dataloader.PreprocessedDataloader, metaclass=ABCMeta):
    @staticmethod
    def combine_batch(data_batch: list[_network.DataCompound]) -> _network.DataCompound:
        return Dataloader.combine_batch(data_batch)


class ExampleDataset(Dataset):
    def __init__(self, src: Union[str, PathLike]):
        self.src: Union[str, PathLike] = src
        self.file_list: list = _listdir(self.src)

    def __len__(self) -> int:
        return len(self.file_list)

    def cut(self, i: slice) -> Self:
        o = ExampleDataset(self.src)
        o.file_list = self.file_list[i]
        return o

    def get(self, i: int) -> _network.DataCompound:
        with open("%s/%s" % (self.src, self.file_list[i]), "r") as f:
            return _network.DataCompound(_Tensor([int(self.file_list[i])]), _Tensor(eval(f.read())))
