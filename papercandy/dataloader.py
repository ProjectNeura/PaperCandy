from typing import Union
from typing_extensions import Self
from os import listdir as _listdir, PathLike


from papercandy import network as _network
from papercandy.universal import dataloader as _dataloader


Dataset = _dataloader.Dataset
Dataloader = _dataloader.Dataloader


class ExampleDataset(Dataset):
    def __init__(self, src: Union[str, PathLike]):
        self.src: Union[str, PathLike] = src
        self.file_list: list = _listdir(self.src)

    def __len__(self) -> int:
        return len(self.file_list)

    def cut(self, i: slice) -> Self:
        o = ExampleDataset(self.src)
        o.file_list = self.file_list
        return o

    def get(self, i: int) -> _network.DataCompound:
        with open("%s/%s" % (self.src, self.file_list[i]), "r") as f:
            return _network.DataCompound(self.file_list[i], f.read())
