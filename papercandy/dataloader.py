from typing import Union
from os import listdir as _listdir, PathLike


from universal import dataloader as _dataloader


Dataset = _dataloader.Dataset
Dataloader = _dataloader.Dataloader


class ExampleDataset(Dataset):
    def __init__(self, src: Union[str, PathLike]):
        self.src: Union[str, PathLike] = src
        self.file_list: list = _listdir(self.src)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> dict:
        with open("%s/%s" % (self.src, self.file_list[index]), "r") as f:
            return {self.file_list[index]: f.read()}


if __name__ == '__main__':
    dl = Dataloader(ExampleDataset("../../data"))
    for d in dl:
        print(d)
