from math import ceil as _ceil
from abc import abstractmethod
from typing_extensions import Self
from typing import Iterator
from multiprocessing import Pool as _Pool


from papercandy import network as _network  # To make the DataCompound's inner type adaptable


class Dataset(object):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> _network.DataCompound:
        raise NotImplementedError


class UniversalDataloader(Iterator):
    def __iter__(self) -> Self:
        return self

    def __call__(self) -> list[_network.DataCompound]:
        return next(self)

    @abstractmethod
    def __next__(self) -> list[_network.DataCompound]:
        raise NotImplementedError

    @abstractmethod
    def load_batch(self, size: int) -> list[_network.DataCompound]:
        raise NotImplementedError


class Dataloader(UniversalDataloader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, num_works: int = 1):
        if batch_size < 1:
            raise ValueError("`batch_size` must be at least 1.")
        if num_works < 1:
            raise ValueError("`num_works` must be at least 1.")
        if num_works > batch_size:
            raise ValueError("`num_works` must be less than `batch_size`.")

        self.dataset: Dataset = dataset
        self._batch_size: int = batch_size
        self._num_works: int = num_works
        self._iter_pointer: int = 0
        self._pool: _Pool = _Pool(num_works)

    def __iter__(self) -> Iterator:
        return self

    def __len__(self) -> int:
        return _ceil(len(self.dataset) / self._batch_size)

    def __next__(self) -> list[_network.DataCompound]:
        rest = len(self.dataset) - self._iter_pointer
        if rest <= 0:
            raise StopIteration
        batch_size = self._batch_size if rest > self._batch_size else rest
        data = self.load_batch(batch_size)
        self._iter_pointer += batch_size
        return data

    def move_iter_pointer(self, n: int) -> Self:
        self._iter_pointer += n
        return self

    def load_batch(self, size: int) -> list[_network.DataCompound]:
        res_list = []
        if self._num_works == 1:
            res_list += self._load_batch(self.dataset, self._iter_pointer, size)
        else:
            spw = int(size / self._num_works)
            rest = size % self._num_works
            rest = spw if rest == 0 else rest
            work_res_list = []
            for i in range(self._num_works - 1):
                work_res_list.append(
                    self._pool.apply_async(self._load_batch, args=(self.dataset, self._iter_pointer, spw, i * spw)))
            work_res_list.append(
                self._pool.apply_async(self._load_batch, args=(self.dataset, self._iter_pointer, rest, size - rest)))
            self._pool.close()
            self._pool.join()
            for work_res in work_res_list:
                res_list += work_res.get()
        return res_list

    @staticmethod
    def _load_batch(dataset: Dataset, iter_pointer: int, size: int, base: int = 0) -> list[_network.DataCompound]:
        iter_pointer += base
        res_list = []
        for i in range(size):
            res_list.append(dataset[iter_pointer + i])
        return res_list
