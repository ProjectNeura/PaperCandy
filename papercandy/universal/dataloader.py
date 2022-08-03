from copy import copy as _copy
from math import ceil as _ceil
from abc import abstractmethod
from typing import Iterator, Union
from typing_extensions import Self
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
    def cut(self, i: slice) -> Self:
        raise NotImplementedError

    @abstractmethod
    def get(self, i: int) -> _network.DataCompound:
        raise NotImplementedError

    def __getitem__(self, item: Union[int, slice]) -> Union[_network.DataCompound, Self]:
        if isinstance(item, int):
            return self.get(item)
        if isinstance(item, slice):
            return self.cut(item)


class UniversalDataloader(Iterator):
    def __init__(self, dataset: Dataset):
        self.dataset: Dataset = dataset

    def __getitem__(self, item: slice) -> Self:
        """
        In this method, all indexes are item index instead of batch indexes.
        :param item: an index slice
        :return: another edited object
        """
        if not isinstance(item, slice):
            raise TypeError("`item` must be a slice.")
        o = _copy(self)
        o.dataset = self.dataset[item]
        return o

    def __iter__(self) -> Self:
        return self

    def __call__(self) -> _network.DataCompound:
        return next(self)

    @abstractmethod
    def __next__(self) -> _network.DataCompound:
        raise NotImplementedError

    def load_next_batch(self) -> _network.DataCompound:
        return next(self)

    @abstractmethod
    def load_batch(self, start: int, stop: int) -> _network.DataCompound:
        raise NotImplementedError


class Dataloader(UniversalDataloader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, num_works: int = 1):
        if batch_size < 1:
            raise ValueError("`batch_size` must be at least 1.")
        if num_works < 1:
            raise ValueError("`num_works` must be at least 1.")
        if num_works > batch_size:
            raise ValueError("`num_works` must be less than `batch_size`.")

        super(Dataloader, self).__init__(dataset)
        self._batch_size: int = batch_size
        self._num_works: int = num_works
        self._iter_pointer: int = 0
        self._pool: Union[_Pool, None] = None if num_works < 2 else _Pool(num_works)

    def __iter__(self) -> Iterator:
        return self

    def __len__(self) -> int:
        return _ceil(len(self.dataset) / self._batch_size)

    def __getitem__(self, item: Union[int, slice]) -> Union[_network.DataCompound, Self]:
        """
        In this method, all indexes are batch index instead of item indexes.
        :param item: int for a data compound(batched), slice for cutting
        :return: a data compound(batched) or another edited object
        """
        if isinstance(item, int):
            return self._load_batch(self.dataset, self._batch_size, self._batch_size * item)
        if isinstance(item, slice):
            item = self._multiply_slice(item)
            return super(Dataloader, self).__getitem__(item)

    def __next__(self) -> _network.DataCompound:
        rest = len(self.dataset) - self._iter_pointer
        if rest <= 0:
            raise StopIteration
        batch_size = self._batch_size if rest > self._batch_size else rest
        try:
            return self.load_batch(self._iter_pointer, self._iter_pointer + batch_size)
        finally:
            self._iter_pointer += batch_size

    def _multiply_slice(self, s: slice) -> slice:
        start, stop, step = None, None, None
        if s.start is not None:
            start = s.start * self._batch_size
        if s.stop is not None:
            stop = s.stop * self._batch_size
        if s.step is not None:
            step = s.step * self._batch_size
        return slice(start, stop, step)

    def move_iter_pointer(self, n: int) -> Self:
        self._iter_pointer += n
        return self

    def load_batch(self, start: int, stop: int) -> _network.DataCompound:
        res_list = []
        size = stop - start
        if self._num_works == 1:
            res_list += self._load_batch(self.dataset, size, start)
        else:
            self._pool = _Pool(self._num_works)
            spw = int(size / self._num_works)
            rest = size % self._num_works
            rest = spw if rest == 0 else rest
            work_res_list = []
            for i in range(self._num_works - 1):
                work_res_list.append(
                    self._pool.apply_async(self._load_batch, args=(self.dataset, spw, start + i * spw)))
            work_res_list.append(
                self._pool.apply_async(self._load_batch, args=(self.dataset, rest, start + size - rest)))
            self._pool.close()
            self._pool.join()
            for work_res in work_res_list:
                res_list += work_res.get()
        return self.combine_batch(res_list)

    @staticmethod
    def _load_batch(dataset: Dataset, size: int, base: int = 0) -> list[_network.DataCompound]:
        res_list = []
        for i in range(size):
            res_list.append(dataset[base + i])
        return res_list

    @abstractmethod
    def combine_batch(self, data_batch: list[_network.DataCompound]) -> _network.DataCompound:
        raise NotImplementedError
