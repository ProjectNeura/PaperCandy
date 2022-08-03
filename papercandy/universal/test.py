from typing import Union, Any
from abc import abstractmethod


from papercandy import network as _network
from papercandy.universal import dataloader as _dl, config as _cfg


class Tester(object):
    def __init__(self, config: _cfg.Config, dataloader: _dl.Dataloader):
        self._nc: Union[_network.NetworkC, None] = None
        self._config: _cfg.Config = config
        self._dataloader: _dl.Dataloader = dataloader
        self._epoch: int = 0

    def _check_requirements(self) -> bool:
        return self._config is not None and self._dataloader is not None and self._nc is not None

    def _check_requirements_and_raise_exception(self):
        if not self._check_requirements():
            raise AttributeError("Tester hasn't been completely prepared.")

    def test(self, num_batches: int) -> list[_network.ResultCompound]:
        self._check_requirements_and_raise_exception()
        res_list = []
        local_epoch = 0
        for data in self._dataloader:
            if local_epoch >= num_batches:
                break
            if self._config.get_predefined("gpu_acceleration", True):
                data = data.gpu()
            o = self._test_one_batch(self._epoch, self._nc, data)
            res_list.append(_network.ResultCompound(data, o))
            local_epoch += 1
            self._epoch += 1
        return res_list

    @abstractmethod
    def _test_one_batch(self, epoch: int, network: Any, data: _network.DataCompound) -> Any:
        """
        :param epoch: global epoch
        :param network: network (not container)
        :param data: data batch in form of a data compound
        :return: network output
        """
        raise NotImplementedError
