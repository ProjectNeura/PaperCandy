from typing import Union, Any
from abc import abstractmethod

from papercandy import network as _network
from papercandy.universal import dataloader as _dl, config as _cfg


class TrainingMonitor(object):
    def on_updated(self, trainer, epoch: int, loss: float, input_data: _network.DataCompound, output: Any):
        """
        :param trainer: trainer object
        :type trainer: Trainer
        :param epoch: epoch number
        :param loss: loss value
        :param input_data: input data compound
        :param output: network output
        """
        pass

    def on_batch_finished(self, trainer, epoch: int):
        """
        :param trainer: trainer object
        :type trainer: Trainer
        :param epoch: epoch number
        """
        pass

    def on_finished(self, trainer, epoch: int):
        """
        :param trainer: trainer object
        :type trainer: Trainer
        :param epoch: epoch number
        """
        pass


class Trainer(object):
    def __init__(self, config: _cfg.Config = None, dataloader: _dl.Dataloader = None):
        self._nc: Union[_network.NetworkC, None] = None
        self._lfc: Union[_network.LossFunctionC, None] = None
        self._oc: Union[_network.OptimizerC, None] = None
        self._config: Union[_cfg.Config, None] = config
        self._dataloader: Union[_dl.Dataloader, None] = dataloader
        self._epoch: int = 0
        self.losses: list[float] = []

    def _check_requirements(self) -> bool:
        return self._config is not None and self._dataloader is not None and self._nc is not None \
               and self._lfc is not None and self._oc is not None

    def _check_requirements_and_raise_exception(self):
        if not self._check_requirements():
            raise AttributeError("Trainer hasn't been completely prepared.")

    def get_config(self) -> _cfg.Config:
        return self._config

    def get_dataloader(self) -> _dl.Dataloader:
        return self._dataloader

    def get_epoch(self) -> int:
        return self._epoch

    def set_network(self, nc: _network.NetworkC):
        if self._config.get_predefined("gpu_acceleration", True):
            nc = nc.gpu()
        self._nc = nc

    def get_network(self) -> Union[_network.NetworkC, None]:
        return self._nc

    def set_loss_function(self, lfc: _network.LossFunctionC):
        if self._config.get_predefined("gpu_acceleration", True):
            lfc = lfc.gpu()
        self._lfc = lfc

    def get_loss_function(self) -> Union[_network.LossFunctionC, None]:
        return self._lfc

    def set_optimizer(self, oc: _network.OptimizerC):
        if self._config.get_predefined("gpu_acceleration", True):
            oc = oc.gpu()
        self._oc = oc

    def get_optimizer(self) -> Union[_network.OptimizerC, None]:
        return self._oc

    def train(self, num_batches: int, monitor: TrainingMonitor = TrainingMonitor()):
        """
        NOTICE: When every time this method being called it'll start from the beginning of the dataloader.
        :param num_batches: batches limit
        :param monitor: training monitor
        :return:
        """
        self._check_requirements_and_raise_exception()
        local_epoch = 0
        for data in self._dataloader:
            if local_epoch >= num_batches:
                break
            if self._config.get_predefined("gpu_acceleration", True):
                data = data.gpu()
            self.train_one_batch(data, monitor)
            local_epoch += 1
            self._epoch += 1
        monitor.on_finished(self, self._epoch)

    def train_one_batch(self, data_batch: _network.DataCompound, monitor: TrainingMonitor):
        self._check_requirements_and_raise_exception()
        o, loss = self._train_one_batch(self._epoch, self._nc.get(), self._lfc.get(), self._oc.get(), data_batch)
        self.losses.append(loss)
        monitor.on_updated(self, self._epoch, loss, data_batch, o)
        monitor.on_batch_finished(self, self._epoch)

    @abstractmethod
    def _train_one_batch(self, epoch: int, network: Any, loss_function: Any, optimizer: Any,
                         data: _network.DataCompound) -> [Any, float]:
        """
        :param epoch: global epoch
        :param network: network (not container)
        :param loss_function: loss function (not container)
        :param optimizer: optimizer (not container)
        :param data: single data, not batched
        :return: network output, loss
        """
        raise NotImplementedError
