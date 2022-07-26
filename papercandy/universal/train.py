from typing import Union, Any
from abc import abstractmethod

from papercandy import network as _network
from papercandy.universal import dataloader as _dl, config as _cfg


class TrainingMonitor(object):
    def on_start(self, epoch: int, loss_function: _network.LossFunctionC): pass

    def on_batch_start(self, epoch: int, loss_function: _network.LossFunctionC): pass

    def on_updated(self, epoch: int, loss: float, input_data: Any, output_data: Any): pass

    def on_batch_finish(self, epoch: int, loss_function: _network.LossFunctionC): pass

    def on_finish(self, epoch: int, loss_function: _network.LossFunctionC): pass


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
        if self._config.get("cuda_acceleration", must_exists=True, required_type=bool):
            nc = nc.cuda()
        self._nc = nc

    def get_network(self) -> Union[_network.NetworkC, None]:
        return self._nc

    def set_loss_function(self, lfc: _network.LossFunctionC):
        if self._config.get("cuda_acceleration", must_exists=True, required_type=bool):
            lfc = lfc.cuda()
        self._lfc = lfc

    def get_loss_function(self) -> Union[_network.LossFunctionC, None]:
        return self._lfc

    def set_optimizer(self, oc: _network.OptimizerC):
        if self._config.get("cuda_acceleration", must_exists=True, required_type=bool):
            oc = oc.cuda()
        self._oc = oc

    def get_optimizer(self) -> Union[_network.OptimizerC, None]:
        return self._oc

    def train(self, num_batches: int = 1, monitor: TrainingMonitor = TrainingMonitor()):
        monitor.on_start(self.get_epoch(), self.get_loss_function())
        epoch = 0
        for data in self._dataloader:
            if epoch >= num_batches:
                break

            self.train_one_batch(data, monitor)
        self._epoch += epoch
        monitor.on_finish(self.get_epoch(), self.get_loss_function())

    def train_one_batch(self, data_batch: list[_network.DataCompound], monitor: TrainingMonitor):
        monitor.on_batch_start(self.get_epoch(), self.get_loss_function())
        for d in data_batch:
            o, loss = self.train_single(self.get_epoch(), self.get_network().get(), self.get_loss_function().get(),
                                        self.get_optimizer().get(), d)
            self.losses.append(loss)
            monitor.on_updated(self.get_epoch(), loss, d, o)
        monitor.on_batch_finish(self.get_epoch(), self.get_loss_function())

    @abstractmethod
    def train_single(self, epoch: int, network: Any, loss_function: Any, optimizer: Any, data: _network.DataCompound) \
            -> [Any, float]:
        """
        :param epoch: global epoch
        :param network: network (not container)
        :param loss_function: loss function (not container)
        :param optimizer: optimizer (not container)
        :param data: single data, not batched
        :return: network output, loss
        """
        raise NotImplementedError
