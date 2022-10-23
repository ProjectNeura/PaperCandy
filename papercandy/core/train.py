from typing import Union, Any
from copy import copy as _copy
from abc import abstractmethod, ABCMeta

from papercandy.core import network as _network, dataloader as _dl, config as _cfg


class TrainingMonitor(object):
    def on_updated(self, trainer, epoch: int, loss: float, result: _network.ResultCompound):
        """
        :param trainer: trainer object
        :type trainer: Trainer
        :param epoch: epoch number
        :param loss: loss value
        :param result: result compound
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


class Trainer(object, metaclass=ABCMeta):
    def __init__(self, dataloader: _dl.Dataloader):
        self._nc: Union[_network.NetworkC, None] = None
        self._lfc: Union[_network.LossFunctionC, None] = None
        self._oc: Union[_network.OptimizerC, None] = None
        self._config: _cfg.Config = _cfg.CONFIG().CURRENT
        self._dataloader: _dl.Dataloader = dataloader
        self._epoch: int = 0
        self.losses: list[float] = []

    def _check_requirements(self) -> bool:
        return self._config is not None and self._dataloader is not None and self._nc is not None \
               and self._lfc is not None and self._oc is not None

    def _check_requirements_or_raise_err(self):
        if not self._check_requirements():
            raise AttributeError("Trainer hasn't been completely prepared. Mostly due to some required attributes "
                                 "being None.")

    def get_config(self) -> _cfg.Config:
        return self._config

    def get_dataloader(self) -> _dl.Dataloader:
        return self._dataloader

    def get_epoch(self) -> int:
        return self._epoch

    def set_network(self, nc: _network.NetworkC):
        if self._config.get_predefined("gpu_acceleration"):
            nc = nc.gpu()
        self._nc = nc

    def get_network(self) -> Union[_network.NetworkC, None]:
        return self._nc

    def set_loss_function(self, lfc: _network.LossFunctionC):
        if self._config.get_predefined("gpu_acceleration"):
            lfc = lfc.gpu()
        self._lfc = lfc

    def get_loss_function(self) -> Union[_network.LossFunctionC, None]:
        return self._lfc

    def set_optimizer(self, oc: _network.OptimizerC):
        if self._config.get_predefined("gpu_acceleration"):
            oc = oc.gpu()
        self._oc = oc

    def get_optimizer(self) -> Union[_network.OptimizerC, None]:
        return self._oc

    def train(self, num_batches: int, monitor: TrainingMonitor = TrainingMonitor()):
        """
        Train the network by traversing the dataloader until epoch reaches either `num_batches` or the length of the
            dataloader. Fill `num_batches` with an integer that is larger than the length of the dataloader if you want
            to go through the whole dataset.
        NOTICE: When every time this method being called it'll start from the beginning of the dataloader.
        :param num_batches: the maximum number of batches
        :param monitor: training monitor
        """
        self._check_requirements_or_raise_err()
        local_epoch = 0
        for data in self._dataloader:
            if local_epoch >= num_batches:
                break
            if self._config.get_predefined("gpu_acceleration"):
                data = data.gpu()
            o, loss = self._train_one_batch(self._epoch, self._nc.get(), self._lfc.get(), self._oc.get(), data)
            self.losses.append(loss)
            monitor.on_updated(self, self._epoch, loss, _network.ResultCompound(data, o))
            monitor.on_batch_finished(self, self._epoch)
            local_epoch += 1
            self._epoch += 1
        monitor.on_finished(self, self._epoch)

    @abstractmethod
    def _train_one_batch(self, epoch: int, network: Any, loss_function: Any, optimizer: Any,
                         data: _network.DataCompound) -> [Any, float]:
        """
        :param epoch: global epoch
        :param network: network (not container)
        :param loss_function: loss function (not container)
        :param optimizer: optimizer (not container)
        :param data: data batch
        :return: {network output, loss}
        """
        raise NotImplementedError


class TrainerDataUtils(object, metaclass=ABCMeta):
    @staticmethod
    def vibration(losses: list) -> bool:
        if len(losses) == 2:
            return losses[1] > losses[0]
        if all(x > y for x, y in zip(losses, losses[1:])):
            return False
        new_losses = []
        for i in range(0, len(losses), 2):
            new_losses.append(0.5 * sum(losses[i: i + 2]))
        return TrainerDataUtils.vibration(new_losses)

    @staticmethod
    def analyse(trainer: Trainer) -> dict:
        length = len(trainer.losses)
        if length < 20:
            raise RuntimeError("Samples not enough to analyse.")
        min_loss = min(trainer.losses)
        max_loss = max(trainer.losses)
        average_loss = sum(trainer.losses) / length
        vibration = TrainerDataUtils.vibration(trainer.losses[:round(length * 0.1)])
        print(f"Loss: {round(min_loss, 2)}~{round(max_loss, 2)}({round(average_loss, 2)} on average)\n"
              f"Vibration: {vibration}")
        return {
            "min_loss": min_loss,
            "max_loss": max_loss,
            "average_loss": average_loss,
            "vibration": vibration,
        }

    @staticmethod
    def limit_losses(trainer: Trainer, n: float) -> Trainer:
        trainer = _copy(trainer)
        trainer.losses = [i for i in trainer.losses if i <= n]
        return trainer

    @staticmethod
    def scale_losses(trainer: Trainer, ratio: float) -> Trainer:
        """
        Uniformly remove a certain proportion of the loss data.
        :param trainer: the object to be operated
        :param ratio: the proportion
        :return: another edited object
        """
        trainer = _copy(trainer)
        if ratio > 1:
            raise ValueError("Not expandable, which means `ratio` cannot be bigger than 1.")
        if ratio <= 0:
            raise ValueError("`ratio` cannot be negative.")
        ratio = ratio * 2
        if ratio < 1:
            ratio = 1 / ratio
        ratio = round(ratio)
        g_size = ratio + 1
        if ratio < 1:
            for i in range(len(trainer.losses) // g_size):
                trainer.losses.pop(ratio - i + i * g_size)
        else:
            trainer.losses = trainer.losses[::g_size]
        return trainer

    @staticmethod
    def remove_losses(trainer: Trainer, n: int) -> Trainer:
        """
        Uniformly remove a certain amount of the loss data.
        :param trainer: the object to be operated
        :param n: the amount
        :return: another edited object
        """
        return TrainerDataUtils.scale_losses(trainer, n / len(trainer.losses))
