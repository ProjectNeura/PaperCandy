from os import PathLike
from typing import Union
from torch.cuda import is_available as _is_available

from papercandy.core import config as _config

Config = _config.Config
Bool = _config.Bool
CONFIG = _config.ConfigContainer


def new_config(filename: Union[str, PathLike]) -> Config:
    cfg = _config.new_config(filename)
    if not _is_available():
        cfg.set("gpu_acceleration", "False")
    return cfg
