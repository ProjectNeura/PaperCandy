from os import PathLike
from typing import Union
from torch.cuda import is_available as _is_available

from papercandy.universal import config as _config

Config = _config.Config
Bool = _config.Bool


def new_config(filename: Union[str, PathLike]) -> Config:
    cfg = _config.new_config(filename)
    if not _is_available():
        cfg.put("gpu_acceleration", "False")
    return cfg
