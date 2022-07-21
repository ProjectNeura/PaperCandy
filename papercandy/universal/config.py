from os import PathLike
from typing import Union, Any
from typing_extensions import Self
from multiprocessing import cpu_count as _cpu_count

_required_configs: dict = {
    "data": ("./data", str),
    "batch_size": ("16", int),
    "num_works": (str(_cpu_count()), int),
    "cuda_acceleration": ("False", bool),
    "device": ("0", int),
}


class Config(object):
    def __init__(self, filename: Union[str, PathLike]):
        self._filename: Union[str, PathLike] = filename
        self._config: dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self._config.keys()

    def put(self, key: str, value: str):
        if hasattr(self, key):
            raise KeyError(f"{key} is used.")
        setattr(self, key, value)
        self._config[key] = value

    def load(self) -> Self:
        with open(self._filename, "r") as f:
            lines = f.readlines()
            last = ""
            for line in lines:
                line = (last + line).replace(" ", "").replace("\n", "")
                if last != "" and last[-1] != ":" and line[0] != ":":
                    last += line
                    continue
                if line[-1] == ":":
                    last += line
                    continue
                sp = line.split(":")
                if len(sp) == 1:
                    last = line
                    continue
                key, val = sp[0], "".join(sp[1:])
                self.put(key, val)
                last = ""
            if last != "":
                raise SyntaxError("Config file didn't end.")
        return self

    def get(self, key: str, must_exists: bool = False, required_type: type = str) -> Union[Any, None]:
        if key not in self._config.keys():
            if must_exists:
                raise KeyError(f"No such configuration: \"{key}\".")
            return None
        val = self._config[key]
        return required_type(val)

    def get_predefined(self, key: str, must_exists: bool = False) -> Union[Any, None]:
        required_type = str
        if key in _required_configs.keys():
            required_type = _required_configs[key][1]
        return self.get(key, must_exists, required_type)


def new_config(filename: Union[str, PathLike]) -> Config:
    config = Config(filename).load()
    # Check required configurations
    for req_cfg_key in _required_configs.keys():
        if req_cfg_key not in config:
            config.put(req_cfg_key, _required_configs[req_cfg_key][0])
    return config
