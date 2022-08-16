from os import PathLike
from typing import Union, Any
from typing_extensions import Self


from papercandy.core.singleton import singleton


def Bool(s: str) -> bool:
    return s.lower() == "true"


_required_configs: dict = {
    "gpu_acceleration": ("False", Bool),
    "device": ("0", int),
}


class Config(object):
    def __init__(self):
        self._config: dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self._config.keys()

    def set(self, key: str, value: str):
        if hasattr(self, key) and key not in self._config.keys():
            raise KeyError(f"{key} is used.")
        setattr(self, key, value)
        self._config[key] = value

    def check_required_configs(self, must_exist: bool = False) -> Self:
        """
        Check for each required configuration. When not found, if `must_exist` is True throw an exception,
            otherwise, fill with the default value.
        :param must_exist: whether to raise an error when a certain required configuration is not included
        :return: self
        """
        for req_cfg_key in _required_configs.keys():
            if req_cfg_key not in self:
                if must_exist:
                    raise KeyError(f"Configuration should include key \"{req_cfg_key}\".")
                self.set(req_cfg_key, _required_configs[req_cfg_key][0])
        return self

    def loads(self, lines: list[str]) -> Self:
        """
        Load the configuration content.
        :param lines: content in lines
        :return: self
        """
        last = ""
        for line in lines:
            if line == "":
                continue
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
            self.set(key, val)
            last = ""
        if last != "":
            raise SyntaxError("Config file didn't end.")
        return self

    def load(self, filename: Union[str, PathLike]) -> Self:
        """
        Load the configuration file.
        :param filename: filename
        :return: self
        """
        with open(filename, "r") as f:
            return self.loads(f.readlines())

    def get(self, key: str, must_exist: bool = False, required_type: type = str, default_val: Any = None) \
            -> Union[Any, None]:
        """
        Get the configuration value with the key.
        :param key: the key
        :param must_exist: whether to raise an error when the key is not found
        :param required_type: expected type of the value
            NOTICE: This will be directly called to convert the type, which bool type doesn't support.
                Therefor we provide an alternative method (pretended to be a type-class)
                "papercandy.core.config.Bool" to solve this problem.
        :param default_val: the default value to return when the key is not found
            NOTICE: This only works when `must_exist` is False.
        :return: the value
        """
        if key not in self._config.keys():
            if must_exist:
                raise KeyError(f"No such configuration: \"{key}\".")
            return default_val
        val = self._config[key]
        return val if required_type == str else required_type(val)

    def get_predefined(self, key: str, must_exists: bool = False) -> Union[Any, None]:
        required_type = str
        if key in _required_configs.keys():
            required_type = _required_configs[key][1]
        return self.get(key, must_exists, required_type)


def new_config(filename: Union[str, PathLike]) -> Config:
    return Config().load(filename).check_required_configs()


@singleton
class ConfigContainer(object):
    def __init__(self):
        self.DEFAULT: Config = Config().loads([""]).check_required_configs()
        self.CURRENT: Config = Config().loads([""]).check_required_configs()
