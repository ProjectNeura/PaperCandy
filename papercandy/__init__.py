from papercandy.test import *
from papercandy.train import *
from papercandy.config import *
from papercandy.drawing import *
from papercandy.network import *
from papercandy.version import *
from papercandy.dataloader import *


CONFIG = None


def load_config(filename: Union[str, PathLike]):
    global CONFIG
    CONFIG = new_config(filename)
