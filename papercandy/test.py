from typing import Any
from torch import no_grad as _no_grad

from papercandy import network as _network
from papercandy.core import test as _test


class Tester(_test.Tester):
    def _test_one_batch(self, epoch: int, network: Any, data: _network.DataCompound) -> Any:
        with _no_grad():
            return network(data.data)
