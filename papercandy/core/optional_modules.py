class _SimulatedCOOTA(object):
    class Generator(object):
        def generate(self, size: int, *args, parse: bool = True):
            raise NotImplementedError


try:
    import coota as _coota
    _coota_is_available: bool = True
except ImportError:
    _coota = _SimulatedCOOTA
    _coota_is_available: bool = False


def coota_is_available() -> bool:
    return _coota_is_available
