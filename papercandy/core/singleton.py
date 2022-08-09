from multiprocessing import Lock as _Lock


def singleton(cls):
    _instance: dict = {}
    _lock: _Lock = _Lock()

    def _singleton(*args, **kwargs):
        if cls in _instance:
            return _instance[cls]
        _lock.acquire()
        if cls not in _instance:    # double check
            _instance[cls] = cls(*args, **kwargs)
        _lock.release()
        return _instance[cls]
    return _singleton
