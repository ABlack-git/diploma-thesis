class Singelton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singelton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigRepo(metaclass=Singelton):
    def __init__(self):
        self._repo = {}

    def save(self, name, cfg):
        self._repo[name] = cfg

    def get(self, name):
        return self._repo[name]
