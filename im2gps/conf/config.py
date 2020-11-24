from dataclasses import dataclass, fields

from omegaconf import MISSING

from im2gps.conf.data.config import DataConfig
from im2gps.conf.imret.config import ImRetConfig


@dataclass
class Config:
    data: DataConfig = MISSING
    imret: ImRetConfig = MISSING


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


def save_configs(cfg: Config):
    conf_repo = ConfigRepo()
    conf_repo.save(Config.__name__, cfg)
    for field in fields(Config):
        conf_repo.save(field.type.__name__, getattr(cfg, field.name))


def load_config(name):
    conf_repo = ConfigRepo()
    return conf_repo.get(name)
