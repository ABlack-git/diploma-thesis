import yaml
import logging
import logging.config
import importlib.resources as res

from dataclasses import fields
from typing import List
from omegaconf import OmegaConf
from im2gps.utils import Singelton
from im2gps.conf.im2gps.configschema import Config


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


def load_config(additional_configs: List[str] = None, schema=Config, base_cfg_package="im2gps.conf",
                base_cfg="config.yaml"):
    """
    In reality this function return omegaconf.DictConfig, but we ducktype it to Config
    :param additional_configs: list of additional configs
    :param schema
    :param base_cfg_package
    :param base_cfg

    :return: configuration
    """
    schema = OmegaConf.structured(schema)
    base_cfg = OmegaConf.create(res.read_text(base_cfg_package, base_cfg))
    if additional_configs is not None:
        others = []
        for cfg_path in additional_configs:
            others.append(OmegaConf.create(__read_config_file(cfg_path)))
        cfg = OmegaConf.merge(schema, base_cfg, *others)
    else:
        cfg = OmegaConf.merge(schema, base_cfg)

    return cfg


def __read_config_file(path) -> dict:
    with open(path, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
        return cfg


def configure_logging(verbosity: str, filename=None, package="im2gps.conf", conf_file="logging.yaml"):
    logging_conf = yaml.safe_load(res.read_text(package, conf_file))
    if filename is not None:
        logging_conf['handlers']['file']['filename'] = filename
    logging.config.dictConfig(logging_conf)
    if verbosity == 'disable':
        logging.disable(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(logging._nameToLevel[verbosity.upper()])
