import hydra

from im2gps.conf.config import Config, save_configs
from im2gps.imret.cirtorchnet import get_descriptors


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: Config):
    save_configs(cfg)
    get_descriptors()


if __name__ == '__main__':
    main()
