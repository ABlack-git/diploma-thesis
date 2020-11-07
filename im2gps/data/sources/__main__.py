import hydra

from im2gps.data.sources.config import DSConfig
from im2gps.configutils import ConfigRepo
from im2gps.data.sources.flickr import collect_photos


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DSConfig):
    cr = ConfigRepo()
    cr.save(DSConfig.__name__, cfg)
    collect_photos()


if __name__ == '__main__':
    main()
