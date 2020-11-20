import hydra
from mongoengine import connect

from im2gps.data.sources.config import DSConfig
from im2gps.configutils import ConfigRepo
from im2gps.data.sources.flickr import collect_photos_metadata, download_photos


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DSConfig):
    cr = ConfigRepo()
    cr.save(DSConfig.__name__, cfg)
    connect(db=cfg.db.database, host=cfg.db.host, port=cfg.db.port)
    if cfg.app.download == "meta":
        collect_photos_metadata()
    elif cfg.app.download == "photos":
        download_photos()
    else:
        raise ValueError(f"Unknown download type: {cfg.app.download}")


if __name__ == '__main__':
    main()
