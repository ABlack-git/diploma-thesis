import hydra
from mongoengine import connect

from im2gps.conf.config import Config, save_configs
from im2gps.data.sources.flickr import collect_photos_metadata, download_photos


@hydra.main(config_path='../../conf', config_name='config')
def main(cfg: Config):
    data_conf = cfg.data
    save_configs(cfg)
    connect(db=data_conf.db.database, host=data_conf.db.host, port=data_conf.db.port)
    if data_conf.download == "meta":
        collect_photos_metadata()
    elif data_conf.download == "photos":
        download_photos()
    else:
        raise ValueError(f"Unknown download type: {data_conf.download}")


if __name__ == '__main__':
    main()
