import click
import logging

from mongoengine import connect
from im2gps.conf import config
from im2gps.cli.im2gps import data, index, localisation, stats
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


@click.group()
@click.option("--config-path", "-c", multiple=True, default=None, required=False,
              help='Specify additional application configs, can be used multiple times to provide multiple configs')
@click.option("--no-db", required=False, is_flag=True, default=False, help="Don't try to connect to database")
@click.option("-v", "--verbosity",
              type=click.Choice(['disable', 'debug', 'info', 'warn', 'error', 'critical'], case_sensitive=False),
              default='info', help="Provide verbosity level")
@click.option("--show-config", is_flag=True, default=False, help='Print current configuration to stdout')
def entry_point(config_path, no_db, verbosity, show_config):
    config.configure_logging(verbosity)
    log.debug("Loading configuration")
    cfg = config.load_config(config_path)
    config.save_configs(cfg)
    if not no_db:
        log.debug("Connecting to database")
        connect(db=cfg.db.database, host=cfg.db.host, port=cfg.db.port)
    if show_config:
        print(OmegaConf.to_yaml(cfg))


entry_point.add_command(data.data)
entry_point.add_command(localisation.localisation)
entry_point.add_command(index.index)
entry_point.add_command(stats.stats)

if __name__ == '__main__':
    entry_point()
