import click
import json
import logging
import im2gps.services.localisation as loc
from im2gps.conf.config import ConfigRepo, Config

log = logging.getLogger(__name__)


@click.group()
def localisation():
    pass


@localisation.command()
@click.option("--db-path", default=None, type=str, help="Path to database")
@click.option("--test-q-path", default=None, type=str, help="Path to test set")
@click.option("-k", default=-1, help="Number of nearest neighbours")
@click.option("--gpu-enabled", is_flag=True, default=False,
              help="With this flag gpu will be used to find nearest neighbours")
@click.option("--gpu-id", default=-1, help="Gpu id")
@click.option("--loc-type", type=click.Choice(['1nn', 'kde', 'avg']))
@click.option("--sigma", type=float, default=None)
@click.option("-m", type=float, default=None)
@click.option("--avg-type", type=click.Choice(['weighted', 'regular']), default=None)
def test_localization(db_path, test_q_path, k, gpu_enabled, gpu_id, loc_type, sigma, m, avg_type):
    cfg: Config = ConfigRepo().get(Config.__name__)
    if db_path is None:
        db_path = cfg.data.datasets.train
    if test_q_path is None:
        test_q_path = cfg.data.datasets.test_queries
    assert isinstance(k, int) and k > 0, "k should be integer greater than 0"
    if gpu_enabled:
        if gpu_id == -1:
            gpu_id = cfg.properties.gpu_id
        assert isinstance(gpu_id, int) and gpu_id >= 0, "gpu_id should be integer greater or equal than 0"
    kwargs = {}
    if sigma is not None:
        kwargs['sigma'] = sigma
    if m is not None:
        kwargs['m'] = m
    if avg_type is not None:
        kwargs['avg_type'] = avg_type
    log.info(f"Starting localization test with following parameters: db_path={db_path}, "
             f"test_q_path={test_q_path}, k={k}, gpu_enabled={gpu_enabled}, gpu_id={gpu_id}")
    accuracy, error = loc.test_localization(db_path, test_q_path, k, gpu_enabled, gpu_id, loc_type, **kwargs)
    print('accuracy:')
    print(json.dumps(accuracy, indent=4))
    print('error:')
    print(json.dumps(error, indent=4))
