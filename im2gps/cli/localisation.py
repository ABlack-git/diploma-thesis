import click
import json
import im2gps.utils as utils
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
    accuracy, error, _ = loc.test_localization(db_path, test_q_path, k, gpu_enabled, gpu_id, loc_type, **kwargs)
    print('accuracy:')
    print(json.dumps(accuracy, indent=4))
    print('error:')
    print(json.dumps(error, indent=4))


@localisation.command()
@click.option("--file-path", "-f", default=None, type=str, help='Path to file with queries')
@click.option("--dataset", "-d", type=click.Choice(['train', 'test', 'val']), default=None)
@click.option("--save", "-s", is_flag=True, default=False)
@click.option("--output-path", "-o", default=None, type=str)
def photo_densities(file_path, dataset, save, output_path):
    cfg: Config = ConfigRepo().get(Config.__name__)
    if file_path is None and dataset is not None:
        if dataset == 'train':
            file_path = cfg.data.datasets.train
        elif dataset == 'test':
            file_path = cfg.data.datasets.test_queries
        elif dataset == 'val':
            file_path = cfg.data.datasets.validation_queries
    else:
        raise ValueError("Either -f or -d option should be provided")
    densities = loc.get_image_density_at_query_loc(file_path)
    if save:
        if output_path is None:
            output_path = cfg.properties.output_dir + "/image-densities.json"
        utils.create_output_folders(output_path, with_filename=True)
        with open(output_path, 'w') as file_path:
            json.dump(densities, file_path)
    else:
        print(densities)
