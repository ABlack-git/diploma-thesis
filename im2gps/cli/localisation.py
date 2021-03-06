import click
import json
import im2gps.utils as utils
import logging
import im2gps.services.localisation as loc
from im2gps.conf.config import ConfigRepo, Config
from im2gps.lib.index import IndexType
from im2gps.lib.localisation import LocalisationType

log = logging.getLogger(__name__)


@click.group()
def localisation():
    pass


@localisation.command()
@click.option("--db-path", '-d', default=None, type=str, help="Path to database. If this option is not provided, "
                                                              "application will try to load this from config")
@click.option("--test-path", '-t', default=None, type=str, help="Path to test set. If this option is not provided, "
                                                                "application will try to load this from config")
@click.option("-k", default=1, help="Number of nearest neighbours")
@click.option("--gpu-enabled", '-g', is_flag=True, default=False,
              help="With this flag gpu will be used to find nearest neighbours")
@click.option("--gpu-id", default=-1, help="ID of GPU to use. If this option is not provided, "
                                           "application will try to load this from config")
@click.option("--loc-type", '-l',
              type=click.Choice([LocalisationType.NN.type_str, LocalisationType.KDE.type_str,
                                 LocalisationType.AVG.type_str]),
              help="Chose which localisation type to use")
@click.option("--sigma", type=float, default=None,
              help="Parameter required for KDE. This corresponds to standard deviation of distribution used in KDE")
@click.option("-m", type=float, default=None, help="Parameter required for kde and avg localisation")
@click.option("--avg-type", type=click.Choice(['weighted', 'regular']), default=None,
              help="Chose which average type to use in case of avg localisation")
@click.option("--index-type", '-i', type=click.Choice([IndexType.L2_INDEX.type_str, IndexType.COSINE_INDEX.type_str]),
              default=IndexType.L2_INDEX.type_str,
              help="Chose which index type to use. This will affect how distance between two descriptors is measured. "
                   f"Default value is {IndexType.L2_INDEX.type_str}")
def test_localization(db_path, test_path, k, gpu_enabled, gpu_id, loc_type, sigma, m, avg_type, index_type):
    cfg: Config = ConfigRepo().get(Config.__name__)
    if db_path is None:
        db_path = cfg.data.datasets.train
    if test_path is None:
        test_path = cfg.data.datasets.test_queries
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
             f"test_q_path={test_path}, k={k}, gpu_enabled={gpu_enabled}, gpu_id={gpu_id}")

    if index_type == IndexType.L2_INDEX.type_str:
        index_type = IndexType.L2_INDEX
    elif index_type == IndexType.COSINE_INDEX.type_str:
        index_type = IndexType.COSINE_INDEX
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    accuracy, error, _ = loc.test_localization(db_path, test_path, k, gpu_enabled, gpu_id, loc_type, index_type,
                                               **kwargs)
    print('accuracy:')
    print(json.dumps(accuracy, indent=4))
    print('error:')
    print(json.dumps(error, indent=4))


@localisation.command()
@click.option("--output-path", "-o", default=None, type=str)
@click.option("--start-from", '-s', default=0, type=int)
@click.option("--save-every", '-i', default=25000, type=int)
def photo_densities(output_path, start_from, save_every):
    utils.create_output_folders(output_path, with_filename=True)
    loc.get_image_density_at_query_loc(output_path, start_from, save_every)
