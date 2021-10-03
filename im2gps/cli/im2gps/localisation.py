import click
import json
import logging
from click_option_group import OptionGroup

from im2gps.services.localisation import ModelParameters, BenchmarkParameters, perform_localisation_benchmark
from im2gps.conf.config import ConfigRepo, Config
from im2gps.core.index import IndexType, IndexConfig
from im2gps.core.localisation import LocalisationType
from im2gps.data.descriptors import DatasetEnum

log = logging.getLogger(__name__)

model_config = OptionGroup("Model parameters", help="Parameters for localisation model")
index_config = OptionGroup("Index parameters", help="Parameters used for building faiss index")
test_config = OptionGroup("Test options", help="Parameters for running test")


@click.group()
def localisation():
    pass


@localisation.command()
@model_config.option("--loc-type", '-l',
                     type=click.Choice([enum.value for enum in LocalisationType]),
                     default=LocalisationType.NN.value,
                     help=f"Chose which localisation type to use. Default value is {LocalisationType.NN}")
@model_config.option("-k", default=1, help="Number of nearest neighbours. Default value is 1")
@model_config.option("--sigma", type=float, default=None,
                     help="Parameter required for KDE. This corresponds to standard deviation of distribution used "
                          "in KDE. Default value is None")
@model_config.option("-m", type=float, default=None, help="Parameter required for kde and avg localisation")
@index_config.option("--index-type", '-i', type=click.Choice([enum.value for enum in IndexType]),
                     default=IndexType.L2_INDEX.value,
                     help="Chose which index type to use. This will affect how distance between two descriptors i"
                          f"s measured. Default value is {IndexType.L2_INDEX.value}")
@index_config.option("--index-dir", type=str, help="Provide path to the directory where index.if file is stored",
                     default=None)
@index_config.option("--gpu-id", default=-1, help="ID of GPU to use. Default value is -1")
@index_config.option("--gpu-enabled", '-g', is_flag=True, default=False,
                     help="If this flag is set index will be built on GPU. Default value is False")
@test_config.option("--queries", "-q", type=click.Choice([enum.value for enum in DatasetEnum]),
                    default=DatasetEnum.TEST_QUERY.value,
                    help=f"Parameter to specify which dataset to use as queries. "
                         f"Default value is {DatasetEnum.TEST_QUERY.value}")
@test_config.option("--extended-results", "-e", is_flag=True, default=False,
                    help="When this flag is set results will contain per image information, however additional "
                         "information will be only saved to filed and will not be displayed")
@test_config.option("--save", "-s", is_flag=True, default=False, help="If this flag is set, results will be saved on "
                                                                      "disk")
@test_config.option("--path", "-p", type=str, help="Path to where save results")
@click.option("--model-config", is_flag=True, default=False,
              help="If this flag is set will load model configuration from config file, ignoring provided parameters.")
@click.option("--index-config", is_flag=True, default=False,
              help="If this flag is set will load index configuration from config file, ignoring provided parameters.")
@click.option("--test-config", is_flag=True, default=False,
              help="If this flag is set will load test configuration from config file, ignoring provided parameters.")
def test_localisation(**params):
    cfg: Config = ConfigRepo().get(Config.__name__)
    log.debug(f"Running test-localisation, input parameters: {params}")

    model_params = ModelParameters()
    if params['model_config']:
        model_params.localisation_type = LocalisationType(cfg.localisation_model.localisation_type)
        model_params.k = cfg.localisation_model.k
        model_params.m = cfg.localisation_model.m
        model_params.sigma = cfg.localisation_model.sigma
    else:
        model_params.localisation_type = LocalisationType(params['loc_type'])
        model_params.k = params['k']
        model_params.sigma = params['sigma']
        model_params.m = params['m']

    index_params = IndexConfig()
    if params['index_config']:
        index_params.index_dir = cfg.index_config.index_dir
        index_params.index_type = IndexType(cfg.index_config.index_type)
        index_params.gpu_enabled = cfg.index_config.gpu_enabled
        index_params.gpu_id = cfg.index_config.gpu_id
    else:
        index_params.index_dir = params['index_dir']
        index_params.index_type = IndexType(params['index_type'])
        index_params.gpu_enabled = params['gpu_enabled']
        index_params.gpu_id = params['gpu_id']

    test_params = BenchmarkParameters()

    if params['test_config']:
        test_params.query_dataset = DatasetEnum(cfg.test_config.query_dataset)
        test_params.save_result = cfg.test_config.save_results
        test_params.save_path = cfg.test_config.save_path
        test_params.extended_results = cfg.test_config.extended_results
    else:
        test_params.query_dataset = DatasetEnum(params['queries'])
        test_params.save_result = params['save']
        test_params.save_path = params['path']
        test_params.extended_results = params['extended_results']

    result = perform_localisation_benchmark(model_params, index_params, test_params)

    print(json.dumps({"accuracy": result.accuracy, "errors": result.errors,
                      "predictions_by_dist": result.predictions_by_dist}, indent=4))
