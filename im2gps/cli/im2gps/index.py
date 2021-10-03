import click
import os
from click_option_group import OptionGroup

from im2gps.core.index import IndexType, IndexConfig
from im2gps.services.index import create_and_save_index

index_config = OptionGroup("Index parameters", help="Parameters used for building faiss index")


@click.group()
def index():
    pass


@index.command()
@index_config.option("--index-type", '-i', type=click.Choice([enum.value for enum in IndexType]),
                     default=IndexType.L2_INDEX.value,
                     help="Chose which index type to use. This will affect how distance between two descriptors i"
                          f"s measured. Default value is {IndexType.L2_INDEX.value}")
@index_config.option("--gpu-id", default=-1, help="ID of GPU to use. Default value is -1")
@index_config.option("--gpu-enabled", '-g', is_flag=True, default=False,
                     help="If this flag is set index will be built on GPU. Default value is False")
@click.option("--path", "-p", type=str, help="Path to where to store index")
def save_index(**params):
    if not os.path.isdir(params['path']):
        raise ValueError(f"path {params['path']} should be a directory")

    index_params = IndexConfig()
    index_params.index_dir = params['path']
    index_params.index_type = IndexType(params['index_type'])
    index_params.gpu_enabled = params['gpu_enabled']
    index_params.gpu_id = params['gpu_id']

    create_and_save_index(index_params)
