import click

from im2gps.services.stats import density_near_query
from im2gps.data.descriptors import DatasetEnum


@click.group()
def stats():
    pass


@stats.command()
@click.option("-o", "--output-path", type=str)
@click.option("-d", "--dataset", type=click.Choice([enum.value for enum in DatasetEnum]))
@click.option("-i", "--index-file", type=str)
def get_density(output_path, dataset, index_file):
    density_near_query(output_path, DatasetEnum(dataset), index_file)
