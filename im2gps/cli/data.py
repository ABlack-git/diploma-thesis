import click
from im2gps.data.pipelines import flickr, descriptors, splitdata


@click.group()
def data():
    pass


@data.command()
@click.option("--checkpoint-type", type=click.Choice(['from-db', 'from-config'], case_sensitive=True), default=None,
              help="Choose where to load checkpoint from")
def download_photos_meta(checkpoint_type):
    """Download metadata of flicker photos"""
    flickr.collect_photos_metadata(checkpoint_type)


@data.command()
@click.option("--checkpoint-type", type=click.Choice(['from-db', 'from-cli'], case_sensitive=True), default=None)
@click.option("--to-skip", default=0, help="Provide a number where to start download from, only works with "
                                           "--checkpoint-type=from-cli")
def download_photos(checkpoint_type, to_skip):
    """
    Download photos from flickr
    """
    flickr.download_photos(checkpoint_type, to_skip)


@data.command()
def get_descriptors():
    """Make descriptors from photos"""
    descriptors.make_descriptors()


@data.command()
def split_datasets():
    """Split photos into datasets"""
    splitdata.split_data()
