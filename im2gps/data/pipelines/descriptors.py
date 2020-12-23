import logging
import os
import torch
import im2gps.lib.cirtorchnet as cirtorch

from im2gps.conf.config import ConfigRepo
from im2gps.conf.configschema import Config
from im2gps.data.data import get_img_id_from_path, DirectoryIterator
from im2gps.data.descriptors import DescriptorsTable, Descriptor
from im2gps.data.flickr_repo import FlickrPhoto
from im2gps.exceptions import FlickrPhotoNotFound

log = logging.getLogger(__name__)


def make_descriptors():
    cfg: Config = ConfigRepo().get(Config.__name__)
    device = torch.device(f'cuda:{cfg.properties.gpu_id}' if torch.cuda.is_available() else 'cpu')
    log.info(f"Loading network to device: {device.type}, selected gpu is {cfg.properties.gpu_id}")
    net = cirtorch.load_network(cfg.cirtorch.model_dir, device)
    with DescriptorsTable(cfg.data.datasets.train, net.meta['outputdim']) as table, \
            DirectoryIterator.load_or_create(cfg.data.data_directory,
                                             cfg.checkpoints.descriptor_checkpoint.checkpoint_path) as paths:

        for i, file_path in enumerate(paths):
            if not os.path.basename(file_path).lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                log.debug(f"Skipped {file_path}")
                continue
            img_id = get_img_id_from_path(file_path)
            photo: FlickrPhoto = FlickrPhoto.objects(photo_id=img_id).first()
            if photo is None:
                raise FlickrPhotoNotFound(f"Photo with id {img_id} not found")

            img = cirtorch.default_loader(file_path)
            descriptor = cirtorch.get_descriptor_from_image(net, img, cfg.cirtorch.img_resolution, device)

            desc = Descriptor(
                photo_id=photo.photo_id,
                lon=photo.geo.coords['coordinates'][0],
                lat=photo.geo.coords['coordinates'][1],
                descriptor=descriptor
            )
            table.add(desc)
            log.info(f"{i} Saved descriptors of {file_path}")

