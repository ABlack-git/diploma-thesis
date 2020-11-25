import os
import logging
import numpy as np
import torch
import torch.nn as nn

from torch.utils.model_zoo import load_url
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network, ImageRetrievalNet
from cirtorch.datasets.datahelpers import imresize, default_loader
from PIL import Image
from typing import List

from im2gps.conf.config import load_config, Config
from im2gps.data.data import get_img_id_from_path, DirectoryIterator
from im2gps.data.descriptors import DescriptorsTable, Descriptor
from im2gps.data.flickr_repo import FlickrPhoto
from im2gps.exceptions import FlickrPhotoNotFound

from mongoengine import connect, disconnect

log = logging.getLogger(__name__)

PRETRAINED = {
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/'
                                   'rSfM120k-tl-resnet101-gem-w-a155e54.pth'}


def load_network(cfg: Config) -> ImageRetrievalNet:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.imret.gpu)
    state = load_url(PRETRAINED['rSfM120k-tl-resnet101-gem-w'], model_dir=os.path.join(cfg.imret.model_dir))
    net = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'],
                        'whitening': state['meta'].get('whitening', False)})
    net.load_state_dict(state['state_dict'])
    net.eval()
    log.info(f"Loaded network: {net.meta_repr()}")
    return net


def _multi_scale_extract(net: ImageRetrievalNet, img: Image, scales: List[float]) -> np.array:
    v = torch.zeros(net.meta['outputdim'])
    for s in scales:
        if s == 1:
            img_t = img.clone()
        else:
            img_t = nn.functional.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(img_t).cpu().data.squeeze()
    v /= v.norm()
    return v.numpy()


def get_descriptor_from_image(net: ImageRetrievalNet, img: Image, cfg: Config) -> np.array:
    image_resol = cfg.imret.img_resolution
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])])
    imgr = transform(imresize(img, image_resol))
    return _multi_scale_extract(net, imgr.unsqueeze(0), [1, 1 / np.sqrt(2), np.sqrt(2)])


def make_descriptors():
    conf: Config = load_config(Config.__name__)
    connect(db=conf.data.db.database, host=conf.data.db.host, port=conf.data.db.port)
    log.info(f"Loading network to gpu: {conf.imret.gpu}")
    net = load_network(conf)
    with DescriptorsTable(conf.imret.descriptor_file, net.meta['outputdim']) as table, \
            DirectoryIterator.load_or_create(conf.imret.data_dir, conf.imret.checkpoint_path) as paths:

        for file_path in paths:
            if os.path.basename(file_path).lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                continue
            log.info(f"Getting descriptors for {file_path}")
            img_id = get_img_id_from_path(file_path)
            photo: FlickrPhoto = FlickrPhoto.objects(photo_id=img_id).first()
            if photo is None:
                raise FlickrPhotoNotFound(f"Photo with id {img_id} not found")

            img = default_loader(file_path)
            descriptor = get_descriptor_from_image(net, img, conf)

            desc = Descriptor(
                photo_id=photo.photo_id,
                lon=photo.geo.coords['coordinates'][0],
                lat=photo.geo.coords['coordinates'][1],
                descriptor=descriptor
            )
            table.add(desc)
            log.info(f"Saved descriptors of {file_path}")

    disconnect()
