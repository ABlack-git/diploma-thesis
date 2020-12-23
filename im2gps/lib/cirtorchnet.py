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

log = logging.getLogger(__name__)

PRETRAINED = {
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/'
                                   'rSfM120k-tl-resnet101-gem-w-a155e54.pth'}


def load_network(model_dir: str, device: torch.device) -> ImageRetrievalNet:
    state = load_url(PRETRAINED['rSfM120k-tl-resnet101-gem-w'], model_dir=model_dir)
    net = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'],
                        'whitening': state['meta'].get('whitening', False)})
    net.load_state_dict(state['state_dict'])
    net.to(device)
    net.eval()
    log.info(f"Loaded network: {net.meta_repr()}")
    return net


def _multi_scale_extract(net: ImageRetrievalNet, img: torch.Tensor, scales: List[float], device) -> np.array:
    v = torch.zeros(net.meta['outputdim'])
    img = img.to(device)
    for s in scales:
        if s == 1:
            img_t = img.clone()
        else:
            img_t = nn.functional.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(img_t).cpu().data.squeeze()
    v /= v.norm()
    return v.numpy()


def get_descriptor_from_image(net: ImageRetrievalNet, img: Image, img_resolution: int, device) -> np.array:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])])
    imgr: torch.Tensor = transform(imresize(img, img_resolution))
    return _multi_scale_extract(net, imgr.unsqueeze(0), [1, 1 / np.sqrt(2), np.sqrt(2)], device)


def load_image(img_path):
    return default_loader(img_path)
