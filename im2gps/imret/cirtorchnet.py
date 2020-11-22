import os
import logging
import numpy as np
import torch
import torch.nn as nn

from torch.utils.model_zoo import load_url
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network, ImageRetrievalNet
from cirtorch.datasets.datahelpers import imresize, default_loader
from cirtorch.utils.general import get_data_root
from PIL import Image

from typing import List

log = logging.getLogger(__name__)

PRETRAINED = {
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/'
                                   'rSfM120k-tl-resnet101-gem-w-a155e54.pth'}


def load_network() -> ImageRetrievalNet:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    state = load_url(PRETRAINED['rSfM120k-tl-resnet101-gem-w'], model_dir=os.path.join(get_data_root(), 'networks'))
    net = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'],
                        'whitening': state['meta'].get('whitening', False)})
    net.load_state_dict(state['state_dict'])
    net.eval()
    log.info(f"Loaded network: {net.meta_repr()}")
    return net


def _multi_scale_extract(net: ImageRetrievalNet, img: Image, scales: List[float]):
    v = torch.zeros(net.meta['outputdim'])
    for s in scales:
        if s == 1:
            img_t = img.clone()
        else:
            img_t = nn.functional.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(img_t).cpu().data.squeeze()
    v /= v.norm()
    return v


def get_descriptors(net: ImageRetrievalNet, img: Image):
    image_resol = 512
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])])
    imgr = transform(imresize(img, image_resol))
    return _multi_scale_extract(net, imgr.unsqueeze(0), [1, 1 / np.sqrt(2), np.sqrt(2)])


def main():
    # setting up the visible GPU
    net = load_network()
    # load example image
    img = default_loader("/home.zam/toliageo/flower.jpg")

    vec = get_descriptors(net,img)

    # multi-scale extraction, multiple input image resolutions, and aggregation of descriptors

    print(vec)


if __name__ == '__main__':
    main()
