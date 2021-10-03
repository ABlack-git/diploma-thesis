import yaml
import collections
import torch
import torch.nn as nn
from abc import ABC

import im2gps.core.nn.layers as im2gps_layers


class Im2GPSNetwork(nn.Module):
    def __init__(self, transformations_list: list, name, network_config, d2w=None, kde=None, transform_only=True):
        super().__init__()
        self.name = name
        self.network_cofig = network_config
        self.transformations = None if transformations_list is None else nn.Sequential(*transformations_list)
        self.transform_only = transform_only
        self.d2w: im2gps_layers.Descriptors2Weights = d2w
        self.kde: im2gps_layers.KDE = kde

    def forward(self, query=None, neighbours=None, n_coords=None, in_descriptors=None):
        if self.transform_only:
            assert in_descriptors is not None, "Descriptors should be provided"
            return self.transformations(in_descriptors)
        else:
            assert query is not None, "Query descriptor should be provided"
            assert neighbours is not None, "Neighbours descriptors should be provided"
            assert n_coords is not None, "Neighbour coordinates should be provided"

            q = self.transformations(query)
            descriptors = self.transformations(neighbours)
            weights = self.d2w(q, descriptors)
            out = self.kde(weights, n_coords)
            return out

    def __str__(self):
        return yaml.dump(self.network_cofig, default_flow_style=False)


class Im2GPSNetworkInitializer:
    def __init__(self, network_config: dict):
        self._layer_builder = LayerBuilder()
        self.net_conf = network_config

    def _init_transform_layers(self):
        arch = self.net_conf['network']['transformation_layers']
        t_layers = []
        for layer_conf in arch:
            layer = self._layer_builder.build(layer_conf)
            t_layers.append(layer)
        return t_layers

    def _init_custom_layers(self):
        arch = self.net_conf['network']['custom_layers']
        custom_layers = {}
        for layer_conf in arch:
            layer = self._layer_builder.build(layer_conf)
            if layer.__class__.__name__ == 'KDE':
                custom_layers['kde'] = layer
            elif layer.__class__.__name__ == 'Descriptors2Weights':
                custom_layers['d2w'] = layer
            else:
                raise ValueError(f"Unknown layer: {layer.__class__.__name__}")
        return custom_layers

    def init_network(self):
        t_layers = self._init_transform_layers()
        c_layers = self._init_custom_layers()
        name = self.net_conf['network']['name']
        if 'transform_only' in self.net_conf['network']:
            transform_only = self.net_conf['network']['transform_only']
        else:
            transform_only = False
        net = Im2GPSNetwork(t_layers, name, self.net_conf, **c_layers, transform_only=transform_only)

        if 'restore_from' in self.net_conf['network'] and self.net_conf['network']['restore_from'] is not None:
            net.load_state_dict(torch.load(self.net_conf['network']['restore_from']))

        return net


class ModuleBuilder(ABC):
    def __init__(self, *namespaces):
        self._namespaces = collections.ChainMap(*namespaces)

    def _get_instance(self, name, *args, **kwargs):
        try:
            return self._namespaces[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e


class LayerBuilder(ModuleBuilder):
    def __init__(self, *namespaces):
        if len(namespaces) == 0:
            super().__init__(nn.__dict__, im2gps_layers.__dict__)
        else:
            super().__init__(*namespaces)

    def build(self, config):
        name, kwargs = list(config.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        return self._get_instance(name, *args, **kwargs)


class OptimizerBuilder(ModuleBuilder):
    def __init__(self, *namespaces):
        if len(namespaces) == 0:
            super().__init__(torch.optim.__dict__, torch.optim.lr_scheduler.__dict__)
        else:
            super().__init__(*namespaces)

    def build_optimizer(self, config, parameters):
        name, kwargs = list(config.items())[0]
        if kwargs is None:
            kwargs = {}

        return self._get_instance(name, parameters, **kwargs)

    def build_scheduler(self, config, optimizer):
        name, kwargs = list(config.items())[0]
        if kwargs is None:
            kwargs = {}
        return self._get_instance(name, optimizer, **kwargs)


def save_model(net, path):
    torch.save(net.state_dict(), path)
