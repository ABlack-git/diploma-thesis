import yaml
import collections
import torch
import torch.nn as nn
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

    def forward(self, *inputs, **kwargs):
        descriptors = self.transformations(inputs[0])

        if self.transform_only:
            return descriptors
        else:
            q = self.transformations(inputs[1])
            weights = self.d2w(q, descriptors)
            out = self.kde(weights, descriptors, inputs[2])
            return out

    def __str__(self):
        return yaml.dump(self.network_cofig, default_flow_style=False)


class Im2GPSNetworkInitializer:
    def __init__(self, network_config_path):
        self.network_config_path = network_config_path
        self._layer_builder = LayerBuilder(nn.__dict__, im2gps_layers.__dict__)
        with open(network_config_path, 'r') as f:
            self.nc = yaml.load(f, Loader=yaml.FullLoader)

    def _init_transform_layers(self):
        arch = self.nc['network']['transformation_layers']
        t_layers = []
        for layer_conf in arch:
            layer = self._layer_builder.build_layer(layer_conf)
            t_layers.append(layer)
        return t_layers

    def _init_custom_layers(self):
        arch = self.nc['network']['custom_layers']
        custom_layers = {}
        for layer_conf in arch:
            layer = self._layer_builder.build_layer(layer_conf)
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
        name = self.nc['network']['name']
        if 'transform_only' in self.nc['network']:
            transform_only = self.nc['network']['transform_only']
        else:
            transform_only = False
        net = Im2GPSNetwork(t_layers, name, self.nc, **c_layers, transform_only=transform_only)

        if 'restore_from' in self.nc['network'] and self.nc['network']['restore_from'] is not None:
            net.load_state_dict(torch.load(self.nc['network']['restore_from']))

        return net


class LayerBuilder:
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def _get_layer_instance(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def build_layer(self, layer_config):
        name, kwargs = list(layer_config.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        return self._get_layer_instance(name, *args, **kwargs)


def save_model(net, path):
    torch.save(net.state_dict(), path)
