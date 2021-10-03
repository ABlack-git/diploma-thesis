import click
import os
import yaml
from omegaconf import OmegaConf
from click_option_group import OptionGroup

from im2gps.conf.config import load_config, configure_logging
from im2gps.conf.net.configschema import TrainConfig
from im2gps.services.network import NetworkTrainService, ParameterSelectionService

train_properties = OptionGroup("Train properties", help="Providing parameters from this group will override default "
                                                        "parameters from config.")


@click.command()
@click.option("-v", "--verbosity",
              type=click.Choice(['disable', 'debug', 'info', 'warn', 'error', 'critical'], case_sensitive=False),
              default='info', help="Provide verbosity level")
@click.option("-p", "--config-path", type=str, required=True, help="Provide path to train config file")
@click.option("-c", "--checkpoint-path", type=str, default=None, help="Provide path to checkpoint directory")
@train_properties.option("-e", "--num-epochs", type=int, help="Specify number of ")
@train_properties.option("--gpu-id", type=int, help="Specify gpu on which to run training")
@train_properties.option("--print-freq", type=int, help="Specify how often to print training information, e.g. "
                                                        "if you provide value of 5 this will print information "
                                                        "every 5 batches")
@train_properties.option("--sma-window", type=int, help="Window width for computing simple moving average")
@train_properties.option("--num-workers", type=int, help="Num workers to load training data")
def train(verbosity, config_path, checkpoint_path, num_epochs, gpu_id, print_freq, sma_window, num_workers):
    cfg: TrainConfig = load_config([config_path], schema=TrainConfig, base_cfg_package="im2gps.conf.net",
                                   base_cfg="train-config.yaml")
    if num_epochs is not None:
        cfg.properties.num_epochs = num_epochs
    if gpu_id is not None:
        cfg.properties.gpu_id = gpu_id
    if print_freq is not None:
        cfg.properties.print_freq = print_freq
    if sma_window is not None:
        cfg.properties.sma_window = sma_window
    if num_workers is not None:
        cfg.data_config.num_workers = num_workers

    print(OmegaConf.to_yaml(cfg))

    filename = os.path.join(cfg.properties.base_dir, "logs", "train.log")
    configure_logging(verbosity, filename=filename, package="im2gps.conf.net", conf_file="net-logging.yaml")

    train_service = NetworkTrainService.init(cfg)

    if checkpoint_path is not None:
        train_service.load_checkpoint(checkpoint_path)

    train_service.train()


@click.command()
@click.option("-t", "--train-config", type=str, help="Path to training config")
@click.option("-p", "--tuning-config", type=str, help="path to tuning config")
@click.option("-v", "--verbosity",
              type=click.Choice(['disable', 'debug', 'info', 'warn', 'error', 'critical'], case_sensitive=False),
              default='info', help="Provide verbosity level")
def tune(train_config, tuning_config, verbosity):
    train_cfg: TrainConfig = load_config([train_config], schema=TrainConfig, base_cfg_package="im2gps.conf.net",
                                         base_cfg="train-config.yaml")
    with open(tuning_config, "r") as f:
        tuning_cfg = yaml.load(f, yaml.FullLoader)

    filename = os.path.join(tuning_cfg['base_dir'], "logs", "train.log")
    configure_logging(verbosity, filename=filename, package="im2gps.conf.net", conf_file="net-logging.yaml")

    tuning_service = ParameterSelectionService(tuning_cfg, train_cfg)

    tuning_service.grid_search()