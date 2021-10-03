import numpy as np
import shutil
import torch
import yaml
import os
import logging
import time
import copy
import typing as t
from itertools import product
from datetime import datetime
from dataclasses import dataclass, asdict
from mongoengine import connect
from omegaconf import OmegaConf

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from im2gps.core.nn.network import Im2GPSNetworkInitializer, Im2GPSNetwork, OptimizerBuilder
from im2gps.core.nn.datasets import DescriptorsDataset
from im2gps.core.nn.layers import HaversineLoss, KDELoss
from im2gps.utils import Stats
from im2gps.conf.net.configschema import TrainConfig, TrainProperties

log = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    epoch: int
    scheduler_state_dict: dict
    optimizer_state_dict: dict
    min_loss: float


class NetworkTrainService:
    def __init__(self, net, optimizer, criterion, train_loader, val_loader, train_properties: TrainProperties,
                 summary_writer: SummaryWriter, scheduler=None, min_loss=float('inf')):
        self.net: Im2GPSNetwork = net
        self.optimizer: Optimizer = optimizer
        self.criterion: HaversineLoss = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 0
        self.summary_writer = summary_writer
        self.properties = train_properties

        self.min_loss = min_loss

    @classmethod
    def init(cls, train_cfg: TrainConfig):
        log.info("Initializing network training")
        log.info("Reading network config")
        net_config = _read_network_config(train_cfg.net_config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(train_cfg.properties.gpu_id)

        log.info("Building network")
        net = Im2GPSNetworkInitializer(net_config).init_network()
        net.cuda()

        log.info("Adding loss function")
        criterion = KDELoss().cuda()

        log.info(f"Building optimizer: {train_cfg.optimizer_config}")
        opt_builder = OptimizerBuilder()
        optimizer: Optimizer = opt_builder.build_optimizer(train_cfg.optimizer_config, net.parameters())

        scheduler = None
        if train_cfg.scheduler_config is not None:
            log.info(f"Building scheduler: {train_cfg.scheduler_config}")
            scheduler = opt_builder.build_scheduler(train_cfg.scheduler_config, optimizer)

        log.info("Building data loader")
        train_dataset = DescriptorsDataset(train_cfg.data_config.train_ds)

        def connect_to_db(worker):
            connect(db=train_cfg.data_config.db_config.db,
                    host=train_cfg.data_config.db_config.host,
                    port=train_cfg.data_config.db_config.port)

        if train_cfg.data_config.num_workers == 0:
            connect_to_db(0)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=train_cfg.data_config.batch_size,
            num_workers=train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=connect_to_db
        )

        val_dataset = DescriptorsDataset(train_cfg.data_config.val_ds)

        val_loader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=train_cfg.data_config.batch_size,
            num_workers=train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=connect_to_db
        )

        summary_writer = None
        if train_cfg.properties.summary_writer:
            summary_log_dir = os.path.join(train_cfg.properties.base_dir, "runs")
            if not os.path.isdir(summary_log_dir):
                os.makedirs(summary_log_dir)
            summary_writer = SummaryWriter(log_dir=summary_log_dir)

        log.info("Finishing initialization")
        return cls(net, optimizer, criterion, train_loader, val_loader, train_cfg.properties,
                   summary_writer, scheduler=scheduler)

    def train(self):
        losses = []
        for epoch in range(self.start_epoch, self.properties.num_epochs):
            log.info(f"Starting epoch number {epoch}")
            loss = self._train_for_one_epoch(epoch)
            log.info(f"Average training loss: {loss:.3f}")

            # validate
            validation_loss = None
            if self.properties.validate:
                log.info("Running validation...")
                validation_loss = self.validate(epoch)
                log.info(f"Average validation loss: {validation_loss:.3f}")

            losses.append({"loss": loss, "validation_loss": validation_loss, "epoch": epoch})
            # test

            # add stats
            if self.properties.summary_writer:
                self.summary_writer.add_scalar("Loss/train", loss, epoch)
                if self.scheduler is not None:
                    self.summary_writer.add_scalar("Parameters/lr", self.scheduler.get_last_lr())
                if validation_loss is not None:
                    self.summary_writer.add_scalar("Loss/validation", validation_loss, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
                log.debug(f"Current learning rate: {self.scheduler.get_last_lr()}")

            # save checkpoint
            if self.properties.save_checkpoint:
                is_best = loss < self.min_loss
                self.min_loss = min(loss, self.min_loss)
                log.info("Saving checkpoint")
                checkpoint = Checkpoint(epoch, self.scheduler.state_dict(), self.optimizer.state_dict(),
                                        self.min_loss)
                self._save_train_checkpoint(checkpoint, is_best)

        return losses

    def _train_for_one_epoch(self, current_epoch):
        loss_stat = Stats(self.properties.sma_window)
        batch_stat = Stats(self.properties.sma_window)
        data_stat = Stats(self.properties.sma_window)

        start_time = time.time()
        for i, train_tuple in enumerate(self.train_loader):
            data_stat.current = time.time() - start_time

            self.optimizer.zero_grad()

            q, neighbours, q_coords, n_coords, q_ids, n_ids = train_tuple
            q = q.cuda()
            neighbours = neighbours.cuda()
            q_coords = q_coords.cuda()
            n_coords = n_coords.cuda()
            out = self.net(query=q, neighbours=neighbours, n_coords=n_coords)
            loss = self.criterion(out, n_coords, q_coords)
            loss_stat.current = loss.item()

            loss.backward()
            self.optimizer.step()

            batch_stat.current = time.time() - start_time

            if (i + 1) % self.properties.print_freq == 0 or i == 0 or (i + 1) == len(self.train_loader):
                log.info(f"Train: [epoch: {current_epoch + 1}/{self.properties.num_epochs}, "
                         f"batch: {i + 1}/{len(self.train_loader)}]")
                log.debug(f"Data time: [sma: {data_stat.sma:.3f}, avg: {data_stat.avg:.3f}, "
                          f"current: {data_stat.current:.3f}]")
                log.debug(f"Batch time: [sma: {batch_stat.sma:.3f}, avg: {batch_stat.avg:.3f}, "
                          f"current: {batch_stat.current:.3f}]")
                log.info(f"Loss: [sma: {loss_stat.sma:.3f}, avg: {loss_stat.avg:.3f}, "
                         f"current: {loss_stat.current:.3f}]")

            start_time = time.time()
        return loss_stat.avg

    def validate(self, current_epoch):
        loss_stat = Stats(self.properties.sma_window)
        data_stat = Stats(self.properties.sma_window)
        batch_stat = Stats(self.properties.sma_window)

        start_time = time.time()
        for i, val_tuple in enumerate(self.val_loader):
            data_stat.current = time.time() - start_time
            q, neighbours, q_coords, n_coords, _, _ = val_tuple
            q = q.cuda()
            neighbours = neighbours.cuda()
            q_coords = q_coords.cuda()
            n_coords = n_coords.cuda()
            with torch.no_grad():
                out = self.net(query=q, neighbours=neighbours, n_coords=n_coords)
                loss = self.criterion(out, n_coords, q_coords)

            loss_stat.current = loss.item()
            batch_stat.current = time.time() - start_time

            if (i + 1) % self.properties.print_freq == 0 or i == 0 or (i + 1) == len(self.val_loader):
                log.info(f"Validation: [epoch: {current_epoch + 1}/{self.properties.num_epochs}, "
                         f"batch: {i + 1}/{len(self.val_loader)}]")
                log.debug(f"Data time: [sma: {data_stat.sma:.3f}, avg: {data_stat.avg:.3f}, "
                          f"current: {data_stat.current:.3f}]")
                log.debug(f"Batch time: [sma: {batch_stat.sma:.3f}, avg: {batch_stat.avg:.3f}, "
                          f"current: {batch_stat.current:.3f}]")
                log.info(f"Loss: [sma: {loss_stat.sma:.3f}, avg: {loss_stat.avg:.3f}, "
                         f"current: {loss_stat.current:.3f}]")

            start_time = time.time()

        return loss_stat.avg

    def _save_train_checkpoint(self, checkpoint: Checkpoint, is_best):
        checkpoints_dir = os.path.join(self.properties.base_dir, 'checkpoints')
        log.info(f"Directory with checkpoints: {checkpoints_dir}")
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        current_checkpoint_dir = os.path.join(checkpoints_dir,
                                              f"checkpoint_epoch-{checkpoint.epoch}"
                                              f"_{datetime.now().strftime('%Y-%m-%d')}")
        os.makedirs(current_checkpoint_dir)
        log.info(f"Directory of current checkpoint: {current_checkpoint_dir}")

        checkpoint_path = os.path.join(current_checkpoint_dir, f"checkpoint.pth")
        log.info(f"Saving checkpoint to: {checkpoint_path}")
        torch.save(asdict(checkpoint), checkpoint_path)

        model_state_path = os.path.join(current_checkpoint_dir, "model_state.pth")
        log.info(f"Saving model state to: {model_state_path}")
        torch.save(self.net.state_dict(), model_state_path)

        if is_best:
            best_path = os.path.join(checkpoints_dir, 'best_model')
            log.info(f"Saving best model to: {best_path}")
            if not os.path.isdir(best_path):
                os.makedirs(best_path)
            shutil.copyfile(checkpoint_path, os.path.join(best_path, "checkpoint.pth"))
            shutil.copyfile(model_state_path, os.path.join(best_path, "model_state.pth"))

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
        checkpoint = Checkpoint(**checkpoint_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        self.start_epoch = checkpoint.epoch + 1
        self.min_loss = checkpoint.min_loss

        net_state_dict = torch.load(os.path.join(checkpoint_dir, "model_state.pth"))
        self.net.load_state_dict(net_state_dict)


def _read_network_config(config_path):
    with open(config_path, 'r') as f:
        net_conf = yaml.load(f, yaml.FullLoader)
    return net_conf


class ParameterSelectionService:
    def __init__(self, config: dict, train_config: TrainConfig):
        self.config = config
        self.train_config = train_config

        self.__train_service: t.Union[NetworkTrainService, None] = None

    def _get_run_configs(self):
        configs = []
        for optimizer in self.config['optimizers']:
            name, params = tuple(optimizer.items())[0]
            params_product = product(*params.values())
            for params_values, batch_size in product(params_product, self.config['batch_size']):
                params_as_dict = dict(zip(params.keys(), params_values))
                config = ParametersRunConfig(
                    optimizer_config=OmegaConf.create({name: params_as_dict}),
                    batch_size=batch_size,
                    optimizer_name=name,
                    optimizer_params=params_as_dict)
                configs.append(config)
        return configs

    def _init_train_service(self, train_config):
        if self.__train_service is not None:
            del self.__train_service
        self.__train_service = NetworkTrainService.init(train_config)

    def grid_search(self):
        run_configs = self._get_run_configs()
        for i, run_config in enumerate(run_configs):  # type: int, ParametersRunConfig
            summary_path = os.path.join(self.config['base_dir'], f'runs', str(run_config))
            if not os.path.isdir(summary_path):
                os.makedirs(summary_path)
            sw = SummaryWriter(log_dir=summary_path)

            log.info(f"Running parameter selection task: {i + 1}/{len(run_configs)}")
            log.info(f"Current parameters: {run_config.get_params_as_dict()}")
            train_config = copy.deepcopy(self.train_config)
            train_config.optimizer_config = run_config.optimizer_config
            train_config.data_config.batch_size = run_config.batch_size

            self._init_train_service(train_config)

            losses = self.__train_service.train()

            for loss_stat in losses:
                sw.add_scalar("Loss/train", loss_stat['loss'], loss_stat['epoch'])
                sw.add_scalar("Loss/validation", loss_stat['validation_loss'], loss_stat['epoch'])

                sw.add_hparams(hparam_dict=run_config.get_params_as_dict(),
                               metric_dict={"Hparam/train_loss": loss_stat['loss'],
                                            "Hparam/validation_loss": loss_stat['validation_loss']},
                               run_name=str(run_config))

            sw.close()


@dataclass
class ParametersRunConfig:
    optimizer_config: dict
    optimizer_name: str
    optimizer_params: dict
    batch_size: int

    def __str__(self):
        params = '-'.join([f"{k}-{v}" for k, v in self.optimizer_params.items()])
        return f"{self.optimizer_name}-{params}-batch-size-{self.batch_size}"

    def get_params_as_dict(self):
        return {
            **{'optimizer': self.optimizer_name,
               'batch_size': self.batch_size},
            **self.optimizer_params
        }


#
# def test():
#     pass


rc = ParametersRunConfig({}, 'SGD', {"lr": 0.001, "momentum": 0.9}, 64)
print(str(rc))
print(rc.get_params_as_dict())
