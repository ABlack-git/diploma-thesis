import shutil
import torch
import yaml
import os
import logging
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from mongoengine import connect

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from im2gps.core.nn.network import Im2GPSNetworkInitializer, Im2GPSNetwork, OptimizerBuilder
from im2gps.core.nn.datasets import DescriptorsDataset
from im2gps.core.nn.layers import HaversineLoss
from im2gps.utils import Stats
from im2gps.conf.net.configschema import TrainConfig, TrainProperties

log = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    epoch: int
    optimizer_state_dict: dict
    min_loss: float


class NetworkTrainService:
    def __init__(self, net, optimizer, criterion, train_loader, val_loader, train_properties: TrainProperties,
                 scheduler=None, min_loss=float('inf')):
        self.net: Im2GPSNetwork = net
        self.optimizer: Optimizer = optimizer
        self.criterion: HaversineLoss = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 0

        self.properties = train_properties

        self.min_loss = min_loss

    @staticmethod
    def init(train_cfg: TrainConfig) -> 'NetworkTrainService':
        log.info("Initializing network training")
        log.info("Reading network config")
        net_config = _read_network_config(train_cfg.net_config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(train_cfg.properties.gpu_id)

        log.info("Building network")
        net = Im2GPSNetworkInitializer(net_config).init_network()
        net.cuda()

        log.info("Adding loss function")
        criterion = HaversineLoss().cuda()

        log.info(f"Building optimizer: {train_cfg.optimizer_config}")
        opt_builder = OptimizerBuilder()
        optimizer: Optimizer = opt_builder.build_optimizer(train_cfg.optimizer_config, net.parameters())

        scheduler = None
        if train_cfg.scheduler_config is not None:
            log.info(f"Building scheduler: {train_cfg.scheduler_config}")
            scheduler = opt_builder.build_scheduler(train_cfg.scheduler_config, optimizer)

        log.info("Building data loader")
        train_dataset = DescriptorsDataset(train_cfg.data_config.train_ds)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=train_cfg.data_config.batch_size,
            num_workers=train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda worker: connect(db=train_cfg.data_config.db_config.db,
                                                  host=train_cfg.data_config.db_config.host,
                                                  port=train_cfg.data_config.db_config.port)
        )

        val_dataset = DescriptorsDataset(train_cfg.data_config.val_ds)

        val_loader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=train_cfg.data_config.batch_size,
            num_workers=train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda worker: connect(db=train_cfg.data_config.db_config.db,
                                                  host=train_cfg.data_config.db_config.host,
                                                  port=train_cfg.data_config.db_config.port)
        )

        log.info("Finishing initialization")
        return NetworkTrainService(net, optimizer, criterion, train_loader, val_loader, train_cfg.properties,
                                   scheduler=scheduler)

    def train(self):
        for epoch in range(self.start_epoch, self.properties.num_epochs):
            log.info(f"Starting epoch number {epoch}")

            if self.scheduler is not None:
                self.scheduler.step()

            loss = self._train_for_one_epoch(epoch)

            # validate
            if self.properties.validate:
                validation_loss = self.validate(epoch)

            # test

            # save checkpoint
            is_best = loss < self.min_loss
            self.min_loss = min(loss, self.min_loss)

            log.info("Saving checkpoint")
            checkpoint = Checkpoint(epoch, self.optimizer.state_dict(), self.min_loss)
            self._save_train_checkpoint(checkpoint, is_best)

    def validate(self, current_epoch):
        loss_stat = Stats(self.properties.sma_window)
        data_stat = Stats(self.properties.sma_window)
        batch_stat = Stats(self.properties.sma_window)

        start_time = time.time()
        for i, val_tuple in enumerate(self.val_loader):
            data_stat.current = time.time() - start_time
            q, neighbours, q_coords, n_coords, _, _ = val_tuple

            with torch.no_grad:
                out = self.net(query=q, neighbours=neighbours, n_coords=n_coords)
                loss = self.criterion(out, q_coords)

            loss_stat.current = loss.item()
            batch_stat.current = time.time() - start_time

            if (i + 1) % self.properties.print_freq == 0 or i == 0 or (i + 1) == len(self.val_loader):
                log.info(f"Validation: [epoch: {current_epoch + 1}/{self.properties.num_epochs}, "
                         f"batch: {i + 1}/{len(self.val_loader)}]")
                log.info(f"Data time: [sma: {data_stat.sma}, avg: {data_stat.avg}, current: {data_stat.current}]")
                log.info(f"Batch time: [sma: {batch_stat.sma}, avg: {batch_stat.avg}, current: {batch_stat.current}]")
                log.info(f"Loss: [sma: {loss_stat.sma}, avg: {loss_stat.avg}, current: {loss_stat.current}]")

            start_time = time.time()

        return loss_stat.avg

    def _train_for_one_epoch(self, current_epoch):
        loss_stat = Stats(self.properties.sma_window)
        batch_stat = Stats(self.properties.sma_window)
        data_stat = Stats(self.properties.sma_window)

        start_time = time.time()
        for i, train_tuple in enumerate(self.train_loader):
            data_stat.current = time.time() - start_time

            self.optimizer.zero_grad()

            q, neighbours, q_coords, n_coords, _, _ = train_tuple

            out = self.net(query=q, neighbours=neighbours, n_coords=n_coords)
            loss = self.criterion(out, q_coords)
            loss_stat.current = loss.item()

            loss.backward()
            self.optimizer.step()

            batch_stat.current = time.time() - start_time

            if (i + 1) % self.properties.print_freq == 0 or i == 0 or (i + 1) == len(self.train_loader):
                log.info(f"Train: [epoch: {current_epoch + 1}/{self.properties.num_epochs}, "
                         f"batch: {i + 1}/{len(self.train_loader)}]")
                log.info(f"Data time: [sma: {data_stat.sma}, avg: {data_stat.avg}, current: {data_stat.current}]")
                log.info(f"Batch time: [sma: {batch_stat.sma}, avg: {batch_stat.avg}, current: {batch_stat.current}]")
                log.info(f"Loss: [sma: {loss_stat.sma}, avg: {loss_stat.avg}, current: {loss_stat.current}]")

            start_time = time.time()
        return loss_stat.avg

    def _save_train_checkpoint(self, checkpoint: Checkpoint, is_best):
        checkpoints_dir = os.path.join(self.properties.base_dir, 'checkpoints')
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        checkpoint_dir = os.path.join(checkpoints_dir,
                                      f"checkpoint_epoch={checkpoint.epoch}"
                                      f"-{datetime.now().strftime('%Y-%m-%d')}")
        os.makedirs(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
        torch.save(asdict(checkpoint), checkpoint_path)
        model_state_path = os.path.join(checkpoint_path, "model_state.pth")
        torch.save(self.net.state_dict(), model_state_path)

        if is_best:
            best_path = os.path.join(checkpoints_dir, 'best_model')
            shutil.copyfile(checkpoint_path, os.path.join(best_path, "checkpoint.pth"))
            shutil.copyfile(model_state_path, os.path.join(best_path, "model_state.pth"))

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
        checkpoint = Checkpoint(**checkpoint_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.start_epoch = checkpoint.epoch
        self.min_loss = checkpoint.min_loss

        net_state_dict = torch.load(os.path.join(checkpoint_dir, "model_state.pth"))
        self.net.load_state_dict(net_state_dict)


def _read_network_config(config_path):
    with open(config_path, 'r') as f:
        net_conf = yaml.load(f, yaml.FullLoader)
    return net_conf


def test():
    pass
