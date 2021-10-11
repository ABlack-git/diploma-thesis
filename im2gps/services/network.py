import json
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
from im2gps.core.nn.enum import NNEnum
from im2gps.core.nn.datasets import TrainDataset, TestDataset
from im2gps.core.nn.layers import HaversineLoss, KDELoss
from im2gps.core.index import IndexBuilder, Index, IndexConfig, IndexType
from im2gps.utils import Stats
from im2gps.conf.net.configschema import TrainConfig, TrainProperties, TestProperties, ExtendedTestConfig
from im2gps.core.localisation import LocalisationModel, LocalisationType
from im2gps.services.localisation import _get_benchmark_results
from im2gps.data.descriptors import DatasetEnum
from im2gps.core.metric import dist_thresholds

log = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    epoch: int
    scheduler_state_dict: t.Optional[dict]
    optimizer_state_dict: dict
    min_loss: float


class NetworkTrainService:
    def __init__(self, net, optimizer, criterion, train_loader, train_properties: TrainProperties,
                 summary_writer: SummaryWriter = None, val_loader=None, scheduler=None,
                 test_service: 'NetworkTestService' = None):
        self.net: Im2GPSNetwork = net
        self.optimizer: Optimizer = optimizer
        self.criterion: HaversineLoss = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 0
        self.summary_writer = summary_writer
        self.properties = train_properties
        self.test_service = test_service
        self.min_loss = float('inf')

    def train(self):
        losses = []
        for epoch in range(self.start_epoch, self.properties.num_epochs):
            log.info(f"Starting epoch number {epoch}")
            loss = self._train_for_one_epoch(epoch)
            log.info(f"Average training loss: {loss:.3f}")

            validation_loss = float("inf")
            if self.properties.validate:
                log.info("Running validation...")
                validation_loss = self.validate(epoch)
                log.info(f"Average validation loss: {validation_loss:.3f}")

            losses.append({"loss": loss, "validation_loss": validation_loss, "epoch": epoch})

            if self.test_service is not None and (epoch + 1) % self.properties.test_freq == 0:
                log.info("Running test...")
                results = self.test_service.test(test_run=epoch)
                log.info(f"Test accuracy: {results.accuracy}")
                if self.properties.summary_writer:
                    for threshold in dist_thresholds.keys():
                        self.summary_writer.add_scalar(f"Accuracy/test_{threshold}",
                                                       results.accuracy[threshold], epoch)

            if self.properties.summary_writer:
                self.summary_writer.add_scalar("Loss/train", loss, epoch)
                if self.scheduler is not None:
                    self.summary_writer.add_scalar("Parameters/lr", self.scheduler.get_last_lr())
                if validation_loss is not None:
                    self.summary_writer.add_scalar("Loss/validation", validation_loss, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
                log.debug(f"Current learning rate: {self.scheduler.get_last_lr()}")

            if self.properties.save_checkpoint:
                is_best = validation_loss < self.min_loss
                self.min_loss = min(loss, self.min_loss)
                log.info("Saving checkpoint")
                scheduler_state_dict = None
                if self.scheduler is not None:
                    scheduler_state_dict = self.scheduler.state_dict()
                checkpoint = Checkpoint(epoch, scheduler_state_dict, self.optimizer.state_dict(),
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
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        self.start_epoch = checkpoint.epoch + 1
        self.min_loss = checkpoint.min_loss

        net_state_dict = torch.load(os.path.join(checkpoint_dir, "model_state.pth"))
        self.net.load_state_dict(net_state_dict)


class TrainServiceBuilder:
    def __init__(self, train_config: TrainConfig):
        self.train_cfg = train_config

        self.__opt_builder = OptimizerBuilder()

    def __init_network(self):
        net_config = _read_network_config(self.train_cfg.net_config_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.train_cfg.properties.gpu_id)
        net = Im2GPSNetworkInitializer(net_config).init_network()
        net.cuda()
        return net

    def __init_loss(self):
        return KDELoss().cuda()

    def __init_optimizer(self, net):
        log.debug(f"Optimizer config: {self.train_cfg.optimizer_config}")
        return self.__opt_builder.build_optimizer(self.train_cfg.optimizer_config, net.parameters())

    def __init_scheduler(self, optimizer):
        log.debug(f"Scheduler config: {self.train_cfg.scheduler_config}")
        return self.__opt_builder.build_scheduler(self.train_cfg.scheduler_config, optimizer)

    def __init_train_loader(self, worker_init_fn):
        log.debug(f"Train dataset: {self.train_cfg.data_config.train_ds}")
        train_dataset = TrainDataset(self.train_cfg.data_config.train_ds)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.train_cfg.data_config.batch_size,
            num_workers=self.train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

        return train_loader

    def __init_val_loader(self, worker_init_fn):
        log.debug(f"Validation dataset: {self.train_cfg.data_config.val_ds}")
        val_dataset = TrainDataset(self.train_cfg.data_config.val_ds)

        val_loader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=self.train_cfg.data_config.batch_size,
            num_workers=self.train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

        return val_loader

    def __init_summary_writer(self):
        summary_log_dir = os.path.join(self.train_cfg.properties.base_dir, "runs")
        log.debug(f"Summary log dir: {summary_log_dir}")
        if not os.path.isdir(summary_log_dir):
            os.makedirs(summary_log_dir)
        return SummaryWriter(log_dir=summary_log_dir)

    def __init_test_loaders(self, worker_init_fn):
        database_ds = TestDataset(DatasetEnum.DATABASE, self.train_cfg.test_config.dataset_file)

        db_loader = DataLoader(
            database_ds,
            batch_size=self.train_cfg.test_config.batch_size,
            num_workers=self.train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )

        assert self.train_cfg.test_config.test_dataset in set(item.value for item in DatasetEnum), \
            f"Unknown dataset type {self.train_cfg.test_config.test_dataset}"
        test_ds_type = DatasetEnum(self.train_cfg.test_config.test_dataset)
        test_ds = TestDataset(test_ds_type, self.train_cfg.test_config.dataset_file)

        test_loader = DataLoader(
            test_ds,
            batch_size=self.train_cfg.test_config.batch_size,
            num_workers=self.train_cfg.data_config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )

        return db_loader, test_loader

    def __init_test_service(self, net, ds_loader, test_loader):
        return NetworkTestService(net, ds_loader, test_loader, self.train_cfg.test_config.properties)

    def init(self):
        log.info("Initializing train service...")
        log.debug("Building network")
        net = self.__init_network()
        log.debug("Building criterion")
        criterion = self.__init_loss()
        log.debug("Building optimizer")
        optimizer = self.__init_optimizer(net)

        scheduler = None
        if self.train_cfg.scheduler_config is not None:
            log.debug(f"Building scheduler")
            scheduler = self.__init_scheduler(optimizer)

        db = self.train_cfg.data_config.db_config.db
        host = self.train_cfg.data_config.db_config.host
        port = self.train_cfg.data_config.db_config.port

        def connect_to_db(worker):
            connect(db=db, host=host, port=port)

        if self.train_cfg.data_config.num_workers == 0:
            log.debug("Num workers is 0, connecting to db")
            connect_to_db(0)
        log.debug("Building train loaders")
        train_loader = self.__init_train_loader(worker_init_fn=connect_to_db)

        val_loader = None
        if self.train_cfg.properties.validate:
            log.debug("Building val loader")
            val_loader = self.__init_val_loader(worker_init_fn=connect_to_db)

        summary_writer = None
        if self.train_cfg.properties.summary_writer:
            log.debug("Building summary writer")
            summary_writer = self.__init_summary_writer()

        test_service = None
        if self.train_cfg.test_config is not None:
            log.debug("Building test loaders")
            ds_loader, test_loader = self.__init_test_loaders(connect_to_db)
            log.debug("Building test service")
            test_service = self.__init_test_service(net, ds_loader, test_loader)

        train_service = NetworkTrainService(net, optimizer, criterion, train_loader, self.train_cfg.properties,
                                            summary_writer, val_loader, scheduler, test_service)
        log.info("Finished building train service")
        return train_service


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
        self.__train_service = TrainServiceBuilder(train_config).init()

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


class NetworkTestService:
    def __init__(self, net: Im2GPSNetwork, database_set_loader: DataLoader, test_set_loader: DataLoader,
                 properties: TestProperties):
        self.net = net
        self.database_set_loader = database_set_loader
        self.test_set_loader = test_set_loader
        self.properties = properties

    def test(self, test_run=0):
        assert isinstance(test_run, int), "test_run parameter should be int"

        if self.net.training:
            training = True
        else:
            training = False
        self.net.eval()

        if self.net.d2w.dist_type is NNEnum.L2_DIST:
            index_type = IndexType.L2_INDEX
        elif self.net.d2w.dist_type is NNEnum.COS_DIST:
            index_type = IndexType.COSINE_INDEX
        else:
            raise ValueError(f"Unknown d2w distance type: {self.net.d2w.dist_type}")
        index_config = IndexConfig(index_type=index_type)
        index: Index = IndexBuilder(index_config, index_dimension=2048).build()

        ids_list = []
        coords_list = []
        log.info("Starting to load db descriptors")
        for i, ds_tuple in enumerate(self.database_set_loader):
            if (i + 1) % self.properties.print_freq == 0 or i == 0 or (i + 1) == len(self.database_set_loader):
                log.info(f"Loading db descriptors: {i}/{len(self.database_set_loader)}")
            desc, ids, coordinates = ds_tuple
            desc = desc.cuda()
            with torch.no_grad():
                out = self.net(in_descriptors=desc)

            index.add_with_ids(out.cpu().numpy(), ids.numpy())

            ids_list.extend(ids.tolist())
            coords_list.extend(coordinates.tolist())

        log.info("Building and fitting localisation model")
        sigma = self.net.kde.sigma.item()
        m = self.net.d2w.m.item()
        model = LocalisationModel(LocalisationType.KDE, index, sigma=sigma, m=m, k=self.properties.k)
        model.fit(ids_list, coords_list)

        log.info("Starting to load test descriptors")
        queries = []
        q_ids = []
        q_coords = []
        for i, test_tuple in enumerate(self.test_set_loader):
            if (i + 1) % self.properties.print_freq == 0 or i == 0 or (i + 1) == len(self.database_set_loader):
                log.info(f"Loading test descriptors: {i + 1}/{len(self.test_set_loader)}")
            desc, ids, coordinates = test_tuple
            desc = desc.cuda()
            with torch.no_grad():
                out = self.net(in_descriptors=desc)
            queries.extend(out.cpu().tolist())
            q_ids.extend(ids.tolist())
            q_coords.extend(coordinates.tolist())

        log.info("Running localisation prediction")
        predicted_locations = model.predict(np.array(queries))
        results = _get_benchmark_results(predicted_locations, np.array(q_coords), np.array(q_ids))
        log.info(f"Prediction accuracy: {results.accuracy}")

        if self.properties.results_dir is not None:
            self._save_results(results, test_run)
        if training:
            # if network was training return to training mode
            self.net.train()
        return results

    def _save_results(self, results, test_run: int):
        if test_run <= 0:
            file_name = "results.json"
        else:
            file_name = f"results_{test_run}.json"
        if not os.path.isdir(self.properties.results_dir):
            os.makedirs(self.properties.results_dir)
        path = os.path.join(self.properties.results_dir, file_name)
        with open(path, "w") as f:
            json.dump(asdict(results), f)

    @classmethod
    def init(cls, config: ExtendedTestConfig):
        log.info("Initializing test service")
        net_config = _read_network_config(config.net_cfg_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
        net = Im2GPSNetworkInitializer(net_config).init_network()
        net.cuda()

        def worker_init_fn(worker):
            connect(db=config.db_config.db, host=config.db_config.host, port=config.db_config.port)

        if config.num_workers == 0:
            log.info("Connecting to database")
            worker_init_fn(0)

        database_ds = TestDataset(DatasetEnum.DATABASE, config.dataset_file)

        db_loader = DataLoader(
            database_ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )

        assert config.test_dataset in set(item.value for item in DatasetEnum), \
            f"Unknown dataset type {config.test_dataset}"
        test_ds_type = DatasetEnum(config.test_dataset)
        test_ds = TestDataset(test_ds_type, config.dataset_file)

        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )
        if config.load_path is not None:
            log.info(f"Loading network from {config.load_path}")
            net_state_dict = torch.load(config.load_path)
            net.load_state_dict(net_state_dict)
        test_service = cls(net, db_loader, test_loader, config.properties)
        log.info("test service initialization complete")
        return test_service


def _read_network_config(config_path):
    with open(config_path, 'r') as f:
        net_conf = yaml.load(f, yaml.FullLoader)
    return net_conf
