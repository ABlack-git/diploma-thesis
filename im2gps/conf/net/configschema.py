from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Dict, Any, Optional


@dataclass
class DBConfig:
    db: str = MISSING
    host: str = MISSING
    port: int = MISSING


@dataclass
class DataConfig:
    db_config: DBConfig = MISSING
    train_ds: str = MISSING
    val_ds: str = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class TrainProperties:
    base_dir: str = MISSING
    print_freq: int = MISSING
    sma_window: int = MISSING
    num_epochs: int = MISSING
    gpu_id: int = MISSING
    validate: bool = MISSING
    test: bool = MISSING
    save_checkpoint: bool = MISSING
    summary_writer: bool = MISSING
    test_freq: int = MISSING


@dataclass
class TestProperties:
    results_dir: str = MISSING
    k: int = MISSING
    print_freq: int = MISSING
    num_workers: int = MISSING


@dataclass
class TestConfig:
    dataset_file: str = MISSING
    test_dataset: str = MISSING
    properties: TestProperties = MISSING
    batch_size: int = MISSING


@dataclass
class TrainConfig:
    net_config_path: str = MISSING
    optimizer_config: Optional[Dict[Any, Any]] = field(default_factory=dict)
    scheduler_config: Optional[Dict[Any, Any]] = field(default_factory=dict)
    data_config: DataConfig = MISSING
    properties: TrainProperties = MISSING
    test_config: Optional[TestConfig] = None


@dataclass
class ExtendedTestConfig(TestConfig):
    db_config: DBConfig = MISSING
    num_workers: int = MISSING
    net_cfg_path: str = MISSING
    load_path: Optional[str] = None
    gpu_id: int = -1
