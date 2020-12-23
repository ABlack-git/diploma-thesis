from dataclasses import dataclass
from omegaconf import MISSING
from typing import List


# ----- DATA CONFIG ----- #
@dataclass
class FlickrCreds:
    key: str = MISSING
    secret: str = MISSING


@dataclass
class CredsConfig:
    flickr: FlickrCreds = MISSING


@dataclass
class DBConfig:
    host: str = MISSING
    port: int = MISSING
    database: str = MISSING


@dataclass
class DatasetsConfig:
    split_file: str = MISSING
    train: str = MISSING
    train_queries: str = MISSING
    test_queries: str = MISSING
    validation_queries: str = MISSING


@dataclass
class DataConfig:
    data_directory: str = MISSING
    creds: CredsConfig = MISSING
    db: DBConfig = MISSING
    datasets: DatasetsConfig = MISSING


# ----- CIRTORCH CONFIG ----- #
@dataclass
class CirtorchConfig:
    model_dir: str = MISSING
    img_resolution: int = MISSING


# ----- PROPERTIES CONFIG ----- #
@dataclass
class PropertiesConfig:
    gpu_id: int = MISSING
    output_dir: str = MISSING


# ----- CHECKPOINTS CONFIG ----- #
@dataclass
class MetaCheckpoint:
    page: int = MISSING
    per_page: int = MISSING
    start_date: str = MISSING
    interval_width: int = MISSING


@dataclass
class DescriptorCheckpoint:
    checkpoint_path: str = MISSING


@dataclass
class CheckpointsConfig:
    meta_checkpoint: MetaCheckpoint = MISSING
    descriptor_checkpoint: DescriptorCheckpoint = MISSING


# ----- FILTERS CONFIG ----- #
@dataclass
class FlickFilter:
    tags: List[str] = MISSING
    place_id: str = MISSING
    woeid: int = MISSING
    media: str = MISSING
    has_geo: int = MISSING
    accuracy: int = MISSING


@dataclass
class FiltersConfig:
    flickr: FlickFilter = MISSING


@dataclass
class Config:
    data: DataConfig = MISSING
    cirtorch: CirtorchConfig = MISSING
    properties: PropertiesConfig = MISSING
    checkpoints: CheckpointsConfig = MISSING
    filters: FiltersConfig = MISSING
