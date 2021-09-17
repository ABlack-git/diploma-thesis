from dataclasses import dataclass
from omegaconf import MISSING
from typing import List


# ----- CREDENTIALS CONFIG ----- #
@dataclass
class FlickrCreds:
    key: str = MISSING
    secret: str = MISSING


@dataclass
class CredsConfig:
    flickr: FlickrCreds = MISSING


# ----- DB CONFIG ----- #
@dataclass
class DBConfig:
    host: str = MISSING
    port: int = MISSING
    database: str = MISSING


# ----- CIRTORCH CONFIG ----- #
@dataclass
class CirtorchConfig:
    model_dir: str = MISSING
    img_resolution: int = MISSING
    gpu_id: int = MISSING


# ----- PROPERTIES CONFIG ----- #
@dataclass
class PropertiesConfig:
    output_dir: str = MISSING
    data_directory: str = MISSING
    indices_dir: str = MISSING
    split_file: str = MISSING


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


# ----- MODEL CONFIG ----- #
@dataclass
class ModelConfig:
    localisation_type: str = MISSING
    sigma: float = MISSING
    m: float = MISSING
    k: int = MISSING


# ----- INDEX CONFIG ----- #
@dataclass
class IndexConfig:
    index_dir: str = MISSING
    index_type: str = MISSING
    gpu_enabled: bool = MISSING
    gpu_id: int = MISSING


# ----- TEST CONFIG ----- #
@dataclass
class LocalisationTestConfig:
    query_dataset: str = MISSING
    extended_results: bool = MISSING
    save_results: bool = MISSING
    save_path: bool = MISSING


@dataclass
class Config:
    credentials: CredsConfig = MISSING
    db: DBConfig = MISSING
    cirtorch: CirtorchConfig = MISSING
    localisation_model: ModelConfig = MISSING
    index_config: IndexConfig = MISSING
    test_config: LocalisationTestConfig = MISSING
    properties: PropertiesConfig = MISSING
    checkpoints: CheckpointsConfig = MISSING
    filters: FiltersConfig = MISSING
