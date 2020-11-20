from omegaconf import MISSING
from dataclasses import dataclass
from typing import List, Union


@dataclass
class FlickrProperties:
    key: str = MISSING
    secret: str = MISSING


@dataclass
class MapillaryProperties:
    clientId: str = MISSING


@dataclass
class CredsConfig:
    flickr: FlickrProperties = MISSING
    mapillary: MapillaryProperties = MISSING


@dataclass
class FlickFilter:
    tags: List[str] = MISSING
    place_id: str = MISSING
    woeid: int = MISSING
    media: str = MISSING
    has_geo: int = MISSING
    accuracy: int = MISSING


@dataclass
class FilterConfig:
    flickr: FlickFilter = MISSING


@dataclass
class MongoConfig:
    host: str = MISSING
    port: int = MISSING
    database: str = MISSING


@dataclass
class LoadFromConfig:
    to_load: bool = MISSING
    page: int = MISSING
    per_page: int = MISSING
    start_date: str = MISSING
    interval_width: int = MISSING


@dataclass
class MetaCheckpoint:
    page: int = MISSING
    per_page: int = MISSING
    start_date: str = MISSING
    interval_width: int = MISSING


@dataclass
class DataCheckpoint:
    skip: str = MISSING


@dataclass
class AppConfig:
    download: str = MISSING
    checkpoint_type: str = MISSING
    data_directory: str = MISSING


@dataclass
class DSConfig:
    creds: CredsConfig = MISSING
    filters: FilterConfig = MISSING
    db: MongoConfig = MISSING
    app: AppConfig = MISSING
    checkpoint: Union[MetaCheckpoint, DataCheckpoint] = MISSING
