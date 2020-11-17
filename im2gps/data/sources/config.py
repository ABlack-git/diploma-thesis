from omegaconf import MISSING
from dataclasses import dataclass
from typing import List


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
class LoadConfig:
    load_from_db: bool = MISSING
    load_from_config: LoadFromConfig = MISSING


@dataclass
class AppConfig:
    load: LoadConfig = MISSING


@dataclass
class DSConfig:
    creds: CredsConfig = MISSING
    filters: FilterConfig = MISSING
    db: MongoConfig = MISSING
    app: AppConfig = MISSING
