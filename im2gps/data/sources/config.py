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
class AppConfig:
    load_checkpoint: bool = MISSING


@dataclass
class DSConfig:
    creds: CredsConfig = MISSING
    filters: FilterConfig = MISSING
    db: MongoConfig = MISSING
    app: AppConfig = MISSING
