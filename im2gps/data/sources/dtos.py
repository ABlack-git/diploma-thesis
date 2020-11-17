import datetime as dt

from dataclasses import dataclass, field


@dataclass
class LoadDto:
    page: int = field(default=None)
    per_page: int = field(default=None)
    start_date: dt.datetime = field(default=None)
    interval_width: dt.timedelta = field(default=None)


@dataclass
class PhotoDto(LoadDto):
    photo: dict = field(default=None)
