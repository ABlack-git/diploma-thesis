import datetime as dt

from dataclasses import dataclass


@dataclass
class PhotoDto:
    photo: dict
    page: int
    per_page: int
    start_date: dt.date
    interval_width: dt.timedelta
