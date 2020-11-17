import sys
import flickrapi
import logging
import datetime as dt
import time

from im2gps.data.sources.dtos import PhotoDto, LoadDto
from im2gps.configutils import ConfigRepo
from im2gps.data.sources.config import DSConfig
from im2gps.data.sources.repo import FlickrPhoto, FlickrCheckpoint
from im2gps.data.sources.exceptions import FlickrClientError, DownloadError

from typing import Generator

log = logging.getLogger(__name__)


class FlickerClient:
    _MAX_RESULTS = 4000
    _MIN_RESULTS = 3000
    _DATE_FORMAT = "%d-%m-%Y %H:%M:00"
    """
    Flickr photos.search request will return only first 4000 search results. This class is a wrapper to flickrapi, 
    which solves the mentioned problem by splitting requests by date range.  
    """

    def __init__(self, cfg: DSConfig):
        self.flickr = flickrapi.FlickrAPI(api_key=cfg.creds.flickr.key, secret=cfg.creds.flickr.secret,
                                          format='parsed-json')
        self.num_retries = 5

    def search_photos(self, start_date: dt.datetime = None,
                      max_date: dt.date = None,
                      interval_width: dt.timedelta = None,
                      page: int = None,
                      **kwargs) -> Generator[PhotoDto, None, None]:
        if 'min_upload_date' in kwargs or 'max_upload_date' in kwargs:
            raise ValueError('min_upload_date or max_upload_date should not be specified')

        # setup default args
        start_date = dt.datetime(2005, 1, 1) if start_date is None else start_date
        max_date = dt.datetime.combine(dt.date.today(), dt.time()) if max_date is None else max_date
        interval_width = dt.timedelta(days=365) if interval_width is None else interval_width
        page = 1 if page is None else page

        current_start = start_date
        i_width = interval_width
        # loop over dates
        while current_start < max_date:
            i_width = self._find_date_range(start_date=current_start,
                                            interval_width=i_width,
                                            max_allowed_date=max_date,
                                            **kwargs)
            min_upload_date = current_start.strftime(self._DATE_FORMAT)
            max_upload_date = (current_start + i_width).strftime(self._DATE_FORMAT)
            # loop over pages
            current_page = page
            max_page = sys.maxsize
            while current_page <= max_page:
                result = self.search_photo_with_retry(page=current_page, min_upload_date=min_upload_date,
                                                      max_upload_date=max_upload_date, **kwargs)
                log.debug(f"page: {result['photos']['page']}, pages: {result['photos']['pages']},"
                          f"perpage: {result['photos']['perpage']}, total: {result['photos']['total']}")
                if max_page == sys.maxsize:
                    max_page = int(result['photos']['pages'])
                # loop over each photo on a page
                for photo in result['photos']['photo']:
                    yield PhotoDto(photo=photo, page=current_page, per_page=int(result['photos']['perpage']),
                                   start_date=current_start, interval_width=i_width)

                current_page = current_page + 1
            page = 1
            current_start = current_start + i_width + dt.timedelta(days=1)

    def _find_date_range(self, start_date: dt.datetime,
                         interval_width: dt.timedelta,
                         max_allowed_date: dt.datetime,
                         **kwargs) -> dt.timedelta:
        """
        Finds suitable date interval s.t. flickapi search will return less than MAX_RESULTS. Performs a binary search
        on ``min_date + interval_width`` interval.
        :param start_date: start date of the interval
        :param interval_width: width of time interval
        :param max_allowed_date: max query date
        :param kwargs: search parameters from flickrapi
        :return: a width of time interval, s.t. for given time interval flickrapi will return less results than
         MAX_RESULTS.
        """
        for key in ('per_page', 'page', 'extras'):
            if key in kwargs:
                del kwargs[key]

        max_date = max_allowed_date if start_date + interval_width >= max_allowed_date else start_date + interval_width
        min_date = start_date
        result = self.search_photo_with_retry(min_upload_date=start_date.strftime(self._DATE_FORMAT),
                                              max_upload_date=max_date.strftime(self._DATE_FORMAT),
                                              **kwargs)

        total = int(result['photos']['total'])
        increase_factor = 1.25
        while total < self._MIN_RESULTS:
            if start_date + interval_width * increase_factor > max_allowed_date:
                break
            else:
                log.debug(f"Increasing interval width. {total} is less than {self._MIN_RESULTS}")
                interval_width = interval_width * increase_factor
                max_date = start_date + interval_width
            result = self.search_photo_with_retry(min_upload_date=start_date.strftime(self._DATE_FORMAT),
                                                  max_upload_date=max_date.strftime(self._DATE_FORMAT),
                                                  **kwargs)
            total = int(result['photos']['total'])

        if total <= self._MAX_RESULTS:
            return max_date - start_date

        while min_date <= max_date:
            delta = (max_date - min_date) // 2
            delta = delta if delta >= dt.timedelta(hours=6) else dt.timedelta(hours=6)
            mid_date = min_date + delta
            log.debug(f"Checking interval from {start_date} to {mid_date}")
            result = self.search_photo_with_retry(min_upload_date=start_date.strftime(self._DATE_FORMAT),
                                                  max_upload_date=mid_date.strftime(self._DATE_FORMAT),
                                                  **kwargs)
            total = int(result['photos']['total'])
            log.debug(f"min_date: {min_date}, max_date: {max_date}, mid_date: {mid_date}, number of photos: {total}")
            if self._MIN_RESULTS <= total <= self._MAX_RESULTS:
                return mid_date - start_date
            elif total < self._MIN_RESULTS:
                min_date = mid_date
            else:  # total > self._MAX_RESULTS
                max_date = mid_date
        raise DownloadError(f"Could not find suitable date range. Total number of photos: {total}, "
                            f"min_date: {min_date}, max_date: {max_date}")

    def search_photo_with_retry(self, **kwargs) -> dict:
        response = self.flickr.photos.search(**kwargs)
        # TODO: remove response['photos']['total'] is None from condition if possible, not generic
        if response is None or response['stat'] == 'fail' or response['photos']['total'] is None:
            for i in range(self.num_retries):
                log.warning(f"Received {response} from Flickr API, retrying... {i + 1}/{self.num_retries}")
                time.sleep(1)
                response = self.flickr.photos.search(**kwargs)
                if response is not None and response['stat'] == 'ok' and response['photos']['total'] is not None:
                    return response
            raise FlickrClientError(f"Flickr API call has failed. Response: {response}")
        return response


def _get_loading_params(cfg: DSConfig) -> LoadDto:
    if cfg.app.load.load_from_db:
        log.info("Loading checkpoint from database...")
        checkpoint = FlickrCheckpoint.load_latest()
        if checkpoint is not None:
            log.info(f"Found checkpoint. start date: {checkpoint.start_date}, "
                     f"end date: {checkpoint.start_date + dt.timedelta(checkpoint.interval_width)}, "
                     f"page: {checkpoint.page}")
            return LoadDto(page=checkpoint.page, per_page=checkpoint.per_page, start_date=checkpoint.start_date,
                           interval_width=dt.timedelta(days=checkpoint.interval_width))
        else:
            log.info(f"Checkpoint doesn't exist in database. Starting download from the beginning...")
            return LoadDto()

    if cfg.app.load.load_from_config.to_load:
        log.info("Loading checkpoint from config...")
        load_cfg = cfg.app.load.load_from_config
        return LoadDto(page=load_cfg.page, per_page=load_cfg.per_page,
                       start_date=dt.datetime.strptime(load_cfg.start_date, "%Y-%m-%d"),
                       interval_width=dt.timedelta(hours=load_cfg.interval_width))

    log.info("Starting download from the beginning")
    return LoadDto()


def collect_photos_metadata():
    cr = ConfigRepo()
    cfg: DSConfig = cr.get(DSConfig.__name__)
    flickr_client = FlickerClient(cfg)
    load_cfg = _get_loading_params(cfg)
    for i, photoDto in enumerate(flickr_client.search_photos(
            start_date=load_cfg.start_date,
            interval_width=load_cfg.interval_width,
            page=load_cfg.page,
            has_geo=cfg.filters.flickr.has_geo,
            tags=",".join(cfg.filters.flickr.tags),
            media=cfg.filters.flickr.media,
            extras='geo,tags,owner_name,date_upload,date_taken,url_m,url_c,url_l,url_o',
            accuracy=cfg.filters.flickr.accuracy,
            tag_mode='bool'
    )):
        flickr_photo: FlickrPhoto = FlickrPhoto.from_dict(photoDto.photo)
        flickr_photo.save()
        if (i + 1) % 1000 == 0:
            log.info(f"Saving checkpoint. page: {photoDto.page}, start date: {photoDto.start_date}, "
                     f"end date: {photoDto.start_date + photoDto.interval_width}, per page: {photoDto.per_page}")
            chkpt = FlickrCheckpoint.from_dto(photoDto)
            chkpt.save()
