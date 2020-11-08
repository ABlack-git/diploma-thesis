import sys
import flickrapi
import logging
import datetime as dt

from mongoengine.errors import NotUniqueError

from im2gps.configutils import ConfigRepo
from im2gps.data.sources.config import DSConfig
from im2gps.data.sources.repo import FlickrPhoto

log = logging.getLogger(__name__)


class FlickerClient:
    _MAX_RESULTS = 3900
    _MIN_RESULTS = 3000
    _DATE_FORMAT = "%d-%m-%Y"
    """
    Flickr photos.search request will return only first 4000 search results. This class is a wrapper to flickrapi, 
    which solves the mentioned problem by splitting requests by date range.  
    """

    def __init__(self):
        cr = ConfigRepo()
        cfg: DSConfig = cr.get(DSConfig.__name__)
        self.flickr = flickrapi.FlickrAPI(api_key=cfg.creds.flickr.key, secret=cfg.creds.flickr.secret,
                                          format='parsed-json')

    def search_photos(self, start_date: dt.date = dt.date(2005, 1, 1), max_date: dt.date = dt.date.today(), **kwargs):
        if 'min_upload_date' in kwargs or 'max_upload_date' in kwargs:
            raise ValueError('min_upload_date or max_upload_date should not be specified')
        current_start = start_date
        interval_width = dt.timedelta(days=365)
        # loop over dates
        while current_start < max_date:
            interval_width = self._find_date_range(start_date=current_start,
                                                   interval_width=interval_width,
                                                   max_allowed_date=max_date,
                                                   **kwargs)
            min_upload_date = current_start.strftime(self._DATE_FORMAT)
            max_upload_date = (current_start + interval_width).strftime(self._DATE_FORMAT)
            # loop over pages
            page = 1
            max_page = sys.maxsize
            while page <= max_page:
                result = self.flickr.photos.search(page=page, min_upload_date=min_upload_date,
                                                   max_upload_date=max_upload_date, **kwargs)
                log.debug(f"Page: {result['photos']['page']}, pages: {result['photos']['pages']},"
                          f"perpage: {result['photos']['perpage']}, total: {result['photos']['total']}")
                if max_page == sys.maxsize:
                    max_page = int(result['photos']['pages'])
                # loop over each photo on a page
                for photo in result['photos']['photo']:
                    # TODO: wrap photo in dataclass with metadata
                    yield photo

                page = page + 1
            current_start = current_start + interval_width + dt.timedelta(days=1)

    def _find_date_range(self, start_date: dt.date,
                         interval_width: dt.timedelta,
                         max_allowed_date: dt.date,
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
        result = self.flickr.photos.search(min_upload_date=start_date.strftime(self._DATE_FORMAT),
                                           max_upload_date=max_date.strftime(self._DATE_FORMAT),
                                           **kwargs)
        total = int(result['photos']['total'])
        if total <= self._MAX_RESULTS:
            return max_date - start_date
        while min_date <= max_date:
            mid_date = min_date + (max_date - min_date) // 2
            result = self.flickr.photos.search(min_upload_date=start_date.strftime(self._DATE_FORMAT),
                                               max_upload_date=mid_date.strftime(self._DATE_FORMAT),
                                               **kwargs)
            total = int(result['photos']['total'])
            if self._MIN_RESULTS <= total <= self._MAX_RESULTS:
                return mid_date - start_date
            elif total < self._MIN_RESULTS:
                min_date = mid_date + dt.timedelta(days=1)
            else:  # total > self._MAX_RESULTS
                max_date = mid_date - dt.timedelta(days=1)
        raise Exception("Could not find suitable date range")


def collect_photos():
    flickr_client = FlickerClient()
    cr = ConfigRepo()
    cfg: DSConfig = cr.get(DSConfig.__name__)
    for photo in flickr_client.search_photos(
            has_geo=cfg.filters.flickr.has_geo,
            tags=",".join(cfg.filters.flickr.tags),
            media=cfg.filters.flickr.media,
            extras='geo,tags,owner_name,date_upload,date_taken,url_m,url_c,url_l,url_o',
            tag_mode='bool'
    ):
        flickr_photo: FlickrPhoto = FlickrPhoto.from_dict(photo)
        try:
            flickr_photo.save(force_insert=True)
        except NotUniqueError as e:
            log.warning(f"Photo with id {flickr_photo.photo_id} already exists")
