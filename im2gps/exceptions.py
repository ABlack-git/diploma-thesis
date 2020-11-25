class Im2GpsError(Exception):
    pass


class FlickrClientError(Im2GpsError):
    pass


class DownloadError(Im2GpsError):
    pass


class FlickrPhotoNotFound(Im2GpsError):
    pass
