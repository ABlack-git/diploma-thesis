import logging
import time
import os
import requests
import datetime as dt
import io

from PIL import Image

from im2gps.data.flickrclient import FlickerClient, LoadDto
from im2gps.conf.config import ConfigRepo
from im2gps.conf.configschema import Config, CheckpointsConfig
from im2gps.data.flickr_repo import FlickrPhoto, FlickrCheckpoint, ImgUrl

log = logging.getLogger(__name__)


def collect_photos_metadata(checkpoint_type: str = None):
    cfg: Config = ConfigRepo().get(Config.__name__)
    flickr_client = FlickerClient(cfg.credentials)
    load_cfg = _get_metadata_checkpoint(cfg.checkpoints, checkpoint_type)
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


def download_photos(checkpoint_type, to_skip):
    cfg: Config = ConfigRepo().get(Config.__name__)

    root_dir = cfg.properties.data_directory
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    to_skip = _get_data_checkpoint(checkpoint_type, to_skip)

    photo: FlickrPhoto
    URL_TYPES = ('l', 'o', 'c', 'm')  # ordered by priority
    cursor = FlickrPhoto.objects.order_by('date_upload').skip(to_skip).batch_size(500).timeout(False)
    for i, photo in enumerate(cursor):
        img_url: ImgUrl = None

        for url_type in URL_TYPES:
            if url_type in photo.urls:
                img_url = photo.urls[url_type]

                response = _get_with_retry(img_url.url)
                if response.status_code != requests.status_codes.codes.ok:
                    log.warning(f"Response status is {response.status_code}. Skipping this photo "
                                f"(id: {photo.photo_id}).")
                    img_url = None
                    continue

                img = Image.open(io.BytesIO(response.content))
                folder = os.path.join(root_dir, f"{photo.date_upload.year}", f"{photo.date_upload.month}")
                if not os.path.exists(folder):
                    os.makedirs(folder)

                extension = os.path.split(img_url.url)[1].split(".")[1]
                file_name = f"{photo.photo_id}.{extension}"

                path = os.path.join(root_dir, folder, file_name)

                try:
                    log.info(f"Saving image number {i + to_skip} (id: {photo.photo_id}, "
                             f"upload_date: {photo.date_upload}) to {path}")
                    img.save(path)
                    break  # stop iterating links if found a valid one
                except OSError:
                    log.warning(f"Error when saving photo... Will retry if other links exist", exc_info=True)
                    img_url = None

        if img_url is None:
            log.warning(f"Image with id {photo.photo_id} doesn't have required url")

    del cursor


def _get_metadata_checkpoint(cfg: CheckpointsConfig, checkpoint_type: str = None) -> LoadDto:
    if checkpoint_type == 'from-db':
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
    elif checkpoint_type == 'from-config':
        log.info("Loading checkpoint from config...")
        load_cfg = cfg.meta_checkpoint
        return LoadDto(page=load_cfg.page, per_page=load_cfg.per_page,
                       start_date=dt.datetime.strptime(load_cfg.start_date, "%Y-%m-%d"),
                       interval_width=dt.timedelta(hours=load_cfg.interval_width))
    elif checkpoint_type is None:
        log.info("Starting download from the beginning")
        return LoadDto()
    else:
        raise ValueError(f"Unknown checkpoint type {checkpoint_type}, allowed values are from_db and "
                         f"from_config or should be empty")


def _get_data_checkpoint(checkpoint_type: str = None, to_skip: int = 0):
    if checkpoint_type == "from-db":
        raise NotImplemented("Loading from checkpoint not yet implemented")
    elif checkpoint_type == "from-cli":
        return to_skip
    elif checkpoint_type is None:
        return 0
    else:
        raise ValueError(f"Unknown checkpoint type {checkpoint_type}, allowed values are from_db and "
                         f"from_config or should be empty")


def _get_with_retry(url):
    num_retry = 5
    response = requests.get(url)
    if response.status_code != requests.status_codes.codes.ok:
        retry = 0
        while response.status_code != requests.status_codes.codes.ok and retry < num_retry:
            log.warning(f"Error getting {url}, status code was {response.status_code}. Retrying...")
            log.debug(f"Resplessonse body: {response.content}")
            time.sleep(1)
            response = requests.get(url)
            retry = retry + 1
    return response
