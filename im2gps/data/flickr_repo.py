import datetime as dt

from im2gps.data.sources.dtos import PhotoDto
from mongoengine import Document, EmbeddedDocument, EmbeddedDocumentField, LongField, StringField, DateTimeField, \
    ListField, IntField, PointField, MapField


class GeoInfo(EmbeddedDocument):
    coords = PointField(required=True)
    accuracy = IntField(required=True, default=0)
    context = IntField(required=True, default=0)
    place_id = StringField(required=False)
    woe_id = LongField(required=False)


class ImgUrl(EmbeddedDocument):
    url_types = ('m', 'c', 'l', 'o')
    url_type = StringField(required=True, choices=url_types)
    url = StringField(required=True)
    height = IntField(required=True)
    width = IntField(required=True)


class FlickrPhoto(Document):
    photo_id = LongField(primary_key=True)
    owner = StringField(required=True)
    secret = StringField(required=True)
    server = StringField(required=True)
    title = StringField(required=False)
    date_upload = DateTimeField(required=True)
    owner_name = StringField(required=False)
    tags = ListField(field=StringField(), default=list)
    geo = EmbeddedDocumentField(GeoInfo)
    urls = MapField(EmbeddedDocumentField(ImgUrl))
    meta = {'collection': 'flickr'}

    @staticmethod
    def from_dict(data: dict) -> 'FlickrPhoto':
        geo_dict = {'coords': [float(data['longitude']), float(data['latitude'])]}
        if 'accuracy' in data:
            geo_dict['accuracy'] = int(data['accuracy'])
        if 'context' in data:
            geo_dict['context'] = int(data['context'])
        if 'place_id' in data:
            geo_dict['place_id'] = data['place_id']
        if 'woeid' in data:
            geo_dict['woe_id'] = int(data['woeid'])
        geo = GeoInfo(**geo_dict)
        flickr_photo_dict = {'photo_id': int(data['id']), 'owner': data['owner'], 'secret': data['secret'],
                             'server': data['server'],
                             'date_upload': dt.datetime.fromtimestamp(int(data['dateupload'])),
                             'geo': geo,
                             'tags': data['tags'].split(),
                             'urls': {}}
        if 'title' in data:
            flickr_photo_dict['title'] = data['title']
        if 'ownername' in data:
            flickr_photo_dict['owner_name'] = data['ownername']
        for url_type in ImgUrl.url_types:
            if f'url_{url_type}' in data:
                img_url = ImgUrl(url_type=url_type, url=data[f'url_{url_type}'], height=data[f'height_{url_type}'],
                                 width=data[f'width_{url_type}'])
                flickr_photo_dict['urls'][url_type] = img_url
        return FlickrPhoto(**flickr_photo_dict)


class FlickrCheckpoint(Document):
    page = IntField(required=True)
    per_page = IntField(required=True)
    start_date = DateTimeField(required=True)
    interval_width = IntField(required=True)
    created_on = DateTimeField(required=True, default=dt.datetime.now)

    meta = {'collection': 'flickr.checkpoint'}

    @staticmethod
    def from_dto(dto: PhotoDto):
        return FlickrCheckpoint(page=dto.page, per_page=dto.per_page, start_date=dto.start_date,
                                interval_width=dto.interval_width.seconds)

    @classmethod
    def load_latest(cls) -> 'FlickrCheckpoint':
        return cls.objects.order_by('-created_on').first()
