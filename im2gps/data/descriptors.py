import numpy as np
import pickle
import enum
import mongoengine as me
import logging

log = logging.getLogger(__name__)


class DatasetEnum(enum.Enum):
    TEST_QUERY = 'test_query'
    DATABASE = 'database'
    VALIDATION_QUERY = 'validation_query'
    TRAIN_QUERY = 'train_query'
    GENERAL = 'general'


class MongoDescriptor(me.Document):
    photo_id = me.LongField(primary_key=True, required=True)
    coords = me.PointField(db_field="coordinates", required=True)
    dataset = me.EnumField(DatasetEnum, required=True)
    binary_descriptor = me.BinaryField(required=True, db_field="descriptor")
    meta = {'collection': 'flickr.descriptors_512'}

    @property
    def descriptor(self):
        if self.binary_descriptor is not None:
            return pickle.loads(self.binary_descriptor)
        else:
            return None

    @descriptor.setter
    def descriptor(self, descriptor_array: np.ndarray):
        if not isinstance(descriptor_array, np.ndarray):
            raise ValueError("Descriptor array should be of type np.ndarray")
        self.binary_descriptor = pickle.dumps(descriptor_array, protocol=2)

    @property
    def coordinates(self):
        return self.coords['coordinates']

    @coordinates.setter
    def coordinates(self, coord_dict):
        if 'lat' not in coord_dict or 'lng' not in coord_dict:
            raise ValueError("error setting coordinates, lat or lng is not in dict")
        self.coords = [coord_dict['lng'], coord_dict['lat']]

    def set_coordinates(self, lng, lat):
        self.coords = [lng, lat]

    @classmethod
    def get_ids_and_coords(cls, dataset: DatasetEnum) -> dict:
        if isinstance(dataset, DatasetEnum):
            objects = cls.objects(dataset=dataset)
        else:
            raise ValueError(f"dataset should be one of DatasetEnum, was {dataset}")
        data = dict()
        total = objects.count()

        log.debug(f"Getting {dataset.value} dataset from db. Total number of documents in db {total}")

        ids = np.zeros(shape=total, dtype=int)
        coordinates = np.zeros(shape=(total, 2))

        batch_size = 100000
        for i, descriptor in enumerate(objects.batch_size(batch_size).only('photo_id', 'coords')):
            ids[i] = descriptor.photo_id
            coordinates[i, :] = descriptor.coordinates

            if (i + 1) % batch_size == 0:
                log.debug(f"Processed {i + 1} documents")

        data['ids'] = ids
        data['coordinates'] = coordinates
        return data

    @classmethod
    def get_as_data_dict(cls, dataset: DatasetEnum = None) -> dict:
        if isinstance(dataset, DatasetEnum):
            objects = cls.objects(dataset=dataset)
        else:
            raise ValueError(f"dataset should be one of DatasetEnum, was {dataset}")
        data = dict()
        total = objects.count()
        dim = cls.objects.first().descriptor.shape[0]

        log.debug(f"Getting {dataset.value} dataset from db. Total number of documents in db {total}")

        descriptors = np.zeros(shape=(total, dim))
        ids = np.zeros(shape=total, dtype=int)
        coordinates = np.zeros(shape=(total, 2))
        batch_size = 50000
        for i, descriptor in enumerate(objects.batch_size(batch_size)):
            descriptors[i, :] = descriptor.descriptor
            ids[i] = descriptor.photo_id
            coordinates[i, :] = descriptor.coordinates

            if (i + 1) % batch_size == 0:
                log.debug(f"Processed {i + 1} documents")

        data['ids'] = ids
        data['descriptors'] = descriptors
        data['coordinates'] = coordinates
        return data
