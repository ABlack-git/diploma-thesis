import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from im2gps.data.descriptors import MongoDescriptor, DatasetEnum


class TrainDataset(Dataset):
    def __init__(self, ds_file_path):
        with open(ds_file_path, 'r') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        query_id = self.dataset[index]['query']
        neighbours_ids = self.dataset[index]['neighbours']
        if 'target' in self.dataset[index]:
            target_id = self.dataset[index]['target']
            target_coords = MongoDescriptor.objects(photo_id=target_id).first().coordinates
        else:
            target_coords = None

        query_doc = MongoDescriptor.objects(photo_id=query_id).first()
        query = query_doc.descriptor
        ground_truth_coord = query_doc.coordinates

        neighbour_docs = MongoDescriptor.objects(photo_id__in=neighbours_ids)
        neighbours = []
        n_coords = []
        for n_doc in neighbour_docs:
            neighbours.append(n_doc.descriptor)
            n_coords.append(n_doc.coordinates)

        query = torch.tensor(query, dtype=torch.float32)
        neighbours = torch.tensor(neighbours, dtype=torch.float32)
        if target_coords is None:
            ground_truth_coord = torch.tensor(ground_truth_coord, dtype=torch.float32)
        else:
            ground_truth_coord = torch.tensor(target_coords, dtype=torch.float32)
        n_coords = torch.tensor(n_coords, dtype=torch.float32)
        query_id = torch.tensor(query_id)
        neighbours_ids = torch.tensor(neighbours_ids)
        return query, neighbours, ground_truth_coord, n_coords, query_id, neighbours_ids


class TestDataset(Dataset):
    def __init__(self, dataset_type: DatasetEnum, datasets_file):
        self.dataset_type = dataset_type
        with open(datasets_file, 'r') as f:
            file = json.load(f)
        self.__ids = file[self.dataset_type.value]
        del file

    def __len__(self):
        return len(self.__ids)

    def __getitem__(self, index) -> T_co:
        photo_id = self.__ids[index]
        doc: MongoDescriptor = MongoDescriptor.objects(photo_id=photo_id).first()
        descriptor = torch.tensor(doc.descriptor, dtype=torch.float32)
        coordinates = torch.tensor(doc.coordinates, dtype=torch.float32)

        return descriptor, photo_id, coordinates
