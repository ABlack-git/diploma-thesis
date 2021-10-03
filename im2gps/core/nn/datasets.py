import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from im2gps.data.descriptors import MongoDescriptor


class DescriptorsDataset(Dataset):
    def __init__(self, ds_file_path):
        with open(ds_file_path, 'r') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        query_id = self.dataset[index]['query']
        neighbours_ids = self.dataset[index]['neighbours']
        query_doc = MongoDescriptor.objects(photo_id=query_id).first()
        query = query_doc.descriptor
        q_coords = query_doc.coordinates

        neighbour_docs = MongoDescriptor.objects(photo_id__in=neighbours_ids)
        neighbours = []
        n_coords = []
        for n_doc in neighbour_docs:
            neighbours.append(n_doc.descriptor)
            n_coords.append(n_doc.coordinates)

        query = torch.tensor(query, dtype=torch.float32)
        neighbours = torch.tensor(neighbours, dtype=torch.float32)
        q_coords = torch.tensor(q_coords, dtype=torch.float32)
        n_coords = torch.tensor(n_coords, dtype=torch.float32)
        query_id = torch.tensor(query_id)
        neighbours_ids = torch.tensor(neighbours_ids)
        return query, neighbours, q_coords, n_coords, query_id, neighbours_ids
