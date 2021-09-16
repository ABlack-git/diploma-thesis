from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
from im2gps.data.old_descriptors import DescriptorsTable
from tqdm import tqdm
from mongoengine import connect

path_to_descriptors = {
    DatasetEnum.DATABASE: "/Users/zakharca/Documents/Study/thesis/descriptors/512_flickr_descriptor_train.h5",
    DatasetEnum.TRAIN_QUERY: "/Users/zakharca/Documents/Study/thesis/descriptors/512_flickr_descriptor_train_q.h5",
    DatasetEnum.TEST_QUERY: "/Users/zakharca/Documents/Study/thesis/descriptors/512_flickr_descriptor_test_q.h5",
    DatasetEnum.VALIDATION_QUERY: "/Users/zakharca/Documents/Study/thesis/descriptors/512_flickr_descriptor_val_q.h5"
}


def migrate():
    tables_pbar = tqdm(path_to_descriptors.items(), desc="Processing ")
    for dataset, path in tables_pbar:
        tables_pbar.set_description(f"Processing {dataset.value} dataset")
        with DescriptorsTable(path, 2048) as table:
            for descriptor in tqdm(table.iterrows(), leave=False, total=len(table)):
                desc = MongoDescriptor(photo_id=descriptor.photo_id, dataset=dataset)
                desc.coordinates = [descriptor.lon, descriptor.lat]
                desc.descriptor = descriptor.descriptor
                desc.save()


if __name__ == '__main__':
    print("Connecting to db")
    connect(db="im2gps", host="localhost", port=51998)
    print("Starting migration")
    migrate()
