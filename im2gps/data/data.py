import os


def get_image_paths(root_dir, extensions=('.jpg', '.jpeg', '.png')):
    for root, subdir, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)
