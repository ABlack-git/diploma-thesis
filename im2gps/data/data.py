import os
import pickle

from typing import Tuple, List


def get_img_id_from_path(path: str):
    return int(os.path.basename(path).split("_")[0])


class DirectoryIterator:
    """
    This class is a simple iterator over directories that returns paths to files. State of this iterator can be saved
    and restored to resume from
    """

    def __init__(self, root_dir, checkpoint_path, follow_hidden=False):
        self.queue: list = [root_dir]
        self.checkpoint_path = checkpoint_path
        self.current = None
        self.follow_hidden = follow_hidden

    def __iter__(self):
        return self

    def __next__(self):
        self.__raise_sto_iter_on_empty_queue()
        path = self.queue.pop(0)
        path = self.__on_hidden_file(path)
        while os.path.isdir(path):
            dirs, non_dirs = self.__get_children(path)
            self.queue.extend(non_dirs)  # add files first
            self.queue.extend(dirs)
            self.__raise_sto_iter_on_empty_queue()
            path = self.queue.pop(0)
            path = self.__on_hidden_file(path)
        self.current = path
        return path

    def __on_hidden_file(self, path):
        if self.__path_is_hidden(path):
            if self.follow_hidden:
                return path
            else:
                new_path = self.queue.pop(0)
                while self.__path_is_hidden(new_path):
                    new_path = self.queue.pop(0)
                return new_path
        return path

    def __path_is_hidden(self, path):
        name = os.path.basename(path)
        return name.startswith('.')

    def __raise_sto_iter_on_empty_queue(self):
        if not self.queue:
            raise StopIteration

    def __get_children(self, path) -> Tuple[List[str], List[str]]:
        names = os.listdir(path)
        dirs, non_dirs = [], []
        for name in names:
            if os.path.isdir(os.path.join(path, name)):
                dirs.append(os.path.join(path, name))
            else:
                non_dirs.append(os.path.join(path, name))
        return dirs, non_dirs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save checkpoint on exception
        if exc_type is not None:
            self.queue.insert(0, self.current)
            self.save_checkpoint(self.checkpoint_path)

    def save_checkpoint(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_or_create(cls, root_dir, checkpoint_path, **kwargs) -> 'DirectoryIterator':
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as file:
                return pickle.load(file)
        else:
            return cls(root_dir, checkpoint_path, **kwargs)
