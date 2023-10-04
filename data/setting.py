import os
from dataclasses import dataclass

@dataclass
class DatabaseSetting():
    classes: int
    dataset_path: str
    img_path: str
    train_datasets: list
    test_datasets: list

    @property
    def train_pathes(self):
        return [os.path.join(self.dataset_path, f) for f in self.train_datasets]

    @property
    def test_pathes(self):
        if isinstance(self.test_datasets, list):
            return [os.path.join(self.dataset_path, f) for f in self.test_datasets]
        else:
            return None

    def __repr__(self) -> str:
        tmp = f'-------------------------------------------\n'\
              f'Database Info:\n'\
              f'Classification For: {self.classes} classes\n'\
              f'Dataset loc in: {self.dataset_path}\n'\
              f'Image loc in: {self.img_path}\n'\
              f'Train in: {self.train_datasets}\n'\
              f'Test in: {self.test_datasets}\n'
        return tmp