from CNNBenchUtils.Datasets.BaseDatasetHandler import BaseDatasetHandler
from CNNBenchUtils.Datasets.Dataset import Dataset


class BaseDatasetFileLoader:
    def __init__(self, dataset=None):
        self.dataset = None
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif dataset is not None:
            raise TypeError("Invalid Dataset!")

    def open(self, filename):
        if self.dataset is None:
            raise ValueError("No Dataset!")

        if self.dataset.dataset_handler is None:
            raise ValueError("No Dataset Handler!")

        if not self.dataset.dataset_handler.loaded:
            raise FileNotFoundError("Dataset not loaded")

        return None
