from CNNBenchUtils.Datasets.BaseDatasetHandler import BaseDatasetHandler


class BaseFileLoader:
    def __init__(self, dataset_handler=None):
        if issubclass(type(dataset_handler), BaseDatasetHandler):
            self.dataset_handler = dataset_handler
        elif dataset_handler is not None:
            raise TypeError("Invalid Dataset Handler!")

    def open(self, filename):
        if self.dataset_handler is None:
            raise ValueError("No Dataset Handler!")
        if not self.dataset_handler.loaded:
            raise FileNotFoundError("Dataset not loaded")

        return None