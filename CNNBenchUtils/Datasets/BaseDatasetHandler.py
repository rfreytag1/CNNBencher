class BaseDatasetHandler:
    def __init__(self, datasetpath=None):
        self.loaded = False
        self.datasetpath = datasetpath

    def is_loaded(self):
        return self.loaded

    def list(self, subpath=None):
        pass

    def load(self, datasetpath):
        self.datasetpath = datasetpath
        return True

    def open(self, filename):
        pass

    def close(self):
        pass
