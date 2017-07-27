import os

from CNNBenchUtils.Datasets.BaseDatasetHandler import BaseDatasetHandler


class DatasetDirectoryHandler(BaseDatasetHandler):
    def __init__(self, datasetpath=None):
        super(DatasetDirectoryHandler, self).__init__(datasetpath)
        self.loaded = True

    def list(self, subpath=None):
        if subpath is None:
            subpath = '.'

        dir_list = os.listdir(os.path.join(self.datasetpath, subpath))
        dir_list = [os.path.join(subpath, dirent) for dirent in dir_list]

        return dir_list

    def open(self, filename):
        fobj = open(os.path.join(self.datasetpath, filename), 'rb')
        return fobj