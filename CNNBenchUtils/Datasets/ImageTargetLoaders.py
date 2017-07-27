import os
import numpy as np

from CNNBenchUtils.Datasets.BaseDatasetFileLoader import *


class ImageTargetLoader:
    def __init__(self, dataset=None, fileloader=None):
        self.properties = {}
        self.dataset = dataset
        self.fileloader = None
        if fileloader is not None and issubclass(type(fileloader), BaseDatasetFileLoader):
            self.fileloader = fileloader

    def open(self, filename):
        img = self.fileloader.open(filename)
        if img is None:
            return None, None

        target = os.path.split(os.path.split(filename)[0])[-1] # get directory name which is at the same time also the class name
        try:
            idx = self.dataset.classes.index(target)
        except ValueError as e:
            print("Loading File \"", filename, "\"!(", e, ")")
            return None, None

        target = np.zeros(len(self.dataset.classes), dtype='float32')
        target[idx] = 1.0

        # convert to floats between 0 and 1
        img = np.asarray(img / 255., dtype='float32')

        img = img.reshape(-1, self.dataset.properties['image.dimensions'][2], self.dataset.properties['image.dimensions'][1], self.dataset.properties['image.dimensions'][0])
        target = target.reshape(-1, len(self.dataset.classes))

        return img, target


class BatchImageTargetLoader:
    def __init__(self, file_loader=None):
        self.file_loader = file_loader
        self.dataset = file_loader.dataset
        self.batch_size = self.dataset.get_prop('batch.size')
        self.image_target_loader = ImageTargetLoader(self.dataset, self.file_loader)
        self.files = self.dataset.sample_files

    def __get_chunk(self):
        for path_batch in range(0, len(self.dataset.sample_files), self.batch_size):
            yield self.files[path_batch:path_batch + self.batch_size]

    def validate(self):
        self.files = self.dataset.validation_files

    def train(self):
        self.files = self.dataset.sample_files

    def next_batch(self):
        for fp_chunk in self.__get_chunk():
            image_batch_buffer = np.zeros((self.dataset.properties['batch.size'],
                                           self.dataset.properties['image.dimensions'][2],
                                           self.dataset.properties['image.dimensions'][1],
                                           self.dataset.properties['image.dimensions'][0]), dtype='float32')

            target_batch_buffer = np.zeros((self.dataset.properties['batch.size'],
                                            len(self.dataset.classes)), dtype='float32')

            batch_num = 0
            for file in fp_chunk:
                image_buffer, target_buffer = self.image_target_loader.open(file)
                if image_buffer is None:
                    continue

                try:
                    image_batch_buffer[batch_num] = image_buffer
                    target_batch_buffer[batch_num] = target_buffer
                    batch_num += 1
                except Exception as e:
                    print("Culprit \"", file, "\":", e)

            image_batch_buffer = image_batch_buffer[:batch_num]
            target_batch_buffer = target_batch_buffer[:batch_num]

            yield image_batch_buffer, target_batch_buffer
