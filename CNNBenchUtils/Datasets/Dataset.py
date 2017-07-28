import os
import random


class Dataset:
    def __init__(self, dataseth=None):
        self.dataset_handler = dataseth

        self.properties = {
            'classes.max': 20,
            'samplesize': 30,
            'dir.samples': "samples",
            'dir.noise': "noise",
            'validation.frac': 0.1,
            'image.dimensions': [512, 512, 1],
            'batch.size': 64
        }

        self.initialized = False

        self.classes = []
        self.sample_files = []
        self.validation_files = []
        self.noise_files = []

        if dataseth is not None:
            self.init_classes()
            self.init_files()

    def init_classes(self):
        dataset_root = self.dataset_handler.list(self.properties['dir.samples'])
        for ent in dataset_root:
            if os.path.isdir(os.path.join(self.dataset_handler.datasetpath, ent)) and len(self.classes) < self.get_prop('classes.max'):
                self.classes.append(os.path.split(ent)[-1])
        random.shuffle(self.classes)

    def init_files(self):
        class_samples = {}
        for c in self.classes:
            class_samples[c] = 0
            class_dir = self.dataset_handler.list(os.path.join(self.properties['dir.samples'], c))
            for f in class_dir:
                if class_samples[c] < self.properties['samplesize']:
                    self.sample_files.append(f)

        random.shuffle(self.sample_files)

        if self.properties['validation.frac'] > 0.0:
            split_size = int(len(self.sample_files) * self.properties['validation.frac'])
            self.validation_files = self.sample_files[0:split_size]
            self.sample_files = self.sample_files[split_size+1:]

        self.initialized = True

    def set_prop(self, key, value=None):
        self.properties[key] = value

    def get_prop(self, key):
        return self.properties.get(key)