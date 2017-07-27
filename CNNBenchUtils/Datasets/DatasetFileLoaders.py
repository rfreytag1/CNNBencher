import time
import cv2
import numpy as np

from CNNBenchUtils.Datasets.BaseDatasetFileLoader import BaseDatasetFileLoader


class CachedImageDatasetFileLoader(BaseDatasetFileLoader):
    file_cache = {}

    def __init__(self, dataset=None, max_cachesize=32):
        super(CachedImageDatasetFileLoader, self).__init__(dataset)
        self.max_cachesize = max_cachesize

    def __find_oldest_cache_entry(self):
        oldest_cache_entry_name = ''
        oldest_cache_entry = {
            'file': None,
            'last_access': time.time()
        }

        for cache_entry_name, cache_entry in CachedImageDatasetFileLoader.file_cache.items():
            if cache_entry['last_access'] < oldest_cache_entry['last_access']:
                oldest_cache_entry = cache_entry
                oldest_cache_entry_name = cache_entry_name

        return oldest_cache_entry_name

    def open(self, filename):
        super(CachedImageDatasetFileLoader, self).open(filename)
        if filename in CachedImageDatasetFileLoader.file_cache:
            cache_entry = CachedImageDatasetFileLoader.file_cache[filename]
            cache_entry['last_access'] = time.time()
            return cache_entry['file']

        # file = self.dataset.open(filename)
        src = self.dataset.dataset_handler.open(filename)
        if src is None:
            return None

        srcb = np.frombuffer(src.read(), dtype='byte')

        file = cv2.imdecode(srcb, cv2.IMREAD_ANYCOLOR)

        if file is None:
            return None

        if self.dataset.properties['image.dimensions'][2] < 3:
            # TODO: add channel remix down to two channels
            try:
                file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print("Image color conversion for \"", filename, "\" failed!(", e, ")")
                return None
        else:
            pass

        file = cv2.resize(file, (self.dataset.properties['image.dimensions'][0], self.dataset.properties['image.dimensions'][1]))

        if len(CachedImageDatasetFileLoader.file_cache) >= self.max_cachesize:
            CachedImageDatasetFileLoader.file_cache.pop(self.__find_oldest_cache_entry())

        CachedImageDatasetFileLoader.file_cache[filename] = {
            "file": file,
            "last_access": time.time()
        }

        return file