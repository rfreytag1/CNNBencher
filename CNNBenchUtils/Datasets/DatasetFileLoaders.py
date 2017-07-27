import time
import cv2
import numpy as np

from CNNBenchUtils.Datasets.BaseDatasetFileLoader import BaseFileLoader


class CachedImageFileLoader(BaseFileLoader):
    file_cache = {}

    def __init__(self, dataset_handler=None, max_cachesize=32):
        super(CachedImageFileLoader, self).__init__(dataset_handler)
        self.max_cachesize = max_cachesize

    def __find_oldest_cache_entry(self):
        oldest_cache_entry = {
            'file': None,
            'last_access': time.time()
        }

        for cache_entry in CachedImageFileLoader.file_cache:
            if cache_entry['last_access'] < oldest_cache_entry['last_access']:
                oldest_cache_entry = cache_entry

        return oldest_cache_entry

    def open(self, filename):
        super(CachedImageFileLoader, self).open(filename)
        if filename in CachedImageFileLoader.file_cache:
            cache_entry = CachedImageFileLoader.file_cache[filename]
            cache_entry['last_access'] = time.time()
            return cache_entry

        # file = self.dataset.open(filename)
        src = self.dataset_handler.open(filename)
        if src is None:
            return None

        srcb = np.frombuffer(src.read(), dtype='byte')

        file = cv2.imdecode(srcb, cv2.IMREAD_ANYCOLOR)

        if file is None:
            return None

        if len(CachedImageFileLoader.file_cache) >= self.max_cachesize:
            CachedImageFileLoader.file_cache.pop(self.__find_oldest_cache_entry())

        CachedImageFileLoader.file_cache[filename] = {
            "file": file,
            "last_access": time.time()
        }

        return file