import queue
import threading

# Loading images with CPU background threads during GPU forward passes saves a lot of time
# Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)


class ThreadedBatchGenerator:
    def __init__(self, batch_loader):
        self.batch_loader = batch_loader
        self.batch_queue = queue.Queue(maxsize=batch_loader.file_loader.max_cachesize)
        self.batch_end = object()

        self.producer_thread = threading.Thread(target=self.__batch_producer(), daemon=True)

    def __batch_producer(self, validate=None):
        if validate is not None:
            if validate is True:
                self.batch_loader.validate()
            else:
                self.batch_loader.train()

        for item in self.batch_loader.next_batch():
            self.batch_queue.put(item)
        self.batch_queue.put(self.batch_end)

    def batch(self):
        self.producer_thread.start()
        # run as consumer (read items from queue, in current thread)
        item = self.batch_queue.get()
        while item is not self.batch_end:
            yield item
            self.batch_queue.task_done()
            item = self.batch_queue.get()
