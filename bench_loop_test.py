#!/usr/bin/env python3

import logging
import os
import queue
import time

import cv2
import numpy as np

import CNNBenchUtils.Datasets.DatasetHandlers as cnndth
import CNNBenchUtils.Datasets.DatasetFileLoaders as cnndfh

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as cnnbp
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder as cnnb
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTestFunctionBuilder as cnntef
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTrainingFunctionBuilder as cnntrf
import CNNBenchUtils.DynamicValues.ValueTypes as dvt
from CNNBenchUtils.Datasets.Dataset import Dataset

BATCH_SIZE=128
MULTI_LABEL=True
TRAIN=[]

# Loading images with CPU background threads during GPU forward passes saves a lot of time
# Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)
def threaded_batch_generator(generator, num_cached=10):
    import queue
    bqueue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            bqueue.put(item)
        bqueue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = bqueue.get()
    while item is not sentinel:
        yield item
        bqueue.task_done()
        item = bqueue.get()

class ThreadedBatchGenerator:
    queue_end = object()

    def __init__(self, dataset=None, batch_size=128):
        self.dataset = dataset
        self.file_queue = queue.Queue()
        self.batch_size = batch_size
        self.classes_maxnum = 20
        self.classes_count = 20
        self.classes = []
        self.image_dim = [512, 256, 3]
        self.cache_size = 20

    def getDatasetChunk(self, split):
        # get batch-sized chunks of image paths
        for i in range(0, len(split), self.batch_size):
            yield split[i:i + self.batch_size]

    def openImage(self, path):
        return np.ndarray([2,3], np.float32)

    def loadImageAndTarget(self, path):

        # here we open the image
        img = self.openImage(path)

        # image augmentation?

        # we want to use subfolders as class labels
        label = path.split("/")[-1]

        # we need to get the index of our label from CLASSES
        index = self.classes.index(label)

        # allocate array for target
        target = np.zeros(self.classes_maxnum, dtype='float32')

        # we set our target array = 1.0 at our label index, all other entries remain 0.0
        target[index] = 1.0

        # transpose image if dim=3
        try:
            img = np.transpose(img, (2, 0, 1))
        except:
            pass

        # we need a 4D-vector for our image and a 2D-vector for our targets
        img = img.reshape(-1, self.image_dim[2], self.image_dim[1], self.image_dim[0])
        target = target.reshape(-1, self.classes_maxnum)

        return img, target

    def getNextImageBatch(self, split):

        # fill batch
        for chunk in self.getDatasetChunk(split):

            # allocate numpy arrays for image data and targets
            image_batch_buffer = np.zeros((self.batch_size, self.image_dim[2], self.image_dim[1], self.image_dim[0]), dtype='float32')
            target_batch_buffer = np.zeros((self.batch_size, self.classes_maxnum), dtype='float32')

            loaded = 0
            for path in chunk:

                try:
                    # load image data and class label from path
                    x, y = self.loadImageAndTarget(path)

                    # pack into batch array
                    image_batch_buffer[loaded] = x
                    target_batch_buffer[loaded] = y
                    loaded += 1
                except:
                    continue

            # trim to actual size
            image_batch_buffer = image_batch_buffer[:loaded]
            target_batch_buffer = target_batch_buffer[:loaded]

            # instead of return, we use yield
            yield image_batch_buffer, target_batch_buffer

    def create_batch(self):
        pass
        '''
        for batched_paths in
        image_batch = np.zeros((self.batch_size, self.image_dim[2], self.image_dim[1], self.image_dim[0]), dtype='float32')
        targets_batch = np.zeros((self.batch_size, self.classes_count))
        '''

    def batch(self):
        import queue
        bqueue = queue.Queue(maxsize=self.cache_size)
        sentinel = object()  # guaranteed unique reference

        # define producer (putting items into queue)
        def producer():
            for item in self.getNextImageBatch():
                bqueue.put(item)
            bqueue.put(sentinel)

        # start producer (in a background thread)
        import threading
        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        # run as consumer (read items from queue, in current thread)
        item = bqueue.get()
        while item is not sentinel:
            yield item
            bqueue.task_done()
            item = bqueue.get()


class ImageTargetLoader:
    def __init__(self, dataset=None, fileloader=None):
        self.properties = {}
        self.dataset = dataset
        self.fileloader = None
        if fileloader is not None and issubclass(type(fileloader), cnndfh.BaseFileLoader):
            self.fileloader = fileloader
        self.properties['image.dimensions'] = [128, 64, 1]

    def open(self, filename):
        img = self.fileloader.open(filename)
        target = os.path.split(os.path.split(filename)[0])[-1] # get directory name which is at the same time also the class name
        try:
            idx = self.dataset.classes.index(target)
        except ValueError as e:
            print("Loading File \"", filename, "\"!(", e, ")")
            return None, None

        target = np.zeros(len(self.dataset.classes), dtype='float32')
        target[idx] = 1.0

        if img is None:
            return None

        if self.properties['image.dimensions'][2] < 3:
            # TODO: add channel remix down to two channels
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print("Eh", e)

                return None, None
        else:
            pass

            # resize to conv input size
        img = cv2.resize(img, (self.properties['image.dimensions'][0], self.properties['image.dimensions'][1]))

        # convert to floats between 0 and 1
        img = np.asarray(img / 255., dtype='float32')

        img = img.reshape(-1, self.properties['image.dimensions'][2], self.properties['image.dimensions'][1], self.properties['image.dimensions'][0])
        target = target.reshape(-1, len(self.dataset.classes))

        return img, target

        
dsh = cnndth.DatasetDirectoryHandler('./dataset/')
dsd = Dataset(dsh)
cifl = cnndfh.CachedImageFileLoader(dsh)
itl = ImageTargetLoader(dsd, cifl)
for sample_file in dsd.sample_files:
    print(sample_file)
    img, target = itl.open(sample_file)
    print(target)

exit()

stages = 10
runs = 5

default_log = logging.getLogger("CNNBencherDefault")
lfh = logging.FileHandler("./test_h.log")

# lfm = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%{name)s: %{module}s > %{funcName}s @ %{lineno}d', '%Y-%m-%dT%H:%M:%S%z')
lfm = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%(name)s: %(module)s > %(funcName)s @ %(lineno)d)', '%Y-%m-%dT%H:%M:%S%z')

lfh.setFormatter(lfm)
lfh.setLevel(logging.DEBUG)

default_log.addHandler(lfh)

default_log.warning("Tis' just a test!", {'lineno': 2})
bdp = cnnbp.BenchDescriptionJSONParser(True)
bench_desc = bdp.parse("./sample_cnn_bench1.json")


print("parsed", len(bench_desc['cnns']), "CNNs")
for cnn, cnnc in bench_desc['cnns'].items():
    print("Layers:", len(cnnc['layers']))
    print("Dynamic Values:", len(cnnc['selector'].dynamic_values))
    netbuilder = cnnb.LasagneCNNBuilder(cnnc)
    # net = netbuilder.build(cnnc)

    tensors = {}
    train_func_builder = cnntrf.LasagneTrainingFunctionBuilder(None, cnnc['training']['function'])
    test_func_builder = cnntef.LasagneTestFunctionBuilder(None, cnnc['training']['function'])

    for stage in range(0, bench_desc['stages']):
        print("Stage", stage+1, "of", bench_desc['stages'])
        cnnc['selector'].select_dvals(stage)

        epochs = cnnc['training']['params']['epochs'].value(stage)

        net = netbuilder.build(stage=stage)
        tensors.clear() # very important or else the functions will build with the wrong tensors(e.g. from previous builds)
        train_func = train_func_builder.build(net, tensors, stage=stage)
        test_func = test_func_builder.build(net, tensors, stage=stage)

        for run in range(0, bench_desc['runs']):
            print("Run", run+1, "of", bench_desc['runs'])
            lr_interp = cnnc['training']['function']['params']['learning_rate.interp'].value(stage)
            learning_rate = None
            if lr_interp != 'none':
                lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                lr_end = float(cnnc['training']['function']['params']['learning_rate.end'].value(stage))
                learning_rate = dvt.ValueLinear(lr_start, lr_end, epochs, True)
            else:
                lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                learning_rate = dvt.ValueStatic(lr_start, epochs, True)

            learning_rate.unlock()

            for epoch in range(0, epochs):
                print("Epoch", epoch+1, "of", epochs)
                # loss = train_func(image_batch, target_batch)
                # test_func...


#dataset = Dataset(bench_desc['datasets'][0]['filename'])
#batch_gen = ThreadedBatchGenerator(dataset)
'''
for image_batch, target_batch in batch_gen.batch():
    loss = train_func(image_batch, target_batch)
    # etc

lpd = open("layerparams.csv", 'w')
lpd.write("stage;")
lnums = {
}
for layerp in bench_desc['cnns']['TestCNN01']['layers']:
    ltype = layerp['type']
    if ltype not in lnums:
        lnums[ltype] = 0
    for lparmname, lparmval in layerp['params'].items():
        lpd.write(layerp['type'] + str(lnums[ltype]) + "." + lparmname + ";")
    lnums[ltype] += 1
lpd.write("\n")

for stage in range(0, 1):
    # make dvalue selection
    bench_desc['cnns']['TestCNN01']['selector'].select_dvals(stage)
    # build net
    net = netbuilder.build(stage=stage)
    train_func = train_func_builder.build(stage=stage)
    test_func = test_func_builder.build(stage=stage)

    # write current values to file
    lpd.write(str(stage) + ";")
    for layerp in bench_desc['cnns']['TestCNN01']['layers']:
        for lparmname, lparmval in layerp['params'].items():
            lpd.write(str(lparmval) + ";")
    lpd.write("\n")
    print("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS")
    print("MODEL HAS", l.count_params(net), "PARAMS")

    for run in range(0, 5):
        # do training and testing here
        pass
        
'''

'''
bdfp = open("./sample_cnn_bench1.json", "r")



bench_desc = json.load(bdfp, encoding='utf-8')

cnnvals = {}

valsel = RoundRobinValueSelector()

for net in bench_desc["cnn_configurations"]:
    cnnvals[net['cnn_name']] = {}
    lnum = 0
    for layer in net["layers"][1:]:
        cnnvals[net['cnn_name']][lnum] = {}
        for param in layer["params"]:
            ptype = param['type']
            dval = None
            if ptype == 'static':
                dval = cnndval.ValueStatic(param["value"])
            elif ptype == 'stepped':
                dval = cnndval.ValueStepped(param['start'], param['end'], param['step'])
            elif ptype == 'multi':
                dval = cnndval.ValueMulti(param['values'])

            if dval is not None:
                cnnvals[net['cnn_name']][lnum][param['key']] = dval
        lnum += 1

print(cnnvals)
            # create conv layer

for i in range(0, stages):
    print("Stage", i)
    # build model from json
    total_time = 0
    for j in range(0, runs):
        t0 = time.perf_counter()
        print("Run", j)
        # train and test net, calc losses etc. for quality measurement

        t1 = time.perf_counter()
        total_time += t1-t0

    default_log.info("Stage: " + str(i) + " Run: " + str(j) + " took " + str((total_time / runs) * 1000) + "ms")

    print("Time spent", (total_time/runs) * 1000, "ms")
'''