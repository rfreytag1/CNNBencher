#!/usr/bin/env python3

import logging
import os
import queue
import time
import threading

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


class ImageTargetLoader:
    def __init__(self, dataset=None, fileloader=None):
        self.properties = {}
        self.dataset = dataset
        self.fileloader = None
        if fileloader is not None and issubclass(type(fileloader), cnndfh.BaseDatasetFileLoader):
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
        self.dataset = file_loader.dataset
        self.batch_size = self.dataset.get_prop('batch.size')
        self.image_target_loader = ImageTargetLoader(self.dataset, file_loader)
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
                image_batch_buffer[batch_num] = image_buffer
                target_batch_buffer[batch_num] = target_buffer
                batch_num += 1

            image_batch_buffer = image_batch_buffer[:batch_num]
            target_batch_buffer = target_batch_buffer[:batch_num]

            yield image_batch_buffer, target_batch_buffer


class ThreadedBatchGenerator:
    def __init__(self, batch_loader):
        self.batch_loader = batch_loader
        self.batch_queue = queue.Queue(maxsize=batch_loader.dataset.get_prop('batch.size'))
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
        
'''
for sample_file in dsd.sample_files:
    print(sample_file)
    img, target = itl.open(sample_file)
    print(target)
'''

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

dsh = cnndth.DatasetDirectoryHandler('./dataset/')
dsd = Dataset(dsh)
cifl = cnndfh.CachedImageDatasetFileLoader(dsd)
itl = ImageTargetLoader(dsd, cifl)

bitl = BatchImageTargetLoader(cifl)
tbg = ThreadedBatchGenerator(bitl)

print("parsed", len(bench_desc['cnns']), "CNNs")
print(bench_desc['datasets'])
for dataset_name, dataset in bench_desc['datasets'].items():
    cache_image_loader = cnndfh.CachedImageDatasetFileLoader(dataset)
    batch_it_loader = BatchImageTargetLoader(cache_image_loader)
    batch_generator = ThreadedBatchGenerator(batch_it_loader)

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
                    for image_batch, target_batch in batch_generator.batch():
                        loss = train_func(image_batch, target_batch, learning_rate.value(stage))
                        print(loss)

                    for image_batch, target_batch in batch_generator.batch(True):
                        prediction_batch, loss, acc = test_func(image_batch, target_batch)

                        print(prediction_batch)
                        print(loss)
                        print(acc)

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