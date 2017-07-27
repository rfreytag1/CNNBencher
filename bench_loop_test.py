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

print("parsed", len(bench_desc['cnns']), "CNNs")
for dataset_name, dataset in bench_desc['datasets'].items():
    cache_image_loader = cnndfh.CachedImageDatasetFileLoader(dataset)

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

            dataset.set_prop('batch.size', cnnc['layers'][0]['params']['batch_size'].value(stage))
            batch_it_loader = BatchImageTargetLoader(cache_image_loader)
            batch_generator = ThreadedBatchGenerator(batch_it_loader)

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