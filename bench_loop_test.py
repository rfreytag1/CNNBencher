#!/usr/bin/env python3

from datetime import datetime
import time
import logging
import json

from lasagne import layers as l

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as cnnbp

import CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder as cnnb

import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTrainingFunctionBuilder as cnntrf
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTestFunctionBuilder as cnntef

BATCH_SIZE=128
MULTI_LABEL=True
TRAIN=[]
'''
def threadedBatchGenerator(generator, num_cached=10):
    import queue
    batch_queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            batch_queue.put(item)
            batch_queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = batch_queue.get()
    while item is not sentinel:
        yield item
        batch_queue.task_done()
        item = batch_queue.get()


def getDatasetChunk(split):
    # get batch-sized chunks of image paths
    for i in xrange(0, len(split), BATCH_SIZE):
        yield split[i:i + BATCH_SIZE]


def getNextImageBatch(split=TRAIN, doAugmentation=True, batchAugmentation=MULTI_LABEL):
    # fill batch
    for chunk in getDatasetChunk(split):

        # allocate numpy arrays for image data and targets
        x_b = np.zeros((BATCH_SIZE, IM_DIM, IM_SIZE[1], IM_SIZE[0]), dtype='float32')
        y_b = np.zeros((BATCH_SIZE, NUM_CLASSES), dtype='float32')

        ib = 0
        for path in chunk:

            try:

                # load image data and class label from path
                x, y = loadImageAndTarget(path, doAugmentation)

                # pack into batch array
                x_b[ib] = x
                y_b[ib] = y
                ib += 1

            except:
                continue

        # trim to actual size
        x_b = x_b[:ib]
        y_b = y_b[:ib]

        # same class augmentation?
        if doAugmentation and SAME_CLASS_AUGMENTATION and x_b.shape[0] > 2:
            x_b, y_b = getSameClassAugmentation(x_b, y_b)

        # batch augmentation?
        if batchAugmentation and x_b.shape[0] >= BATCH_SIZE // 2:
            x_b, y_b = getAugmentedBatches(x_b, y_b)

            # instead of return, we use yield
            yield x_b, y_b
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

print(len(bench_desc['cnns']['TestCNN01']['layers']))
print(len(bench_desc['selector'].dynamic_values))

netbuilder = cnnb.LasagneCNNBuilder()

net = netbuilder.build(bench_desc['cnns']['TestCNN01'])

train_func_builder = cnntrf.LasagneTrainingFunctionBuilder()
test_func_builder = cnntef.LasagneTestFunctionBuilder()

tensors = {}
train_func = train_func_builder.build(net, bench_desc['cnns']['TestCNN01']['training']['function'], tensors, 0)
test_func = test_func_builder.build(net, bench_desc['cnns']['TestCNN01']['training']['function'], tensors, 0)
test_func_builder.build(stage=1)

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

for stage in range(0, 4):
    # make dvalue selection
    bench_desc['selector'].select_dvals(stage)
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