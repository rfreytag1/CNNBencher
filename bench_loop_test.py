#!/usr/bin/env python3

from datetime import datetime
import time
import logging
import json

from lasagne import layers as l

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as cnnbp

import CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder as cnnb


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

for stage in range(0, 4):
    bench_desc['selector'].select_dvals(stage)
    net = netbuilder.rebuild(stage)
    print("===START PARAMETERS===")
    lnums = {
    }
    for layerp in bench_desc['cnns']['TestCNN01']['layers']:
        ltype = layerp['type']
        if ltype not in lnums:
            lnums[ltype] = 0
        for lparmname, lparmval in layerp['params'].items():
            print(layerp['type'] + str(lnums[ltype]) + "." + lparmname + ": " + str(lparmval))
        lnums[ltype] += 1

    print("===END PARAMETERS===")
    print("MODEL HAS", (sum(hasattr(layer, 'W') for layer in l.get_all_layers(net))), "WEIGHTED LAYERS")
    print("MODEL HAS", l.count_params(net), "PARAMS")

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