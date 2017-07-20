#!/usr/bin/env python3

from datetime import datetime
import time
import logging
import json

import CNNBenchUtils.DynamicValues.ValueTypes as cnndval
from CNNBenchUtils.ValueSelectors.ValueSelectors import *


class CNNBenchDescription(dict):
    pass


class CNNBenchDescriptionParser:
    def __init__(self):
        self.bench_desc = CNNBenchDescription()

    def parse(self, file):
        fp = open(file, 'r')

        raw = json.load(fp, encoding='utf-8')

        self.bench_desc['name'] = raw['benchmark_name']
        self.bench_desc['stages'] = raw['stages']
        self.bench_desc['runs'] = raw['runs']



stages = 10
runs = 5

logging.basicConfig(filename="./test2.log", filemode="a+", level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
default_log = logging.getLogger("CNNBencher")

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
'''
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