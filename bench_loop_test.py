#!/usr/bin/env python3

from datetime import datetime
import time
import logging
import json

from lasagne import layers as l
from lasagne import nonlinearities
from lasagne import init

import CNNBenchUtils.DynamicValues.ValueTypes as cnndval
from CNNBenchUtils.ValueSelectors.ValueSelectors import *

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as cnnbp

class CNNBenchNetBuilder:
    def __init__(self):
        pass

    @staticmethod
    def getdval(dval, stage):
        if dval is not None and issubclass(type(dval), BaseValue):
            return dval.value(stage)

        return None

    @staticmethod
    def getdval_int(dval, stage, default_value=0):
        val = CNNBenchNetBuilder.getdval(dval, stage)

        ival = default_value
        if val is not None:
            try:
                ival = int(val)
            except:
                pass # do something?

        return ival

    @staticmethod
    def getdval_float(dval, stage, default_value=0.0):
        val = CNNBenchNetBuilder.getdval(dval, stage)

        ival = default_value
        if val is not None:
            try:
                ival = float(val)
            except:
                pass # do something? maybe not.

        return ival


    @staticmethod
    def build(cnn_desc, stage=0):
        net = None
        for layer in cnn_desc['layers']:
            ltype = str(layer['type']).lower()
            if ltype == 'input':
                batch_size = CNNBenchNetBuilder.getdval(layer['params'].get('batch_size'), stage)
                channels = CNNBenchNetBuilder.getdval(layer['params'].get('channels'), stage)
                width = CNNBenchNetBuilder.getdval(layer['params'].get('width'), stage)
                height = CNNBenchNetBuilder.getdval(layer['params'].get('height'), stage)
                net = l.InputLayer((batch_size, channels, width, height))
            elif ltype == 'conv':
                conv_type = str(CNNBenchNetBuilder.getdval(layer['params'].get('type'), stage)).lower()
                kernels = CNNBenchNetBuilder.getdval_int(layer['params'].get('kernels'), stage)
                kernel_size = CNNBenchNetBuilder.getdval_int(layer['params'].get('kernel_size'), stage)
                pad = str(CNNBenchNetBuilder.getdval(layer['params'].get('pad'), stage)).lower()
                stride = CNNBenchNetBuilder.getdval_int(layer['params'].get('stride'), stage)

                nonlinearity_type = str(CNNBenchNetBuilder.getdval(layer['params'].get('nonlinearity'), stage)).lower()
                nonlinearity = None
                if nonlinearity_type == 'elu':
                    nonlinearity = nonlinearities.elu

                weights_type = str(CNNBenchNetBuilder.getdval(layer['params'].get('weigths.type'), stage)).lower()
                weights = None
                if weights_type == 'henormal':
                    weights_gain = CNNBenchNetBuilder.getdval_float(layer['params'].get('stride'), stage, 1.0)
                    weights = init.HeNormal(gain=weights_gain)

                if conv_type == '2d':
                    net = l.batch_norm(l.Conv2DLayer(net, num_filters=kernels, filter_size=kernel_size, pad=pad, stride=stride, W=weights, nonlinearity=nonlinearity))
                elif conv_type == '3d':
                    net = l.batch_norm(l.Conv3DLayer(net, num_filters=kernels, filter_size=kernel_size, pad=pad, stride=stride, W=weights, nonlinearity=nonlinearity))
            elif ltype == 'pooling':
                pool_type = str(CNNBenchNetBuilder.getdval(layer['params'].get('type'), stage)).lower()
                pool_size = CNNBenchNetBuilder.getdval_int(layer['params'].get('poolsize'), stage, 1)

                if pool_type.startswith('max'):
                    if pool_type.endswith('2d'):
                        net = l.MaxPool2DLayer(net, pool_size=pool_size)

        return net


def param_stepped_parse(param, stages, gapless):
    pstart = int(param['start'])
    pend = int(param['end'])
    pstep = int(param['step'])
    dval = cnndval.ValueSteppedInt(pstart, pend, pstep, stages, gapless)
    return dval


stages = 10
runs = 5

logging.basicConfig(filename="./test2.log", filemode="a+", level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
default_log = logging.getLogger("CNNBencher")

bdp = cnnbp.BenchDescriptionJSONParser(True)
bench_desc = bdp.parse("./sample_cnn_bench1.json")

print(bench_desc)

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