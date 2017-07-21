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


def input_layer_builder(net, layer, stage=0):
    batch_size = CNNBenchNetBuilder.getdval(layer['params'].get('batch_size'), stage)
    channels = CNNBenchNetBuilder.getdval(layer['params'].get('channels'), stage)
    width = CNNBenchNetBuilder.getdval(layer['params'].get('width'), stage)
    height = CNNBenchNetBuilder.getdval(layer['params'].get('height'), stage)
    return l.InputLayer((batch_size, channels, width, height))


def conv_layer_builder(net, layer, stage=0):
    conv_type = CNNBenchNetBuilder.getdval_str(layer['params'].get('type'), stage).lower()
    kernels = CNNBenchNetBuilder.getdval_int(layer['params'].get('kernels'), stage, 2)
    kernel_size = CNNBenchNetBuilder.getdval_int(layer['params'].get('kernel_size'), stage, 3)
    pad = CNNBenchNetBuilder.getdval_str(layer['params'].get('pad'), stage, 'same')
    stride = CNNBenchNetBuilder.getdval_int(layer['params'].get('stride'), stage, 2)
    nonlinearity_type = CNNBenchNetBuilder.getdval_str(layer['params'].get('nonlinearity'), stage).lower()
    weights_type = CNNBenchNetBuilder.getdval_str(layer['params'].get('weigths.type'), stage).lower()

    nonlinearity = None
    weights = None

    if nonlinearity_type == 'elu':
        nonlinearity = nonlinearities.elu
    else:
        nonlinearity = nonlinearities.elu

    if weights_type == 'henormal':
        weights_gain = CNNBenchNetBuilder.getdval_float(layer['params'].get('stride'), stage, 1.0)
        weights = init.HeNormal(gain=weights_gain)
    else:
        weights = init.Normal()

    if conv_type == '2d':
        return l.batch_norm(
            l.Conv2DLayer(net, num_filters=kernels, filter_size=kernel_size, pad=pad, stride=stride, W=weights,
                          nonlinearity=nonlinearity))
    elif conv_type == '3d':
        return l.batch_norm(
            l.Conv3DLayer(net, num_filters=kernels, filter_size=kernel_size, pad=pad, stride=stride, W=weights,
                          nonlinearity=nonlinearity))
    else:
        return l.batch_norm(
            l.Conv2DLayer(net, num_filters=kernels, filter_size=kernel_size, pad=pad, stride=stride, W=weights,
                          nonlinearity=nonlinearity))


def pooling_layer_builder(net, layer, stage=0):
    pool_type = CNNBenchNetBuilder.getdval_str(layer['params'].get('type'), stage).lower()
    pool_size = CNNBenchNetBuilder.getdval_int(layer['params'].get('pool_size'), stage, 1)

    if pool_type.startswith('max'):
        if pool_type.endswith('2d'):
            return l.MaxPool2DLayer(net, pool_size=pool_size)
        elif pool_type.endswith('1d'):
            return l.MaxPool1DLayer(net, pool_size=pool_size)
        elif pool_type.endswith('3d'):
            return l.MaxPool3DLayer(net, pool_size=pool_size)
        else:
            return l.MaxPool2DLayer(net, pool_size=pool_size)


def dense_layer_builder(net, layer, stage=0):
    units = CNNBenchNetBuilder.getdval_int(layer['params'].get('units'), stage, 1)
    nonlinearity_type = CNNBenchNetBuilder.getdval_str(layer['params'].get('nonlinearity'), stage).lower()
    weights_type = CNNBenchNetBuilder.getdval_str(layer['params'].get('weigths.type'), stage).lower()

    nonlinearity = None
    weights = None

    if nonlinearity_type == 'elu':
        nonlinearity = nonlinearities.elu
    else:
        nonlinearity = nonlinearities.elu

    if weights_type == 'henormal':
        weights_gain = CNNBenchNetBuilder.getdval_float(layer['params'].get('stride'), stage, 1.0)
        weights = init.HeNormal(gain=weights_gain)
    else:
        weights = init.Normal()

    return l.batch_norm(l.DenseLayer(net, units, W=weights, nonlinearity=nonlinearity))


class CNNBenchNetBuilder:
    layer_builders = {
        'input': input_layer_builder,
        'conv': conv_layer_builder,
        'pooling': pooling_layer_builder
    }

    def __init__(self):
        pass

    @staticmethod
    def register_layer_builder(layer_type, builder_func):
        if not isinstance(layer_type, str):
            raise TypeError('Argument "layer_type" must be a string!')

        if not callable(builder_func):
            raise TypeError('Argument "builder_func" must be a callable!')

        CNNBenchNetBuilder.layer_builders[layer_type] = builder_func

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
    def getdval_str(dval, stage, default_value=''):
        val = CNNBenchNetBuilder.getdval(dval, stage)

        ival = default_value
        if val is not None:
            try:
                ival = str(val)
            except:
                pass # do something? maybe not.

        return ival

    @staticmethod
    def build_layer(net, layer, stage=0):
        layer_type = str(layer['type']).lower()

        layer_builder = CNNBenchNetBuilder.layer_builders.get(layer_type)

        if layer_builder is None or not callable(layer_builder):
            return None

        return layer_builder(net, layer, stage)

    @staticmethod
    def build(cnn_desc, stage=0):
        net = None
        for layer in cnn_desc['layers']:
            ltype = str(layer['type']).lower()
            net = CNNBenchNetBuilder.build_layer(net, layer, stage)
        return net

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

net = CNNBenchNetBuilder.build(bench_desc['cnns']['TestCNN01'])

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