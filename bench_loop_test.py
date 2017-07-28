#!/usr/bin/env python3

import logging
import os

import CNNBenchUtils.Benchmark as CNNBench
# Loading images with CPU background threads during GPU forward passes saves a lot of time
# Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)






        
'''
for sample_file in dsd.sample_files:
    print(sample_file)
    img, target = itl.open(sample_file)
    print(target)
'''
default_log = logging.getLogger("CNNBencherDefault")
lfh = logging.FileHandler("./test_h.log")

# lfm = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%{name)s: %{module}s > %{funcName}s @ %{lineno}d', '%Y-%m-%dT%H:%M:%S%z')
lfm = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%(name)s: %(module)s > %(funcName)s @ %(lineno)d)', '%Y-%m-%dT%H:%M:%S%z')

lfh.setFormatter(lfm)
lfh.setLevel(logging.DEBUG)

default_log.addHandler(lfh)
default_log.info("test")

cnnbench = CNNBench.CNNLasagneBenchmark("sample_cnn_bench1.json")
cnnbench.run()
