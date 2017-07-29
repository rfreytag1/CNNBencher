#!/usr/bin/env python3

import logging
import os
import time
import pynvml
import psutil

import threading

import CNNBenchUtils.Benchmark as CNNBench
# Loading images with CPU background threads during GPU forward passes saves a lot of time
# Credit: J. Schlüter (https://github.com/Lasagne/Lasagne/issues/12)

pynvml.nvmlInit()

gpus = pynvml.nvmlDeviceGetCount()
print("Found", gpus, "GPUs")

nvhandle = pynvml.nvmlDeviceGetHandleByIndex(0)

hw_log_file = open("./hw_log_test.log", 'w+')
hw_log_file.write("time;GPU Temperature(°C);GPU Util(%);GPU VRAM Util(%);GPU VRAM Usage(MB);CPU Util(%);CPU Freq(MHz);RAM Usage(MB);Disk Write(B/s);Disk Read(B/s)\n")


def get_util():
    print("Thread started")
    diskio_write_old = psutil.disk_io_counters().write_bytes
    diskio_read_old = psutil.disk_io_counters().read_bytes
    while True:
        gputemp = pynvml.nvmlDeviceGetTemperature(nvhandle, 0)
        gpuutil = pynvml.nvmlDeviceGetUtilizationRates(nvhandle)
        gpumem = pynvml.nvmlDeviceGetMemoryInfo(nvhandle)
        hw_log_file.write(time.strftime("%Y-%m-%dT%H:%M:%S") + ";" + str(gputemp) + ";" + str(gpuutil.gpu) + ";" + str(gpuutil.memory) + ";" + str(round(gpumem.used / (1024 * 1024), 2)) + ";")
        hw_log_file.write(str(psutil.cpu_percent()) + ";" + str(psutil.cpu_freq().current) + ";" + str(round(psutil.virtual_memory().used / (1024 * 1024), 2)) + ";")
        diskio_write = psutil.disk_io_counters().write_bytes
        diskio_write_bps = (diskio_write - diskio_write_old) / 5
        diskio_write_old = diskio_write

        diskio_read = psutil.disk_io_counters().read_bytes
        diskio_read_bps = (diskio_read - diskio_read_old) / 5
        diskio_read_old = diskio_read

        hw_log_file.write(str(diskio_write_bps) + ";" + str(diskio_read_bps))

        hw_log_file.write("\n")
        hw_log_file.flush()
        time.sleep(5)
        
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

hw_log_thread = threading.Thread(target=get_util)
hw_log_thread.start()

cnnbench = CNNBench.CNNLasagneBenchmark("sample_cnn_bench1.json")
cnnbench.run()

hw_log_thread.join()

hw_log_file.close()
