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


class HWLogger(threading.Thread):
    def __init__(self, logfile=None, header=True):
        super(HWLogger, self).__init__()
        self.logfile_name = logfile

        pynvml.nvmlInit()
        self.gpus = pynvml.nvmlDeviceGetCount()
        self.nvhandles = []
        for gpuid in range(0, self.gpus):
            self.nvhandles.append(pynvml.nvmlDeviceGetHandleByIndex(gpuid))

        self.hw_log_file = open(logfile, 'w+', encoding='utf-8')

        if self.hw_log_file is not None and header:
            self.__write_header()

    def __write_header(self):
        self.hw_log_file.write("time;")
        for gpuid in range(0, self.gpus):
            self.hw_log_file.write("GPU{0} Temperature(°C);GPU{0} Util(%);GPU{0} VRAM Util(%);GPU{0} VRAM Usage(MB);GPU{0} Core Clock;GPU{0} Shader Clock;GPU{0} Memory Clock;".format(gpuid))
        self.hw_log_file.write("CPU Util(%);CPU Freq(MHz);RAM Usage(MB);Disk Write(B/s);Disk Read(B/s)\n")

    def __get_gpu_stats(self):
        for gpuid in range(0, self.gpus):
            gputemp = pynvml.nvmlDeviceGetTemperature(self.nvhandles[gpuid], 0)
            gpuutil = pynvml.nvmlDeviceGetUtilizationRates(self.nvhandles[gpuid])
            gpumem = pynvml.nvmlDeviceGetMemoryInfo(self.nvhandles[gpuid])
            gpuclock_core = pynvml.nvmlDeviceGetClockInfo(self.nvhandles[gpuid], 0)
            gpuclock_sm = pynvml.nvmlDeviceGetClockInfo(self.nvhandles[gpuid], 1)
            gpuclock_mem = pynvml.nvmlDeviceGetClockInfo(self.nvhandles[gpuid], 2)
            self.hw_log_file.write(str(gputemp) + ";" + str(gpuutil.gpu) + ";" + str(gpuutil.memory) + ";" + str(round(gpumem.used / (1024 * 1024), 2)) + ";" + str(gpuclock_core) + ";" + str(gpuclock_sm) + ";"  + str(gpuclock_mem) + ";")

    def __get_disk_stats(self):
        self.__diskio_write_old = psutil.disk_io_counters().write_bytes
        self.__diskio_read_old = psutil.disk_io_counters().read_bytes

        diskio_write = psutil.disk_io_counters().write_bytes
        diskio_write_bps = (diskio_write - self.__diskio_write_old) / 5
        self.__diskio_write_old = diskio_write

        diskio_read = psutil.disk_io_counters().read_bytes
        diskio_read_bps = (diskio_read - self.__diskio_read_old) / 5
        self.__diskio_read_old = diskio_read

        self.hw_log_file.write(str(diskio_write_bps) + ";" + str(diskio_read_bps))

    def __get_cpu_stats(self):
        self.hw_log_file.write(str(psutil.cpu_percent()) + ";" + str(psutil.cpu_freq().current) + ";" + str(round(psutil.virtual_memory().used / (1024 * 1024), 2)) + ";")

    def __get_util(self):
        while True:
            self.hw_log_file.write(time.strftime("%Y-%m-%dT%H:%M:%S") + ";")

            self.__get_gpu_stats()
            self.__get_cpu_stats()
            self.__get_disk_stats()

            self.hw_log_file.write("\n")
            self.hw_log_file.flush()
            time.sleep(5)

    def run(self):
        self.__get_util()
        super(HWLogger, self).run()

        
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
hw_logger = HWLogger(os.path.join(cnnbench.base_dir, "hw_log.csv"))
hw_logger.start()
cnnbench.run()

hw_logger.join()

