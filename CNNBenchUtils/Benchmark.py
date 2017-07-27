import logging
import time
import os

from CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser import BenchDescriptionJSONParser
from CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder import LasagneCNNBuilder
from CNNBenchUtils.CNNBuilders.Lasagne.LasagneTrainingFunctionBuilder import LasagneTrainingFunctionBuilder
from CNNBenchUtils.CNNBuilders.Lasagne.LasagneTestFunctionBuilder import LasagneTestFunctionBuilder
from CNNBenchUtils.Datasets.DatasetFileLoaders import *
from CNNBenchUtils.Datasets.ImageTargetLoaders import *
from CNNBenchUtils.Datasets.BatchGenerators import *
from CNNBenchUtils.DynamicValues.ValueTypes import *


class CNNLasagneBenchmark:
    def __init__(self, description_file=None, base_dir=None, logger=None, file_loader=None):
        self.base_dir = os.path.join('.', time.strftime("%Y-%m-%dT%H:%M:%S"))
        if base_dir is not None and isinstance(base_dir, str):
            self.base_dir = base_dir

        if logger is None:
            self.logger = logging.getLogger(__name__)
            logger_filehandler = logging.FileHandler(os.path.join(base_dir, "benchmark.log"))
            logger_streamhandler = logging.StreamHandler()
            logger_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%(name)s: %(module)s > %(funcName)s @ %(lineno)d)', '%Y-%m-%dT%H:%M:%S%z')
            logger_filehandler.setFormatter(logger_formatter)
            logger_filehandler.setLevel(logging.DEBUG)
            logger_streamhandler.setLevel(logging.INFO)
            self.logger.addHandler(logger_filehandler)
            self.logger.addHandler(logger_streamhandler)
        elif isinstance(logger, logging.Logger):
            self.logger = logger

        self.description_file = description_file
        self.benchmark_description = None
        if self.description_file is not None:
            self.open(self.description_file)

        self.file_loader = CachedImageDatasetFileLoader
        if issubclass(file_loader, BaseDatasetFileLoader):
            self.file_loader = file_loader
        elif file_loader is not None:
            self.logger.warning("File Loader is invalid. Using default.")

    def open(self, description_file):
        self.description_file = description_file
        self.logger.info("trying to parse Benchmark Description...")
        bdp = BenchDescriptionJSONParser(True)
        self.benchmark_description = bdp.parse(description_file)
        if self.benchmark_description is not None:
            self.logger.info("Parsing Benchmark Description successful!")
            self.logger.debug("Benchmark Description has %d Datasets and %d Neural Nets.", len(self.benchmark_description['datasets']), len(self.benchmark_description['datasets']))
        else:
            self.logger.error("Could not parse Benchmark Description!")

    def run(self):
        for dataset_name, dataset in self.benchmark_description['datasets'].items():
            self.logger.debug("Starting Benchmark for Dataset %s", dataset_name)
            cache_image_loader = self.file_loader(dataset)
            if cache_image_loader is None:
                self.logger.error("Could not instantiate File Loader. Aborting...")
                break

            for cnn, cnnc in self.benchmark_description['cnns'].items():
                self.logger.info("Benchmarking Neural Network \"%s\" with %d layers and %d Dynamic Values with Dataset %s.", cnn, len(cnnc['layers']), len(cnnc['selector'].dynamic_values), dataset_name)
                netbuilder = LasagneCNNBuilder(cnnc)
                tensors = {}
                train_func_builder = LasagneTrainingFunctionBuilder(None, cnnc['training']['function'])
                test_func_builder = LasagneTestFunctionBuilder(None, cnnc['training']['function'])

                for stage in range(0, self.benchmark_description['stages']):
                    self.logger.info("Starting Stage %d of %d", stage + 1, self.benchmark_description['stages'])
                    cnnc['selector'].select_dvals(stage)
                    epochs = cnnc['training']['params']['epochs'].value(stage)

                    dataset.set_prop('batch.size', cnnc['layers'][0]['params']['batch_size'].value(stage))
                    batch_it_loader = BatchImageTargetLoader(cache_image_loader)
                    batch_generator = ThreadedBatchGenerator(batch_it_loader)

                    self.logger.debug("Building Neural Network...")
                    net = netbuilder.build(stage=stage)
                    if net is not None:
                        self.logger.debug("Neural Network complete!")
                    else:
                        self.logger.error("Buidling Neural Network \"%s\" for Stage %d failed!", cnn, stage)
                        break

                    tensors.clear()  # very important or else the functions will build with the wrong tensors(e.g. from previous builds)
                    self.logger.debug("Building Training Function...")
                    train_func = train_func_builder.build(net, tensors, stage=stage)
                    if train_func is not None:
                        self.logger.debug("Training Function complete!")
                    else:
                        self.logger.error("Building Training Function for Neural Net \"%s\" in Stage %d failed!", cnn, stage)
                        break

                    self.logger.debug("Building Test Function...")
                    test_func = test_func_builder.build(net, tensors, stage=stage)
                    if test_func is not None:
                        self.logger.debug("Test Function complete!")
                    else:
                        self.logger.error("Building Test Function for Neural Net \"%s\" in Stage %d failed!", cnn, stage)
                        break

                    run_measurements = {
                        "train.time": [],
                        "train.loss": [],
                        "test.time": [],
                        "test.loss": [],
                        "test.accuracy": [],
                        "test.prediction_batch": []
                    }
                    for run in range(0, self.benchmark_description['runs']):
                        self.logger.debug("Starting Run %d of %d...", run + 1, self.benchmark_description['runs'])
                        lr_interp = cnnc['training']['function']['params']['learning_rate.interp'].value(stage)
                        if lr_interp != 'none':
                            lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                            lr_end = float(cnnc['training']['function']['params']['learning_rate.end'].value(stage))
                            learning_rate = ValueLinear(lr_start, lr_end, epochs, True)
                        else:
                            lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                            learning_rate = ValueStatic(lr_start, epochs, True)

                        learning_rate.unlock()

                        for epoch in range(0, epochs):
                            print("Epoch", epoch + 1, "of", epochs)
                            batch_it_loader.train()
                            for image_batch, target_batch in batch_generator.batch():
                                loss = train_func(image_batch, target_batch, learning_rate.value(stage))
                                print(loss)

                            batch_it_loader.validate()
                            for image_batch, target_batch in batch_generator.batch():
                                prediction_batch, loss, acc = test_func(image_batch, target_batch)

                                print(prediction_batch)
                                print(loss)
                                print(acc)
