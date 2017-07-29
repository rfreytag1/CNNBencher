import logging
import time
import os

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as BenchDescriptionJSONParser
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder as LasagneCNNBuilder
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTrainingFunctionBuilder as LasagneTrainingFunctionBuilder
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTestFunctionBuilder as LasagneTestFunctionBuilder
import CNNBenchUtils.Datasets.DatasetFileLoaders as DatasetFileLoaders
import CNNBenchUtils.Datasets.ImageTargetLoaders as ImageTargetLoaders
import CNNBenchUtils.Datasets.BatchGenerators as BatchGenerators
import CNNBenchUtils.DynamicValues.ValueTypes as ValueTypes


class CNNLasagneBenchmark:
    def __init__(self, description_file=None, base_dir=None, logger=None):
        self.base_dir = os.path.join('.', time.strftime("%Y-%m-%dT%H.%M.%S"))
        if base_dir is not None and isinstance(base_dir, str):
            self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        if logger is not None and isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            logger_filehandler = logging.FileHandler(os.path.join(self.base_dir, "benchmark.log"))
            logger_streamhandler = logging.StreamHandler()
            logger_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s (%(name)s: %(module)s > %(funcName)s @ %(lineno)d)', '%Y-%m-%dT%H:%M:%S%z')
            logger_filehandler.setFormatter(logger_formatter)
            logger_filehandler.setLevel(logging.DEBUG)
            logger_streamhandler.setLevel(logging.INFO)
            self.logger.addHandler(logger_filehandler)
            self.logger.addHandler(logger_streamhandler)
            self.logger.info("Test")

        self.description_file = description_file
        self.benchmark_description = None
        if self.description_file is not None:
            self.open(self.description_file)

        self.file_loader_class = DatasetFileLoaders.CachedImageDatasetFileLoader
        self.batch_imagetarget_loader_class = ImageTargetLoaders.BatchImageTargetLoader
        self.batch_generator_class = BatchGenerators.ThreadedBatchGenerator
        self.netbuilder_class = LasagneCNNBuilder.LasagneCNNBuilder
        self.trainfunc_builder_class = LasagneTrainingFunctionBuilder.LasagneTrainingFunctionBuilder
        self.testfunc_builder_class = LasagneTestFunctionBuilder.LasagneTestFunctionBuilder

        self.net_params_csv = None

    def open(self, description_file):
        self.description_file = description_file
        self.logger.info("trying to parse Benchmark Description...")
        bdp = BenchDescriptionJSONParser.BenchDescriptionJSONParser(True)
        self.benchmark_description = bdp.parse(description_file)
        if self.benchmark_description is not None:
            self.logger.info("Parsing Benchmark Description successful!")
            self.logger.debug("Benchmark Description has %d Datasets and %d Neural Nets.",
                              len(self.benchmark_description['datasets']), len(self.benchmark_description['datasets']))
        else:
            self.logger.error("Could not parse Benchmark Description!")

    def __create_net_param_table(self, current_net):
        if self.net_params_csv is not None:
            self.net_params_csv.close()
            self.net_params_csv = None

        self.net_params_csv = open(os.path.join(self.base_dir, "net_params.csv"), 'w+')
        self.net_params_csv.write('stage;')
        # get headings
        layer_counter = {}
        for layer in current_net['layers']:
            layer_type = layer['type']
            if layer_type not in layer_counter:
                layer_counter[layer_type] = 0
            for param in layer['params']:
                self.net_params_csv.write(layer_type + str(layer_counter[layer_type]) + '.' + param + ';')
            layer_counter[layer_type] += 1

        self.net_params_csv.write("\n")
        self.net_params_csv.flush()

    def __dump_param_values(self, current_net, stage):
        # get param values
        self.net_params_csv.write(str(stage) + ';')
        for layer in current_net['layers']:
            for param_name, param in layer['params'].items():
                self.net_params_csv.write(str(param) + ';')

        self.net_params_csv.write("\n")
        self.net_params_csv.flush()

    def run(self):
        for dataset_name, dataset in self.benchmark_description['datasets'].items():
            self.logger.debug("Starting Benchmark for Dataset %s", dataset_name)
            cache_image_loader = self.file_loader_class(dataset)
            if cache_image_loader is None:
                self.logger.error("Could not instantiate File Loader. Aborting...")
                break

            for cnn, cnnc in self.benchmark_description['cnns'].items():
                self.logger.info("Benchmarking Neural Network \"%s\" with %d layers and %d Dynamic Values with Dataset %s.",
                                 cnn, len(cnnc['layers']), len(cnnc['selector'].dynamic_values), dataset_name)
                netbuilder = self.netbuilder_class(cnnc)
                tensors = {}
                train_func_builder = self.trainfunc_builder_class(None, cnnc['training']['function'])
                test_func_builder = self.testfunc_builder_class(None, cnnc['training']['function'])

                self.__create_net_param_table(cnnc)

                for stage in range(0, self.benchmark_description['stages']):
                    self.logger.info("Starting Stage %d of %d", stage + 1, self.benchmark_description['stages'])
                    cnnc['selector'].select_dvals(stage)

                    self.__dump_param_values(cnnc, stage)

                    epochs = cnnc['training']['params']['epochs'].value(stage)

                    dataset.set_prop('batch.size', cnnc['layers'][0]['params']['batch_size'].value(stage))
                    batch_it_loader = self.batch_imagetarget_loader_class(cache_image_loader)
                    batch_generator = self.batch_generator_class(batch_it_loader)

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
                            learning_rate = ValueTypes.ValueLinear(lr_start, lr_end, epochs, True)
                        else:
                            lr_start = float(cnnc['training']['function']['params']['learning_rate.start'].value(stage))
                            learning_rate = ValueTypes.ValueStatic(lr_start, epochs, True)

                        learning_rate.unlock()

                        for epoch in range(0, epochs):
                            self.logger.info("Epoch %d of %d", epoch + 1, epochs)
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
