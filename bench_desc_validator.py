import sys
import time

import CNNBenchUtils.BenchDescriptionParsers.BenchDescriptionJSONParser as BenchDescriptionJSONParser
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneCNNBuilder as LasagneCNNBuilder
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTrainingFunctionBuilder as LasagneTrainingFunctionBuilder
import CNNBenchUtils.CNNBuilders.Lasagne.LasagneTestFunctionBuilder as LasagneTestFunctionBuilder

if len(sys.argv) < 2:
    print("Usage:", sys.argv[0], "<bench_desc.json>")
    exit(1)

bdp = BenchDescriptionJSONParser.BenchDescriptionJSONParser(True)

print("Trying to parse JSON...", end='')
benchmark_description = bdp.parse(sys.argv[1])
if benchmark_description is not None:
    print("success!")
else:
    print("fail!")
    exit(1)

print("Starting Build Test")

start_time = time.perf_counter()

errors = False
for cnn, cnnc in benchmark_description['cnns'].items():
    print("Testing Neural Net Description \"", cnn, "\"")
    netbuilder = LasagneCNNBuilder.LasagneCNNBuilder(cnnc)
    tensors = {}
    train_func_builder = LasagneTrainingFunctionBuilder.LasagneTrainingFunctionBuilder(None, cnnc['training']['function'])
    test_func_builder = LasagneTestFunctionBuilder.LasagneTestFunctionBuilder(None, cnnc['training']['function'])

    for stage in range(0, benchmark_description['stages']):
        stage_errors = False
        print("Testing Stage", stage, "...", end='')

        cnnc['selector'].select_dvals(stage)
        epochs = cnnc['training']['params']['epochs'].value(stage)

        net = netbuilder.build(stage=stage)
        if net is None:
            print("Building NN", cnn, "in Stage", stage, "failed!")
            stage_errors = True

        tensors.clear()  # very important or else the functions will build with the wrong tensors(e.g. from previous builds)
        train_func = train_func_builder.build(net, tensors, stage=stage)
        if train_func is None:
            print("Building Training Function for NN", cnn, "in Stage", stage, "failed!")
            stage_errors = True

        test_func = test_func_builder.build(net, tensors, stage=stage)
        if test_func is None:
            print("Building Test Function for NN", cnn, "in Stage", stage, "failed!")
            stage_errors = True

        if stage_errors:
            print("fail!")
            errors = True
        else:
            print("success!")

delta_time = time.perf_counter() - start_time

print(delta_time)

if not errors:
    print("JSON file seems valid!")
else:
    print("something's wrong :/")
