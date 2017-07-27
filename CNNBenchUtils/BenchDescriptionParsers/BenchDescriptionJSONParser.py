import json

from CNNBenchUtils.DynamicValues.ValueTypes import ValueStatic
from CNNBenchUtils.BenchDescriptionParsers.BaseBenchDescriptionParser import BaseBenchDescriptionParser


class BenchDescriptionJSONParser(BaseBenchDescriptionParser):
    def parse(self, filename):
        fp = open(filename, 'r')

        raw = json.load(fp, encoding='utf-8')

        self.bench_desc['name'] = str(raw['benchmark_name'])
        self.stages = int(raw['stages'])
        self.bench_desc['stages'] = self.stages
        self.bench_desc['runs'] = int(raw['runs'])
        self.bench_desc['cnns'] = {}

        # self.bench_desc['selector'] = self.parse_selector(raw['param_change'])

        # if self.bench_desc['selector'] is None:
        #     return None

        for cnn in raw['cnn_configurations']:
            cnn_name = str(cnn['cnn_name'])
            self.bench_desc['cnns'][cnn_name] = {}
            self.bench_desc['cnns'][cnn_name]['selector'] = self.parse_selector(raw['param_change'])
            self.bench_desc['cnns'][cnn_name]['training'] = {}
            self.bench_desc['cnns'][cnn_name]['training']['params'] = {}
            self.bench_desc['cnns'][cnn_name]['training']['function'] = {}
            self.bench_desc['cnns'][cnn_name]['training']['function']['params'] = {}

            for tparam in cnn['training']['params']:
                pkey = tparam['key']
                dval = self.parse_param(tparam, cnn_name)
                if dval is not None:
                    self.bench_desc['cnns'][cnn_name]['training']['params'][pkey.lower()] = dval

            if self.bench_desc['cnns'][cnn_name]['training']['params'].get('epochs') is None:
                self.bench_desc['cnns'][cnn_name]['training']['params']['epochs'] = ValueStatic(1, self.stages, self.gapless_dvalues)

            for fparam in cnn['training']['function']['params']:
                pkey = fparam['key']
                dval = self.parse_param(fparam, cnn_name)
                if dval is not None:
                    self.bench_desc['cnns'][cnn_name]['training']['function']['params'][pkey] = dval

            if self.bench_desc['cnns'][cnn_name]['training']['function']['params'].get('learning_rate.interp') is None:
                self.bench_desc['cnns'][cnn_name]['training']['function']['params']['learning_rate.interp'] = ValueStatic('linear', self.stages, self.gapless_dvalues)

            if self.bench_desc['cnns'][cnn_name]['training']['function']['params'].get('learning_rate.start') is None:
                self.bench_desc['cnns'][cnn_name]['training']['function']['params']['learning_rate.start'] = ValueStatic(0.01, self.stages, self.gapless_dvalues)

            if self.bench_desc['cnns'][cnn_name]['training']['function']['params'].get('learning_rate.end') is None:
                self.bench_desc['cnns'][cnn_name]['training']['function']['params']['learning_rate.end'] = ValueStatic(0.0001, self.stages, self.gapless_dvalues)

            self.bench_desc['cnns'][cnn_name]['layers'] = []
            # layer_number = 0
            for layer_number, layer in enumerate(cnn['layers']):
                self.bench_desc['cnns'][cnn_name]['layers'].append({})
                self.bench_desc['cnns'][cnn_name]['layers'][layer_number]['type'] = layer['type']
                self.bench_desc['cnns'][cnn_name]['layers'][layer_number]['params'] = {}
                for lparam in layer['params']:
                    pkey = str(lparam['key'])
                    dval = self.parse_param(lparam, cnn_name)
                    if dval is not None:
                        self.bench_desc['cnns'][cnn_name]['layers'][layer_number]['params'][pkey] = dval
                    else:
                        return None

                layer_number += 1

        return self.bench_desc
