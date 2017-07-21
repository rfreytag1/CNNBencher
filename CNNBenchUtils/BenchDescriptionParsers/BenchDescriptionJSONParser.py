import json
import numbers

from CNNBenchUtils.BenchDescriptionParsers.BaseBenchDescriptionParser import BaseBenchDescriptionParser


class BenchDescriptionJSONParser(BaseBenchDescriptionParser):
    def parse(self, filename):
        fp = open(filename, 'r')

        dval_gapless = True

        raw = json.load(fp, encoding='utf-8')

        # TODO: this looks disastrous. clean up. maybe?
        self.bench_desc['name'] = str(raw['benchmark_name'])
        self.stages = int(raw['stages'])
        self.bench_desc['stages'] = self.stages
        self.bench_desc['runs'] = int(raw['runs'])
        self.bench_desc['cnns'] = {}

        self.bench_desc['selector'] = self.parse_selector(raw['param_change'])

        if self.bench_desc['selector'] is None:
            return None

        for cnn in raw['cnn_configurations']:
            cnn_name = str(cnn['cnn_name'])
            self.bench_desc['cnns'][cnn_name] = {}
            self.bench_desc['cnns'][cnn_name]['layers'] = []
            layer_number = 0
            for layer in cnn['layers']:
                self.bench_desc['cnns'][cnn_name]['layers'].append({})
                self.bench_desc['cnns'][cnn_name]['layers'][layer_number]['type'] = layer['type']
                self.bench_desc['cnns'][cnn_name]['layers'][layer_number]['params'] = {}
                for lparams in layer['params']:
                    pkey = str(lparams['key'])
                    pselected = lparams.get('selected')
                    dval = self.parse_param(lparams)
                    if dval is not None:
                        self.bench_desc['selector'].register_dval(dval)
                        self.bench_desc['cnns'][cnn_name]['layers'][layer_number]['params'][pkey] = dval
                        if isinstance(pselected, list):
                            for stagenum in pselected:
                                self.bench_desc['selector'].preselect(dval, stagenum)
                        elif isinstance(pselected, numbers.Number):
                            self.bench_desc['selector'].preselect(dval, pselected)
                        elif isinstance(pselected, str):
                            if pselected.lower() == 'true':
                                self.bench_desc['selector'].preselect(dval)

                layer_number += 1

        return self.bench_desc