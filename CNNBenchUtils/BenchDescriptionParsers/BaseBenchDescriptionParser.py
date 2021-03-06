import numbers
import zipfile
import os

from CNNBenchUtils.Datasets.Dataset import Dataset
from CNNBenchUtils.Datasets.DatasetHandlers import *

from CNNBenchUtils.DynamicValues.ValueTypes import *
from CNNBenchUtils.ValueSelectors.ValueSelectors import *


class BenchDescription(dict):
    '''
    dummy class for benchmark description data structure.
    '''
    def __init__(self):
        super(BenchDescription, self).__init__()
        self['name'] = ''
        self['stages'] = 0
        self['runs'] = 0
        self['datasets'] = {}
        self['cnns'] = {}


class BaseBenchDescriptionParser:
    def __init__(self, gapless_dvalues=False):
        self.param_parsers = {}
        self.selector_parsers = {}

        self.gapless_dvalues = gapless_dvalues

        self.stages = 0

        self.bench_desc = BenchDescription()

        # register default param parsers

        self.register_dvalue_type(ValueStatic.shortname(), ValueStatic.parse)
        self.register_dvalue_type(ValueLinear.shortname(), ValueLinear.parse)
        self.register_dvalue_type(ValueCosine.shortname(), ValueCosine.parse)
        self.register_dvalue_type(ValueStepped.shortname(), ValueStepped.parse)
        self.register_dvalue_type(ValueSteppedInt.shortname(), ValueSteppedInt.parse)
        self.register_dvalue_type(ValueMulti.shortname(), ValueMulti.parse)
        self.register_dvalue_type(ValueMultiRR.shortname(), ValueMultiRR.parse)

        # register default selector parsers

        self.register_dvalue_selector(RandomValueSelector.shortname(), RandomValueSelector.parse)
        self.register_dvalue_selector(OrderedValueSelector.shortname(), OrderedValueSelector.parse)
        self.register_dvalue_selector(RoundRobinValueSelector.shortname(), RoundRobinValueSelector.parse)
        self.register_dvalue_selector(AllValueSelector.shortname(), AllValueSelector.parse)
        self.register_dvalue_selector(ManualValueSelector.shortname(), ManualValueSelector.parse)

    def register_dvalue_type(self, dvtype, parser_func):
        if not isinstance(dvtype, str):
            raise TypeError('Parameter "type" must be a string!')
        if not callable(parser_func):
            raise TypeError('Parameter "parser_func" must be callable!')
        self.param_parsers[dvtype] = parser_func

    def register_dvalue_selector(self, stype, parser_func):
        if not isinstance(stype, str):
            raise TypeError('Parameter "stype" must be a string!')
        if not callable(parser_func):
            raise TypeError('Parameter "parser_func" must be callable!')
        self.selector_parsers[stype] = parser_func

    def parse_dataset(self, dataset):
        '''
        Tries to parse the dataset section and read the dataset file structure to get a file list and the class labels
        :param dataset:
        :return:
        '''
        dataset_file = dataset['filename']
        tmp_dataset = Dataset()

        tmp_dataset.set_prop('classes.max', dataset.get('classes', 20))
        tmp_dataset.set_prop('image.dimensions', dataset.get('image.dimensions', [128, 128, 1]))
        tmp_dataset.set_prop('samplesize', dataset.get('class.samples', 10))
        tmp_dataset.set_prop('validation.frac', dataset.get('validation.frac', 0.1))

        dataset_handler = None
        if dataset_file.endswith('.zip') and zipfile.is_zipfile(dataset_file):
            pass
        elif os.path.isdir(dataset_file):
            dataset_handler = DatasetDirectoryHandler(dataset_file)
        else:
            dataset_handler = DatasetDirectoryHandler(dataset_file)

        if dataset_handler is None:
            return None

        tmp_dataset.dataset_handler = dataset_handler
        tmp_dataset.init_classes()
        tmp_dataset.init_files()

        return tmp_dataset

    def parse_param(self, param, cnn_name):
        '''
        Tries to parse a parameter section
        :param param: parameter section to parse
        :param cnn_name: name of the CNN to associate the parse parameter with(important for selector)
        :return: BaseValue derived instance or None on fail
        '''
        # get specified parameter type value string
        param_type = str(param['type']).lower()
        # try to find an appropriate parser for the type specified
        parse_func = self.param_parsers.get(param_type)

        if parse_func is None or not callable(parse_func):
            return None

        # call specific DynamicValue parser
        dval = parse_func(param, self.stages, self.gapless_dvalues)

        if dval is None:
            return None

        # get preselection value
        pselected = param.get('selected')

        self.bench_desc['cnns'][cnn_name]['selector'].register_dval(dval)
        if isinstance(pselected, list):
            # if the pre-selection value is a list, it's a list of stages in which it will be selected for change
            for stagenum in pselected:
                self.bench_desc['cnns'][cnn_name]['selector'].preselect(dval, stagenum)
        elif isinstance(pselected, numbers.Number):
            # if it's a number, it's just a single stage in which it's selected for change
            self.bench_desc['cnns'][cnn_name]['selector'].preselect(dval, pselected)
        elif isinstance(pselected, str):
            # if it's a str, we handle it as a boolean and select it to be changed throughout all stages
            # TODO: couldn't it actually be a boolean?
            if pselected.lower() == 'true':
                self.bench_desc['cnns'][cnn_name]['selector'].preselect(dval)

        return dval

    def parse_selector(self, selector):
        '''
        Parse and initialize selector
        :param selector: selector section
        :return: BaseValueSelector derived instance or None on fail
        '''
        selector_type = str(selector['selector'])
        parse_func = self.selector_parsers.get(selector_type)

        if parse_func is None or not callable(parse_func):
            return None

        return parse_func(selector)

    def parse(self, filename):
        pass
