import numbers

from CNNBenchUtils.DynamicValues.ValueTypes import *
from CNNBenchUtils.ValueSelectors.ValueSelectors import *


class BenchDescription(dict):
    def __init__(self):
        super(BenchDescription, self).__init__()
        self['name'] = ''
        self['stages'] = 0
        self['runs'] = 0
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

    def parse_param(self, param):
        param_type = str(param['type']).lower()
        parse_func = self.param_parsers.get(param_type)

        if parse_func is None or not callable(parse_func):
            return None

        dval = parse_func(param, self.stages, self.gapless_dvalues)

        if dval is None:
            return None

        pselected = param.get('selected')

        self.bench_desc['selector'].register_dval(dval)
        if isinstance(pselected, list):
            for stagenum in pselected:
                self.bench_desc['selector'].preselect(dval, stagenum)
        elif isinstance(pselected, numbers.Number):
            self.bench_desc['selector'].preselect(dval, pselected)
        elif isinstance(pselected, str):
            if pselected.lower() == 'true':
                self.bench_desc['selector'].preselect(dval)

        return dval

    def parse_selector(self, selector):
        selector_type = str(selector['selector'])
        parse_func = self.selector_parsers.get(selector_type)

        if parse_func is None or not callable(parse_func):
            return None

        return parse_func(selector)

    def parse(self, filename):
        pass
