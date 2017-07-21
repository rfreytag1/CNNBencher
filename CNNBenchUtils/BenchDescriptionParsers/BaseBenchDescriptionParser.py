from CNNBenchUtils.DynamicValues.ValueTypes import *
from CNNBenchUtils.ValueSelectors.ValueSelectors import *


class BenchDescription(dict):
    pass


class BaseBenchDescriptionParser:
    def __init__(self, gapless_dvalues=False):
        self.param_parsers = {}
        self.selector_parsers = {}

        self.gapless_dvalues = gapless_dvalues

        self.stages = 0

        self.bench_desc = BenchDescription()

        # register default param parsers

        self.register_dvalue_type('static', ValueStatic.parse)
        self.register_dvalue_type('linear', ValueLinear.parse)
        self.register_dvalue_type('cosine', ValueCosine.parse)
        self.register_dvalue_type('stepped', ValueStepped.parse)
        self.register_dvalue_type('stepped_int', ValueSteppedInt.parse)
        self.register_dvalue_type('multi', ValueMulti.parse)
        self.register_dvalue_type('multi-rr', ValueMultiRR.parse)

        # register default selector parsers

        self.register_dvalue_selector('random', RandomValueSelector.parse)
        self.register_dvalue_selector('ordered', OrderedValueSelector.parse)
        self.register_dvalue_selector('roundrobin', RoundRobinValueSelector.parse)
        self.register_dvalue_selector('all', AllValueSelector.parse)
        self.register_dvalue_selector('manual', ManualValueSelector.parse)

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
        param_type = str(param['type'])
        parse_func = self.param_parsers.get(param_type)

        if parse_func is None or not callable(parse_func):
            return None

        return parse_func(param, self.stages, self.gapless_dvalues)

    def parse_selector(self, selector):
        selector_type = str(selector['selector'])
        parse_func = self.selector_parsers.get(selector_type)

        if parse_func is None or not callable(parse_func):
            return None

        return parse_func(selector)

    def parse(self, filename):
        pass