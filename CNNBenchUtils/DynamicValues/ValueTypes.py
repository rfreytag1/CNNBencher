import math
from CNNBenchUtils.DynamicValues.BaseValue import BaseValue


class ValueStatic(BaseValue):
    def __init__(self, value=None, stages=0, gapless=False):
        super(ValueStatic, self).__init__(value, stages, gapless)

    @staticmethod
    def parse(param, stages, gapless):
        BaseValue.parse(param, stages, gapless)
        if str(param['type']).lower() == 'static':
            return ValueStatic(param['value'], stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'static'

    def value(self, stage=0):
        return self.val


class ValueStepped(BaseValue):
    def __init__(self, start=0.0, end=1.0, step=0.1, stages=1, gapless=False):
        super(ValueStepped, self).__init__(start, stages, gapless)
        self.start = start
        self.end = end
        self.step = step

        self.actual_end = self.start + (self.step * (self.stages-1))

    @staticmethod
    def parse(param, stages, gapless):
        if str(param['type']).lower() == 'stepped':
            start = float(param['start'])
            end = float(param['end'])
            step = float(param['step'])
            return ValueStepped(start, end, step, stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'stepped'

    def value(self, stage=0):
        super(ValueStepped, self).value(stage)
        if self.is_locked():
            return self.val

        stage = self.actual_stage(stage)

        self.val = self.start + (self.step * stage)

        if self.val > self.end:
            self.val = self.end
            return self.end

        return self.val


class ValueSteppedInt(BaseValue):
    def __init__(self, start=0, end=1, step=1, stages=1, gapless=False):
        super(ValueSteppedInt, self).__init__(int(start), stages, gapless)
        self.start = int(start)
        self.end = int(end)
        self.step = int(step)

        self.actual_end = self.start + (self.step * (self.stages-1))

    @staticmethod
    def parse(param, stages, gapless):
        if str(param['type']).lower() == 'stepped_int':
            start = int(param['start'])
            end = int(param['end'])
            step = int(param['step'])
            return ValueSteppedInt(start, end, step, stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'stepped_int'

    def value(self, stage=0):
        super(ValueSteppedInt, self).value(stage)
        if self.is_locked():
            return self.val

        stage = self.actual_stage(stage)

        self.val = self.start + (self.step * stage)

        if self.val > self.end:
            self.val = self.end
            return self.end

        return self.val


class ValueLinear(BaseValue):
    def __init__(self, start=0.0, end=1.0, stages=1, gapless=False):
        super(ValueLinear, self).__init__(start, stages, gapless)
        self.start = start
        self.end = end

    @staticmethod
    def parse(param, stages, gapless):
        if str(param['type']).lower() == 'linear':
            start = float(param['start'])
            end = float(param['end'])
            return ValueLinear(start, end, stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'linear'

    def value(self, stage=0):
        super(ValueLinear, self).value(stage)
        if self.is_locked():
            return self.val

        stage = self.actual_stage(stage)

        if stage >= self.stages:
            return self.end

        if self.stages > 1:
            frac = (stage / (self.stages - 1))
        else:
            frac = 1.0

        self.val = self.start * (1.0 - frac) + self.end * frac  # standard lerp
        return self.val


class ValueCosine(BaseValue):
    def __init__(self, start=0.0, end=1.0, stages=1, gapless=False):
        super(ValueCosine, self).__init__(start, stages, gapless)
        self.start = start
        self.end = end

    @staticmethod
    def parse(param, stages, gapless):
        if str(param['type']).lower() == 'cosine':
            start = float(param['start'])
            end = float(param['end'])
            return ValueCosine(start, end, stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'cosine'

    def value(self, stage=0):
        super(ValueCosine, self).value(stage)
        if self.is_locked():
            return self.val

        stage = self.actual_stage()

        if stage >= self.stages:
            return self.end

        frac = ((stage + 1) / self.stages) - (1.0 / self.stages)
        cfac = (1 - math.cos(frac * math.pi)) / 2.0
        self.val = self.start * (1.0 - cfac) + self.end * cfac  # standard cosine interp
        return self.val


class ValueMulti(BaseValue):
    def __init__(self, values=None, stages=0, gapless=False):
        super(ValueMulti, self).__init__(None, stages, gapless)
        self.values = values
        if isinstance(self.values, list):
            self.val = self.values[0]
        else:
            self.val = None

    @staticmethod
    def parse(param, stages, gapless):
        if str(param['type']).lower() == 'multi':
            values = list(param['values'])
            return ValueMulti(values, stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'multi'

    def value(self, stage=0):
        super(ValueMulti, self).value(stage)
        if self.is_locked():
            return self.val

        stage = self.actual_stage(stage)

        if stage >= self.stages:
            self.val = self.values[len(self.values) - 1]
            return self.val

        #  probably should be done differently
        idx = math.floor((len(self.values) - 1) * (((stage+1) / self.stages)-(1.0/self.stages)))
        self.val = self.values[int(idx)]
        return self.val


class ValueMultiRR(BaseValue):
    def __init__(self, values=None, stages=0, gapless=False):
        super(ValueMultiRR, self).__init__(None, stages, gapless)
        self.values = values
        if isinstance(self.values, list):
            self.val = self.values[0]
        else:
            self.val = None

    @staticmethod
    def parse(param, stages, gapless):
        if str(param['type']).lower() == 'multi-rr':
            values = list(param['values'])
            return ValueMultiRR(values, stages, gapless)
        return None

    @staticmethod
    def shortname():
        return 'multi-rr'

    def value(self, stage=0):
        super(ValueMultiRR, self).value(stage)
        if self.is_locked():
            return self.val

        stage = self.actual_stage(stage)

        if stage >= self.stages:
            self.val = self.values[len(self.values) - 1]
            return self.val

        idx = stage % len(self.values)
        self.val = self.values[idx]
        return self.val
