import math
from CNNBenchUtils.dynamic_values.BaseValue import BaseValue


class ValueStatic(BaseValue):
    def __init__(self, stages=0, value=0):
        self.val = value
        super(ValueStatic, self).__init__(stages)

    def value(self, stage):
        return self.val


class ValueStepped(BaseValue):
    def __init__(self, start=0.0, end=1.0, step=0.1, stages=1):
        super(ValueStepped, self).__init__(stages)
        self.start = start
        self.end = end
        self.step = step
        self.val = self.start

        self.actual_end = self.start + (self.step * (self.stages-1))

    def value(self, stage):
        if self.is_locked():
            return self.val

        if stage >= self.stages:
            return self.actual_end

        self.val = self.start + (self.step * stage)

        if self.val > self.end:
            self.val = self.end
            return self.end

        return self.val


class ValueLinear(BaseValue):
    def __init__(self, start=0.0, end=1.0, stages=1):
        super(ValueLinear, self).__init__(stages)
        self.start = start
        self.end = end
        self.val = self.start

    def value(self, stage):
        if self.is_locked():
            return self.val

        if stage >= self.stages:
            return self.end

        frac = ((stage + 1) / self.stages) - (1.0 / self.stages)
        self.val = self.start * (1.0 - frac) + self.end * frac  # standard lerp
        return self.val


class ValueMulti(BaseValue):
    def __init__(self, values=None, stages=0):
        super(ValueMulti, self).__init__(stages)
        self.values = values
        if isinstance(self.values, list):
            self.val = self.values[0]
        else:
            self.val = None

    def value(self, stage):
        if self.is_locked():
            return self.val

        if stage >= self.stages:
            self.val = self.values[len(self.values) - 1]
            return self.val

        #  probably should be done differently
        idx = math.floor((len(self.values) - 1) * (((stage+1) / self.stages)-(1.0/self.stages)))
        self.val = self.values[int(idx)]
        return self.val
