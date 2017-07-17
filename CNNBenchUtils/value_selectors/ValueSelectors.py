from CNNBenchUtils.value_selectors.BaseValueSelector import BaseValueSelector


class RandomValueSelector(BaseValueSelector):
    def __init__(self):
        super(RandomValueSelector, self).__init__()

    def select_dvals(self, stage):
        self.lock_all()
        idx = random.randint(0, len(self.dynamic_values)-1)
        dval = self.dynamic_values[idx]
        self.unlock_dval(dval)


class RoundRobinValueSelector(BaseValueSelector):
    def __init__(self):
        super(RoundRobinValueSelector, self).__init__()

    def select_dvals(self, stage):
        self.lock_all()
        idx = stage % len(self.dynamic_values)
        self.unlock_dval(self.dynamic_values[idx])


class OrderedValueSelector(BaseValueSelector):
    def __init__(self):
        super(OrderedValueSelector, self).__init__()

    def select_dvals(self, stage):
        self.lock_all()
        idx = stage
        if idx >= len(self.dynamic_values):
            idx = len(self.dynamic_values) - 1
        self.unlock_dval(self.dynamic_values[idx])


class AllValueSelector(BaseValueSelector):
    def __init__(self):
        super(AllValueSelector, self).__init__()
        self.select_called = False

    def select_dvals(self, stage):
        if not self.select_called:
            self.select_called = True
            self.unlock_all()
