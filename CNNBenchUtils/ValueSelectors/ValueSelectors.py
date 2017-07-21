from CNNBenchUtils.ValueSelectors.BaseValueSelector import BaseValueSelector
from CNNBenchUtils.DynamicValues.BaseValue import BaseValue


class RandomValueSelector(BaseValueSelector):
    def __init__(self):
        super(RandomValueSelector, self).__init__()

    @staticmethod
    def parse(selector):
        BaseValueSelector.parse(selector)

        selector_type = str(selector['selector'])

        if selector_type != 'random':
            return None

        return RandomValueSelector()

    @staticmethod
    def shortname():
        return 'random'

    def select_dvals(self, stage):
        self.lock_all()
        idx = random.randint(0, len(self.dynamic_values)-1)
        dval = self.dynamic_values[idx]
        self.unlock_dval(dval)


class RoundRobinValueSelector(BaseValueSelector):
    def __init__(self):
        super(RoundRobinValueSelector, self).__init__()

    @staticmethod
    def parse(selector):
        BaseValueSelector.parse(selector)

        selector_type = str(selector['selector'])

        if selector_type != 'roundrobin':
            return None

        return RoundRobinValueSelector()

    @staticmethod
    def shortname():
        return 'roundrobin'

    def select_dvals(self, stage):
        self.lock_all()
        idx = stage % len(self.dynamic_values)
        self.unlock_dval(self.dynamic_values[idx])


class OrderedValueSelector(BaseValueSelector):
    def __init__(self):
        super(OrderedValueSelector, self).__init__()

    @staticmethod
    def parse(selector):
        BaseValueSelector.parse(selector)

        selector_type = str(selector['selector'])

        if selector_type != 'ordered':
            return None

        return OrderedValueSelector()

    @staticmethod
    def shortname():
        return 'ordered'

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

    @staticmethod
    def parse(selector):
        BaseValueSelector.parse(selector)

        selector_type = str(selector['selector'])

        if selector_type != 'all':
            return None

        return AllValueSelector()

    @staticmethod
    def shortname():
        return 'all'

    def select_dvals(self, stage):
        if not self.select_called:
            self.select_called = True
            self.unlock_all()


class ManualValueSelector(BaseValueSelector):
    def __init__(self):
        super(ManualValueSelector, self).__init__()
        self.preselected_values = {}
        self.permaselected_values = []

    @staticmethod
    def parse(selector):
        BaseValueSelector.parse(selector)

        selector_type = str(selector['selector'])

        if selector_type != 'manual':
            return None

        return ManualValueSelector()

    @staticmethod
    def shortname():
        return 'manual'

    def preselect(self, dval, stage=-1):
        if issubclass(type(dval), BaseValue):
            if dval not in self.dynamic_values:
                return

            if stage < 0:
                self.permaselected_values.append(dval)
                return

            if stage not in self.preselected_values:
                self.preselected_values[stage] = []

            self.preselected_values[stage].append(dval)
        else:
            raise TypeError("Argument dvals of invalid Type. Must be list of or single BaseValue-based instance.")

    def select_dvals(self, stage):
        self.lock_all()

        for dval in self.permaselected_values:
            self.unlock_dval(dval)

        if stage not in self.preselected_values:
            return

        for dval in self.preselected_values[stage]:
            self.unlock_dval(dval)

    '''
    def preselectn(self, dvals, stage=-1):
        if isinstance(dvals, list):
            for dval in dvals:
                if dval not in self.dynamic_values:
                    return

            if stage == -1:
                self.permaselected_values.extend(dvals)
                return

            if self.preselected_values[stage] is None:
                self.preselected_values[stage] = []

            self.preselected_values[stage].extend(dvals)
    '''