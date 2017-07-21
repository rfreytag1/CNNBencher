from CNNBenchUtils.DynamicValues.BaseValue import BaseValue


class BaseValueSelector:
    def __init__(self):
        self.dynamic_values = [] # list of all registered dynamic values
        self.locked_values = [] # list of currently locked dvalues
        self.unlocked_values = [] # list of currently unlocked dvalues

    def register_dval(self, dval):
        if issubclass(type(dval), BaseValue):
            self.dynamic_values.append(dval)
            if dval.is_locked():
                self.locked_values.append(dval)
            else:
                self.unlocked_values.append(dval)
        else:
            raise TypeError("dval must be BaseValue derived!")

    def preselect(self, dval, stage=-1):
        pass

    def select_dvals(self, stage):
        raise NotImplementedError()

    def lock_dval(self, dval):
        dval.lock()
        self.unlocked_values.remove(dval)
        self.locked_values.append(dval)

    def unlock_dval(self, dval):
        dval.unlock()
        self.locked_values.remove(dval)
        self.unlocked_values.append(dval)

    def lock_all(self):
        for dval in self.unlocked_values:
            print("lock")
            dval.lock()

        self.locked_values.clear()
        self.locked_values.extend(self.dynamic_values)
        self.unlocked_values.clear()

    def unlock_all(self):
        for dval in self.locked_values:
            dval.unlock()

        self.unlocked_values.clear()
        self.unlocked_values.extend(self.dynamic_values)
        self.locked_values.clear()

