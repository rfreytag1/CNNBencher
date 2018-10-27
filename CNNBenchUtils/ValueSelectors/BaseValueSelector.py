from CNNBenchUtils.DynamicValues.BaseValue import BaseValue


class BaseValueSelector:
    '''
    Base class for selectors, which hold all dynamic values and select some(or all) each stage for a change
    '''
    def __init__(self):
        # list of all registered dynamic values
        self.dynamic_values = []
        # list of currently locked dvalues
        self.locked_values = []
        # list of currently unlocked dvalues
        self.unlocked_values = []

    @staticmethod
    def parse(selector):
        '''
        IMPLEMENT THIS!

        This should parse the selector section of a description file.

        This specifically is just a base function to be called by the actual implementation.

        :param selector: section in which the selector is described as a dict
        :return: hopefully a BaseValueSelector derived class instance
        '''
        if not isinstance(selector, dict):
            raise TypeError('Parameter "selector" must be a dict!')

        return None

    @staticmethod
    def shortname():
        '''
        return short name of itself.
        This is used to identify the selector in a description file and call a parse function accordingly.
        :return: short name of selector
        '''
        return 'none'

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
        '''
        gives the option to select dynamic values for specific stages, disregarding the actual algorithm that usually selects dvals.
        :param dval: dynamic value to pre-select
        :param stage: in which stage to select it
        :return:
        '''
        pass

    def select_dvals(self, stage):
        '''
        Select the dynamic values which should update their values.
        :param stage: for which stage to select the dvals
        '''
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

    def reset_all(self):
        for dval in self.dynamic_values:
            dval.reset()
