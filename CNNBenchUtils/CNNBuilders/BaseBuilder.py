from CNNBenchUtils.DynamicValues.BaseValue import BaseValue


class BaseBuilder:
    '''
    base class for builder classes with some helper functions
    '''
    # Helper functions
    @staticmethod
    def getdval(dval, stage):
        '''
        helper function to avoid checking the same stuff all the time regarding the dynamic values
        :param dval: dynamic value(any derived from DynamicValues.BaseValue)
        :param stage: benchmark stage to get value for
        :return: current value of a dynamic value for the requested stage or None if unavailable
        '''
        if dval is not None and issubclass(type(dval), BaseValue):
            return dval.value(stage)

        return None

    @staticmethod
    def getdval_int(dval, stage, default_value=0):
        '''
        helper and wrapper function for getdval which returns a specified default value instead of None
        :param dval: dynamic value
        :param stage: benchmark stage
        :param default_value: default value to return of dynamic value is or returns None
        :return: current integer value of dynamic value or default_value
        '''
        val = BaseBuilder.getdval(dval, stage)

        ival = default_value
        if val is not None:
            try:
                ival = int(val)
            except:
                pass  # do something?

        return ival

    @staticmethod
    def getdval_float(dval, stage, default_value=0.0):
        '''
        helper and wrapper function for getdval which returns a specified default value instead of None
        :param dval: dynamic value
        :param stage: benchmark stage
        :param default_value: default value to return of dynamic value is or returns None
        :return: current float value of dynamic value or default_value
        '''
        val = BaseBuilder.getdval(dval, stage)

        fval = default_value
        if val is not None:
            try:
                fval = float(val)
            except:
                pass  # do something? maybe not.

        return fval

    @staticmethod
    def getdval_str(dval, stage, default_value=''):
        '''
        helper and wrapper function for getdval which returns a specified default value instead of None
        :param dval: dynamic value
        :param stage: benchmark stage
        :param default_value: default value to return of dynamic value is or returns None
        :return: current string value of dynamic value or default_value
        '''
        val = BaseBuilder.getdval(dval, stage)

        sval = default_value
        if val is not None:
            try:
                sval = str(val)
            except:
                pass  # do something? maybe not.

        return sval

    @staticmethod
    def getdval_list(dval, stage, default_value=None):
        '''
        helper and wrapper function for getdval which returns a specified default value instead of None
        :param dval: dynamic value
        :param stage: benchmark stage
        :param default_value: default value to return of dynamic value is or returns None
        :return: current list value of dynamic value or default_value
        '''
        val = BaseBuilder.getdval(dval, stage)

        tval = []
        if default_value is not None and isinstance(default_value, list):
            tval = default_value

        if val is not None:
            try:
                tval = list(val)
            except:
                pass  # do something? maybe not.

        return tval

    @staticmethod
    def getdval_dict(dval, stage, default_value=None):
        '''
        helper and wrapper function for getdval which returns a specified default value instead of None
        :param dval: dynamic value
        :param stage: benchmark stage
        :param default_value: default value to return of dynamic value is or returns None
        :return: current dict value of dynamic value or default_value
        '''
        val = BaseBuilder.getdval(dval, stage)

        tval = {}
        if default_value is not None and isinstance(default_value, dict):
            tval = default_value

        if val is not None:
            try:
                tval = dict(val)
            except:
                pass  # do something? maybe not.

        return tval

    @staticmethod
    def getdval_tuple(dval, stage, default_value=None):
        '''
        helper and wrapper function for getdval which returns a specified default value instead of None
        :param dval: dynamic value
        :param stage: benchmark stage
        :param default_value: default value to return of dynamic value is or returns None
        :return: current tuple value of dynamic value or default_value
        '''
        val = BaseBuilder.getdval(dval, stage)

        tval = ()
        if default_value is not None and isinstance(default_value, tuple):
            tval = default_value

        if val is not None:
            try:
                tval = tuple(val)
            except:
                pass  # do something? maybe not.

        return tval
