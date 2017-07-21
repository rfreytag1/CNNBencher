from CNNBenchUtils.DynamicValues.BaseValue import BaseValue


class BaseBuilder:
    # Helper functions
    @staticmethod
    def getdval(dval, stage):
        if dval is not None and issubclass(type(dval), BaseValue):
            return dval.value(stage)

        return None

    @staticmethod
    def getdval_int(dval, stage, default_value=0):
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
        val = BaseBuilder.getdval(dval, stage)

        tval = default_value
        if val is not None:
            try:
                tval = list(val)
            except:
                pass  # do something? maybe not.

        return tval

    @staticmethod
    def getdval_dict(dval, stage, default_value=None):
        val = BaseBuilder.getdval(dval, stage)

        tval = default_value
        if val is not None:
            try:
                tval = dict(val)
            except:
                pass  # do something? maybe not.

        return tval

    @staticmethod
    def getdval_tuple(dval, stage, default_value=None):
        val = BaseBuilder.getdval(dval, stage)

        tval = default_value
        if val is not None:
            try:
                tval = tuple(val)
            except:
                pass  # do something? maybe not.

        return tval