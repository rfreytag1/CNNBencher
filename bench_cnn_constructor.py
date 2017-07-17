from CNNBenchUtils.dynamic_values.ValueTypes import *
from CNNBenchUtils.value_selectors.ValueSelectors import *

class CNNBenchNetConstructor:
    def load_config(self, file):
        print(file)

if __name__ == '__main__':
    rv = AllValueSelector()
    testval0 = ValueMulti([14, 23, 4, 4], 10)
    testval1 = ValueLinear(0.0, 1.0, 10)
    testval2 = ValueStepped(1.0, 10.0, 1.0, 10)

    rv.register_dval(testval0)
    rv.register_dval(testval1)
    rv.register_dval(testval2)
    for i in range(0, 10):
        rv.select_dvals(i)
        print("stage:", i)
        for cdval in rv.dynamic_values:
            print("\tValue:", round(cdval.value(i), 2), "(locked:", cdval.is_locked(), ")")
