from CNNBenchUtils.DynamicValues.ValueTypes import *
from CNNBenchUtils.ValueSelectors.ValueSelectors import *

class CNNBenchNetConstructor:
    def load_config(self, file):
        print(file)

if __name__ == '__main__':

    stages = 11

    rv = ManualValueSelector()
    #testval0 = ValueMulti([14, 23, 4, 4], stages, True)
    #testval1 = ValueLinear(0.0, 1.0, stages, True)
    testval2 = ValueSteppedInt(64, 512, 32, stages, True)

    #rv.register_dval(testval0)
    #rv.register_dval(testval1)
    rv.register_dval(testval2)

    #rv.preselect(testval1, 5)
    rv.preselect(testval2)

    for i in range(0, stages):
        rv.select_dvals(i)
        print("stage:", i)
        for cdval in rv.dynamic_values:
            print("\tValue:", round(cdval.value(i), 2), "(locked:", cdval.is_locked(), ", gap-less: ", cdval.gapless, ")")
