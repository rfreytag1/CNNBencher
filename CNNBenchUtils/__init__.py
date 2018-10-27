
if __name__ == '__main__':
    import CNNBenchUtils.DynamicValues.ValueTypes as vt
    import CNNBenchUtils.ValueSelectors.ValueSelectors as vs

    stages = 10
    rr_value_selector = vs.RoundRobinValueSelector()
    rand_value_selector = vs.RandomValueSelector()

    cosine_dval = vt.ValueCosine(stages=stages, gapless=True)
    linear_dval = vt.ValueLinear(stages=stages, gapless=True)
    linear_dval_gap = vt.ValueLinear(stages=stages, gapless=False)
    multi_dval = vt.ValueMultiRR(["Value0", "Value1", "Value2", "Value3"], stages=stages, gapless=True)

    rr_value_selector.register_dval(cosine_dval)
    rr_value_selector.register_dval(linear_dval)
    rr_value_selector.register_dval(linear_dval_gap)
    rr_value_selector.register_dval(multi_dval)

    rand_value_selector.register_dval(cosine_dval)
    rand_value_selector.register_dval(linear_dval)
    rand_value_selector.register_dval(linear_dval_gap)
    rand_value_selector.register_dval(multi_dval)

    print("==== RR ====")
    print("{:12s}\t{:12s}\t{:12s}\t{:12s}".format("Cosine", "Linear", "Linear(gap)", "Multi"))
    for i in range(0, stages):
        rr_value_selector.select_dvals(i)
        print("{:<9.6f}{:3s}\t{:<9.6f}{:3s}\t{:<9.6f}{:3s}\t{:<9s}{:3s}".format(cosine_dval.value(i),
                                                                                    "(*)" if not cosine_dval.is_locked() else "",
                                                                                    linear_dval.value(i),
                                                                                    "(*)" if not linear_dval.is_locked() else "",
                                                                                    linear_dval_gap.value(i),
                                                                                    "(*)" if not linear_dval_gap.is_locked() else "",
                                                                                    multi_dval.value(i),
                                                                                    "(*)" if not multi_dval.is_locked() else ""))
    rr_value_selector.reset_all()

    print("==== Rand ====")
    print("{:12s}\t{:12s}\t{:12s}\t{:12s}".format("Cosine", "Linear", "Linear(gap)", "Multi"))
    for i in range(0, stages):
        rand_value_selector.select_dvals(i)
        print("{:<9.6f}{:3s}\t{:<9.6f}{:3s}\t{:<9.6f}{:3s}\t{:<9s}{:3s}".format(cosine_dval.value(i),
                                                                                    "(*)" if not cosine_dval.is_locked() else "",
                                                                                    linear_dval.value(i),
                                                                                    "(*)" if not linear_dval.is_locked() else "",
                                                                                    linear_dval_gap.value(i),
                                                                                    "(*)" if not linear_dval_gap.is_locked() else "",
                                                                                    multi_dval.value(i),
                                                                                    "(*)" if not multi_dval.is_locked() else ""))
