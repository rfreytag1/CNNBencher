from CNNBenchUtils.CNNBuilders.BaseBuilder import BaseBuilder


class BaseTrainingFunctionBuilder(BaseBuilder):
    def __init__(self):
        self.train_func = None
        self.func_desc = {}
        self.net = None

    def build(self, net, func_desc, stage=0):
        pass

    def rebuild(self, stage=0):
        pass
