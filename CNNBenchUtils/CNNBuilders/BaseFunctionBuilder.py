from CNNBenchUtils.CNNBuilders.BaseBuilder import BaseBuilder


class BaseFunctionBuilder(BaseBuilder):
    def __init__(self):
        self.func = None
        self.func_desc = {}
        self.net = None
        self.tensors = {}

    def build(self, net, func_desc, tensors=None, stage=0):
        pass

    def rebuild(self, stage=0):
        pass
