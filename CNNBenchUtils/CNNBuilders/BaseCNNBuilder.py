from CNNBenchUtils.CNNBuilders.BaseBuilder import BaseBuilder


class BaseCNNBuilder(BaseBuilder):
    def __init__(self):
        super(BaseCNNBuilder, self).__init__()

    def build(self, cnn_desc, stage=0):
        pass

    def rebuild(self, stage=0):
        pass
