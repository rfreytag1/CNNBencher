from CNNBenchUtils.CNNBuilders.BaseBuilder import BaseBuilder


class BaseCNNBuilder(BaseBuilder):
    def __init__(self, cnn_desc=None):
        super(BaseCNNBuilder, self).__init__()
        if isinstance(cnn_desc, dict):
            self.cnn_desc = cnn_desc
        elif cnn_desc is not None:
            raise TypeError('Argument "cnn_desc" must be a dict!')

    def build(self, cnn_desc=None, stage=0):
        if cnn_desc is not None and not isinstance(cnn_desc, dict):
            raise TypeError('Argument "cnn_desc" must be a dict!')

        if cnn_desc is not None:
            self.cnn_desc = cnn_desc

        if self.cnn_desc is None:
            raise ValueError('No neural net description specified!')

        return None

