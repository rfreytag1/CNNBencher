from CNNBenchUtils.CNNBuilders.BaseBuilder import BaseBuilder


class BaseLayerBuilder(BaseBuilder):
    '''
    Just a class to specify a common interface for layer building classes/functions
    '''
    @staticmethod
    def build(net, layer, stage=0):
        pass
