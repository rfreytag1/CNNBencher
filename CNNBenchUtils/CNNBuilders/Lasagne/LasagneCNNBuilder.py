from CNNBenchUtils.CNNBuilders.Lasagne.LasagneLayerBuilders import *
from CNNBenchUtils.CNNBuilders.BaseCNNBuilder import BaseCNNBuilder


class LasagneCNNBuilder(BaseCNNBuilder):
    layer_builders = {
        'input': LasagneInputLayerBuilder.build,
        'conv': LasagneConvLayerBuilder.build,
        'pooling': LasagnePoolLayerBuilder.build,
        'dense': LasagneDenseLayerBuilder.build,
        'dropout': LasagneDropoutLayerBuilder.build,
        'batch_norm': LasagneBatchNormLayerBuilder.build
    }

    def __init__(self, cnn_desc=None):
        super(LasagneCNNBuilder, self).__init__(cnn_desc)

    @staticmethod
    def register_layer_builder(layer_type, builder_func):
        if not isinstance(layer_type, str):
            raise TypeError('Argument "layer_type" must be a string!')

        if not callable(builder_func):
            raise TypeError('Argument "builder_func" must be a callable!')

        LasagneCNNBuilder.layer_builders[layer_type] = builder_func

    def __build_layer(self, net, layer, stage=0):
        layer_type = str(layer.get('type', 'none')).lower()

        if layer_type == 'none':
            return

        layer_builder = LasagneCNNBuilder.layer_builders.get(layer_type)

        if layer_builder is None or not callable(layer_builder):
            return None

        return layer_builder(net, layer, stage)

    def build(self, cnn_desc=None, stage=0):
        super(LasagneCNNBuilder, self).build(cnn_desc, stage)
        net = None
        for layer in self.cnn_desc['layers']:
            net = self.__build_layer(net, layer, stage)
        return net
