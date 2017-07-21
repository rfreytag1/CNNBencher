from CNNBenchUtils.CNNBuilders.Lasagne.LasagneLayerBuilders import *
from CNNBenchUtils.CNNBuilders.BaseCNNBuilder import BaseCNNBuilder


class LasagneCNNBuilder(BaseCNNBuilder):
    layer_builders = {
        'input': LasagneInputLayerBuilder,
        'conv': LasagneConvLayerBuilder,
        'pooling': LasagnePoolLayerBuilder,
        'dense': LasagneDenseLayerBuilder,
        'dropout': LasagneDropoutLayerBuilder
    }

    def __init__(self):
        super(LasagneCNNBuilder, self).__init__()
        self.cnn_description = None

    @staticmethod
    def register_layer_builder(layer_type, builder_func):
        if not isinstance(layer_type, str):
            raise TypeError('Argument "layer_type" must be a string!')

        if not callable(builder_func):
            raise TypeError('Argument "builder_func" must be a callable!')

        LasagneCNNBuilder.layer_builders[layer_type] = builder_func

    def build_layer(self, net, layer, stage=0):
        layer_type = str(layer.get('type', 'none')).lower()

        if layer_type == 'none':
            return

        layer_builder = LasagneCNNBuilder.layer_builders.get(layer_type)

        if layer_builder is None or not callable(layer_builder):
            return None

        return layer_builder.build(net, layer, stage)

    def build(self, cnn_desc, stage=0):
        self.cnn_description = cnn_desc
        net = None
        for layer in self.cnn_description['layers']:
            net = self.build_layer(net, layer, stage)
        return net

    def rebuild(self, stage=0):
        if self.cnn_description is None:
            return None

        net = None
        for layer in self.cnn_description['layers']:
            net = self.build_layer(net, layer, stage)
        return net
