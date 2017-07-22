from lasagne import layers as l

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneLayerBuilder import BaseLasagneLayerBuilder


class LasagneInputLayerBuilder(BaseLasagneLayerBuilder):
    @staticmethod
    def build(net, layer, stage):
        batch_size = LasagneInputLayerBuilder.getdval(layer['params'].get('batch_size'), stage)
        width = LasagneInputLayerBuilder.getdval_int(layer['params'].get('width'), stage, 16)
        height = LasagneInputLayerBuilder.getdval_int(layer['params'].get('height'), stage, 16)
        depth = LasagneInputLayerBuilder.getdval_int(layer['params'].get('depth'), stage, 1)

        return l.InputLayer((batch_size, depth, width, height))


class LasagneConvLayerBuilder(BaseLasagneLayerBuilder):
    available_conv_layer_types = {
        'conv1d': l.Conv1DLayer,
        'conv2d': l.Conv2DLayer,
        'conv3d': l.Conv3DLayer
    }

    @staticmethod
    def build(net, layer, stage=0):
        conv_type = LasagneConvLayerBuilder.getdval_str(layer['params'].get('type'), stage, 'conv2d').lower()
        kernels = LasagneConvLayerBuilder.getdval_int(layer['params'].get('kernels'), stage, 2)
        kernel_size = LasagneConvLayerBuilder.getdval_int(layer['params'].get('kernel_size'), stage, 3)
        pad = LasagneConvLayerBuilder.getdval_str(layer['params'].get('pad'), stage, 'same').lower()
        stride = LasagneConvLayerBuilder.getdval_int(layer['params'].get('stride'), stage, 2)
        nonlinearity_type = LasagneConvLayerBuilder.getdval_str(layer['params'].get('nonlinearity'), stage).lower()

        nonlinearity = LasagneConvLayerBuilder.get_nonlinearity(nonlinearity_type)
        weights = LasagneConvLayerBuilder.get_weights_initw(layer['params'])

        layertype = LasagneConvLayerBuilder.available_conv_layer_types.get(conv_type)
        if layertype is None:
            layertype = LasagneConvLayerBuilder.available_conv_layer_types.get('conv2d')

        return layertype(net, num_filters=kernels, filter_size=kernel_size, pad=pad, stride=stride, W=weights, nonlinearity=nonlinearity)


class LasagnePoolLayerBuilder(BaseLasagneLayerBuilder):
    available_pool_layer_types = {
        'pool1d': l.Pool1DLayer,
        'pool2d': l.Pool2DLayer,
        'pool3d': l.Pool3DLayer,
        'maxpool1d': l.MaxPool1DLayer,
        'maxpool2d': l.MaxPool2DLayer,
        'maxpool3d': l.MaxPool3DLayer,
    }

    @staticmethod
    def build(net, layer, stage):
        pool_type = LasagnePoolLayerBuilder.getdval_str(layer['params'].get('type'), stage, 'maxpool2d').lower()
        pool_size = LasagnePoolLayerBuilder.getdval_int(layer['params'].get('pool_size'), stage, 1)
        stride = LasagnePoolLayerBuilder.getdval(layer['params'].get('stride'), stage)
        pad = LasagnePoolLayerBuilder.getdval_int(layer['params'].get('pad'), stage, 0)
        ignore_border = LasagnePoolLayerBuilder.getdval_str(layer['params'].get('ignore_border'), stage, 'true').lower()
        mode = LasagnePoolLayerBuilder.getdval_str(layer['params'].get('mode'), stage, 'max').lower()

        ignore_borderb = True if ignore_border == 'true' else False

        layer_type = LasagnePoolLayerBuilder.available_pool_layer_types.get(pool_type)

        if layer_type is None:
            layer_type = LasagnePoolLayerBuilder.available_pool_layer_types.get('maxpool2d')

        if pool_type.startswith('max'):
            return layer_type(net, pool_size, stride, pad, ignore_borderb)
        else:
            return layer_type(net, pool_size, stride, pad, ignore_borderb, mode)


class LasagneDenseLayerBuilder(BaseLasagneLayerBuilder):
    @staticmethod
    def build(net, layer, stage):
        units = LasagneDenseLayerBuilder.getdval_int(layer['params'].get('units'), stage, 1)
        nonlinearity_type = LasagneDenseLayerBuilder.getdval_str(layer['params'].get('nonlinearity'), stage).lower()

        nonlinearity = LasagneDenseLayerBuilder.get_nonlinearity(nonlinearity_type)
        weights = LasagneConvLayerBuilder.get_weights_initw(layer['params'])

        return l.DenseLayer(net, units, W=weights, nonlinearity=nonlinearity)


class LasagneDropoutLayerBuilder(BaseLasagneLayerBuilder):
    @staticmethod
    def build(net, layer, stage):
        probability = LasagneDropoutLayerBuilder.getdval_float(layer['params'].get('probability'), stage, 0.5)

        return l.DropoutLayer(net, probability)


class LasagneBatchNormLayerBuilder(BaseLasagneLayerBuilder):
    @staticmethod
    def build(net, layer, stage):
        return l.BatchNormLayer(net)
