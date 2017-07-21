from lasagne import nonlinearities

from CNNBenchUtils.CNNBuilders.Lasagne.LasagneWeightInitFactories import *
from CNNBenchUtils.CNNBuilders.BaseLayerBuilder import BaseLayerBuilder


class BaseLasagneLayerBuilder(BaseLayerBuilder):
    available_nonlinerities = {
        'elu': nonlinearities.elu,
        'linear': nonlinearities.linear,
        'rectify': nonlinearities.rectify,
        'sigmoid': nonlinearities.sigmoid,
        'softmax': nonlinearities.softmax,
        'softplus': nonlinearities.softplus,
        'tanh': nonlinearities.tanh
    }

    available_weights_factories = {
        'constant': ConstantWeightInitFactory.instance,
        'uniform': UniformWeightInitFactory.instance,
        'normal': NormalWeightInitFactory.instance,
        'heuniform': HeUniformWeightInitFactory.instance,
        'henormal': HeNormalWeightInitFactory.instance,
        'glorotuniform': GlorotUniformWeightInitFactory.instance,
        'glorotnormal': GlorotNormalWeightInitFactory.instance,
        'sparse': SparseWeightInitFactory.instance,
        'ortho': OrthoWeightInitFactory.instance,
    }

    @staticmethod
    def get_nonlinearity(ntype):
        nonlinearity = BaseLasagneLayerBuilder.available_nonlinerities.get(str(ntype).lower())
        return nonlinearity if nonlinearity is not None else nonlinearities.elu

    @staticmethod
    def get_weights_init(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        winit_factory = BaseLasagneLayerBuilder.available_weights_factories.get(str(wtype).lower())
        if winit_factory is None:
            winit_factory = BaseLasagneLayerBuilder.available_weights_factories.get('henormal')
            wtype = 'henormal'

        gain = gain if gain is not None else 0.01

        winit = winit_factory(wtype, value, gain, stddev, mean, nrange, sparsity)
        print(winit)

        return winit if winit is not None else BaseLasagneLayerBuilder.available_weights_factories.get('henormal')(gain)

    @staticmethod
    def get_weights_initw(lparams, stage=0):
        weights_type = BaseLasagneLayerBuilder.getdval_str(lparams.get('weights.type'), stage).lower()
        weights_gain = BaseLasagneLayerBuilder.getdval(lparams.get('weights.gain'), stage)
        weights_stddev = BaseLasagneLayerBuilder.getdval(lparams.get('weights.stddev'), stage)
        weights_mean = BaseLasagneLayerBuilder.getdval(lparams.get('weights.mean'), stage)
        weights_range = BaseLasagneLayerBuilder.getdval(lparams.get('weights.range'), stage)
        weights_value = BaseLasagneLayerBuilder.getdval(lparams.get('weights.value'), stage)
        weights_sparsity = BaseLasagneLayerBuilder.getdval(lparams.get('weights.sparsity'), stage)

        return BaseLasagneLayerBuilder.get_weights_init(weights_type, weights_value, weights_gain, weights_stddev, weights_mean, weights_range, weights_sparsity)

    @staticmethod
    def register_nonlinearity(nname, nlin_func):
        if not isinstance(nname, str):
            raise TypeError('Parameter "nname" must be a string!')
        if not callable(nlin_func):
            raise TypeError('Parameter "nlin_func" must be callable!')

        BaseLasagneLayerBuilder.available_nonlinerities[nname] = nlin_func

    @staticmethod
    def register_weight_factory(fname, factory_func):
        if not isinstance(fname, str):
            raise TypeError('Parameter "fname" must be a string!')
        if not callable(factory_func):
            raise TypeError('Parameter "factory_func" must be callable!')

        BaseLasagneLayerBuilder.available_nonlinerities[fname] = factory_func

    @staticmethod
    def build(net, layer, stage):
        pass