from lasagne import nonlinearities

from CNNBenchUtils.CNNBuilders.Lasagne.LasagneWeightInitBuilders import *
from CNNBenchUtils.CNNBuilders.BaseLayerBuilder import BaseLayerBuilder


class BaseLasagneLayerBuilder(BaseLayerBuilder):
    '''
    Base class for Layer Builder classes with some lasagne specific helper functions and some helpful data
    '''
    available_nonlinerities = {
        'elu': nonlinearities.elu,
        'linear': nonlinearities.linear,
        'rectify': nonlinearities.rectify,
        'sigmoid': nonlinearities.sigmoid,
        'softmax': nonlinearities.softmax,
        'softplus': nonlinearities.softplus,
        'tanh': nonlinearities.tanh
    }

    available_weights_builders = {
        'constant': ConstantWeightInitBuilder.build,
        'uniform': UniformWeightInitBuilder.build,
        'normal': NormalWeightInitBuilder.build,
        'heuniform': HeUniformWeightInitBuilder.build,
        'henormal': HeNormalWeightInitBuilder.build,
        'glorotuniform': GlorotUniformWeightInitBuilder.build,
        'glorotnormal': GlorotNormalWeightInitBuilder.build,
        'sparse': SparseWeightInitBuilder.build,
        'ortho': OrthoWeightInitBuilder.build,
    }

    @staticmethod
    def get_nonlinearity(ntype):
        '''
        returns the specified nonlinearity function or a sane default.
        :param ntype: desired nonlinearity function
        :return: function as callable
        '''
        nonlinearity = BaseLasagneLayerBuilder.available_nonlinerities.get(str(ntype).lower())
        return nonlinearity if nonlinearity is not None else nonlinearities.elu

    @staticmethod
    def get_weights_init(lparams, stage=0):
        '''
        wrapper for get_weights_init which uses a dict from a benchmark description instead
        :param lparams: parameters to be used
        :param stage: benchmark stage
        :return:
        '''

        weight_params = {}
        for pkey, pdval in lparams.items():
            if pkey.startswith('weights.'):
                weight_params[pkey[pkey.index('.')+1:]] = pdval.value(stage)

        winit_factory = BaseLasagneLayerBuilder.available_weights_builders.get(str(weight_params.get('type', 'henormal')).lower())
        if winit_factory is None:
            winit_factory = BaseLasagneLayerBuilder.available_weights_builders.get('henormal')

        winit = winit_factory(**weight_params)

        return winit

    @staticmethod
    def register_nonlinearity(nname, nlin_func):
        '''
        helper to safely and correctly add new non-linearity functions
        :param nname: name of the function(as usable in the benchmark description file)
        :param nlin_func: callable
        :raises TypeError
        '''
        if not isinstance(nname, str):
            raise TypeError('Parameter "nname" must be a string!')
        if not callable(nlin_func):
            raise TypeError('Parameter "nlin_func" must be callable!')

        BaseLasagneLayerBuilder.available_nonlinerities[nname] = nlin_func

    @staticmethod
    def register_weight_builder(fname, factory_func):
        '''
        helper function to safely and correctly add new factories for weight initializers
        :param fname: initializer name(as usable in the benchmark description file)
        :param factory_func: callable
        :raises TypeError
        '''
        if not isinstance(fname, str):
            raise TypeError('Parameter "fname" must be a string!')
        if not callable(factory_func):
            raise TypeError('Parameter "factory_func" must be callable!')

        BaseLasagneLayerBuilder.available_nonlinerities[fname] = factory_func
