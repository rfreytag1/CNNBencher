from lasagne import regularization
from lasagne import layers as l

import theano
import theano.tensor as t

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneFunctionBuilder import BaseLasagneFunctionBuilder
from CNNBenchUtils.CNNBuilders.Lasagne.LasagneUpdateBuilders import *


class LasagneTrainingFunctionBuilder(BaseLasagneFunctionBuilder):
    update_factories = {
        'adam': AdamUpdatesBuilder.build,
        'adamax': AdamaxUpdatesBuilder.build,
        'adadelta': AdadeltaUpdatesBuilder.build,
        'adagrad': AdagradUpdatesBuilder.build,
        'momentum': MomentumUpdatesBuilder.build,
        'nesterov': NesterovMomentumUpdatesBuilder.build,
        'sgd': SGDUpdatesBuilder.build,
        'rms': RMSUpdatesBuilder.build
    }

    def __init__(self):
        super(LasagneTrainingFunctionBuilder, self).__init__()

    @staticmethod
    def add_update_factory(ufname, fact_func):
        if not isinstance(ufname, str):
            raise TypeError('Argument "ufname" must be a string!')
        if not callable(fact_func):
            raise TypeError('Argument "fact_funct" must be callable!')

        LasagneTrainingFunctionBuilder.update_factories[ufname] = fact_func

    def build(self, net=None, func_desc=None, tensors=None, stage=0):
        '''
        builds the training function based on the function description given in the benchmark file
        :param net: neural net to build the function for. Must be a lasagne.layers.Layer derived type
        :param func_desc: part of the benchmark description that describes the training function as a dict.
        :param tensors: dict of tensors. Existing tensors will be used, missing will be added and since this is mutable the caller will have an updated dict for further usage
        :param stage: benchmark stage for which to build
        :return: training function as a callable and the tensors dict
        '''
        super(LasagneTrainingFunctionBuilder, self).build(net, func_desc, tensors, stage)

        # get parameters to determine how to build the training function
        update_type = LasagneTrainingFunctionBuilder.getdval_str(self.func_desc['params'].get('update.type'), stage, 'adam').lower()

        # pack all 'update.*' parameters into a dict to be used to instantiate the update function
        update_params = {}
        for pkey, pdval in self.func_desc['params'].items():
            if str(pkey).startswith('update.'):
                # for the key we just keep everything after the dot
                # the value is the current value of the dynamic value in that stage
                update_params[pkey[pkey.index('.')+1:]] = pdval.value(stage)

        l2_reg_w = LasagneTrainingFunctionBuilder.getdval_float(self.func_desc['params'].get('regularization.l2_weight', 1e-4), stage)
        multilabel = bool(LasagneTrainingFunctionBuilder.getdval_str(self.func_desc['params'].get('loss.multilabel'), stage, 'false').lower())

        # determine which update function to use. If specified is unavailable, default to adam
        update_func = LasagneTrainingFunctionBuilder.update_factories.get(update_type)
        if update_func is None:
            update_func = LasagneTrainingFunctionBuilder.update_factories.get('adam')

        targets = self.tensors.get('targets')
        if targets is None:
            targets = t.matrix('targets', dtype=theano.config.floatX)
            self.tensors['targets'] = targets

        # we use L2 Norm for regularization(static for now)
        l2_reg = regularization.regularize_layer_params(self.net, regularization.l2) * l2_reg_w
        prediction = l.get_output(self.net)

        loss = self.tensors.get('loss')
        if loss is None:
            if multilabel:
                loss = LasagneTrainingFunctionBuilder.calc_loss_multi(prediction, targets) + l2_reg
            else:
                loss = LasagneTrainingFunctionBuilder.calc_loss(prediction, targets) + l2_reg
            self.tensors['loss'] = loss

        # use a tensor so we can adapt the learning rate
        lr_dynamic = self.tensors.get('learning_rate')
        if lr_dynamic is None:
            lr_dynamic = t.scalar(name='learning_rate')
            self.tensors['learning_rate'] = lr_dynamic

        # get all trainable parameters (weights) of the net
        params = l.get_all_params(self.net, trainable=True)

        # call the previously determined update function(through its wrapper
        param_updates = update_func(loss, params, lr_dynamic, **update_params)

        self.func = theano.function([l.get_all_layers(self.net)[0].input_var, targets, lr_dynamic], loss, updates=param_updates)

        return self.func, self.tensors
