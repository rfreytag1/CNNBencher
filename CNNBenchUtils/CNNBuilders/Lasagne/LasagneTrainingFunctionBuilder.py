from lasagne import regularization
from lasagne import layers as l

import theano
import theano.tensor as t

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneFunctionBuilder import BaseLasagneFunctionBuilder
from CNNBenchUtils.CNNBuilders.Lasagne.LasagneUpdateFactories import *


class LasagneTrainingFunctionBuilder(BaseLasagneFunctionBuilder):
    update_factories = {
        'adam': AdamUpdatesFactory.instance,
        'adamax': AdamaxUpdatesFactory.instance,
        'adadelta': AdadeltaUpdatesFactory.instance,
        'adagrad': AdagradUpdatesFactory.instance,
        'momentum': MomentumUpdatesFactory.instance,
        'nesterov': NesterovMomentumUpdatesFactory.instance,
        'sgd': SGDUpdatesFactory.instance,
        'rms': RMSUpdatesFactory.instance
    }

    def __init__(self):
        super(LasagneTrainingFunctionBuilder, self).__init__()

    '''
    net - neural net on which the training function should act
    func - dict with the function build description/parameters
    tensors - dict with tensors that should be used. if particular tensor doesn't exist yet, an instance will be stored for later use
    stage - benchmark stage to use as basis for parameter value selection
    '''
    def build(self, net, func, tensors=None, stage=0):
        self.func_desc = func
        self.net = net
        if tensors is not None and isinstance(tensors, dict):
            self.tensors = tensors

        update_type = LasagneTrainingFunctionBuilder.getdval_str(func['params'].get('update.type'), stage, 'adam').lower()
        update_b1 = LasagneTrainingFunctionBuilder.getdval(func['params'].get('update.beta1'), stage)
        update_b2 = LasagneTrainingFunctionBuilder.getdval(func['params'].get('update.beta2'), stage)
        update_epsilon = LasagneTrainingFunctionBuilder.getdval(func['params'].get('update.epsilon'), stage)
        update_rho = LasagneTrainingFunctionBuilder.getdval(func['params'].get('update.rho'), stage)
        update_momentum = LasagneTrainingFunctionBuilder.getdval(func['params'].get('update.momentum'), stage)
        l2_reg_w = LasagneTrainingFunctionBuilder.getdval_float(func['params'].get('regularization.l2_weight', 1e-4), stage)
        multilabel = LasagneTrainingFunctionBuilder.getdval_str(func['params'].get('loss.multilabel'), stage, 'false').lower()
        multilabelb = True if multilabel == 'true' else False

        update_func = LasagneTrainingFunctionBuilder.update_factories.get(update_type)
        if update_func is None:
            update_func = LasagneTrainingFunctionBuilder.update_factories.get('adam')

        targets = self.tensors.get('targets')
        if targets is None:
            targets = t.matrix('targets', dtype=theano.config.floatX)
            self.tensors['targets'] = targets

        # we use L2 Norm for regularization
        l2_reg = regularization.regularize_layer_params(net, regularization.l2) * l2_reg_w
        # get the network output
        prediction = l.get_output(net)

        loss = self.tensors.get('loss')
        if loss is None:
            if multilabelb:
                loss = LasagneTrainingFunctionBuilder.calc_loss_multi(prediction, targets) + l2_reg
            else:
                loss = LasagneTrainingFunctionBuilder.calc_loss(prediction, targets) + l2_reg
            self.tensors['loss'] = loss

        # we use dynamic learning rates which change after some epochs
        lr_dynamic = self.tensors.get('learning_rate')
        if lr_dynamic is None:
            lr_dynamic = t.scalar(name='learning_rate')
            self.tensors['learning_rate'] = lr_dynamic

        # get all trainable parameters (weights) of our net
        params = l.get_all_params(net, trainable=True)

        param_updates = update_func('adam', loss, params, lr_dynamic, update_b1, update_b2, update_epsilon, update_rho, update_momentum)

        self.func = theano.function([l.get_all_layers(net)[0].input_var, targets, lr_dynamic], loss, updates=param_updates)

        return self.func

    def rebuild(self, stage=0):
        return self.build(self.net, self.func_desc, stage)
