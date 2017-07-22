from lasagne import updates
from lasagne import objectives
from lasagne import regularization
from lasagne import layers as l

import theano
import theano.tensor as T

from CNNBenchUtils.CNNBuilders.BaseTrainingFunctionBuilder import BaseTrainingFunctionBuilder
from CNNBenchUtils.CNNBuilders.Lasagne.LasagneUpdateFactories import *


class LasagneTrainingFunctionBuilder(BaseTrainingFunctionBuilder):
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


    @staticmethod
    def calc_loss(prediction, targets):
        # categorical crossentropy is the best choice for a multi-class softmax output
        loss = T.mean(objectives.categorical_crossentropy(prediction, targets))
        return loss

    @staticmethod
    def calc_loss_multi(prediction, targets):
        # we need to clip predictions when calculating the log-loss
        prediction = T.clip(prediction, 0.0000001, 0.9999999)
        # binary crossentropy is the best choice for a multi-class sigmoid output
        loss = T.mean(objectives.binary_crossentropy(prediction, targets))
        return loss

    @staticmethod
    def calc_accuracy(prediction, targets):
        # we can use the lasagne objective categorical_accuracy to determine the top1 single label accuracy
        a = T.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
        return a

    @staticmethod
    def calc_accuracy_multi(prediction, targets):
        # we can use the lasagne objective binary_accuracy to determine the multi label accuracy
        a = T.mean(objectives.binary_accuracy(prediction, targets))
        return a

    @staticmethod
    def build(net, func, stage=0):
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

        targets = T.matrix('targets', dtype=theano.config.floatX)

        # we use L2 Norm for regularization
        l2_reg = regularization.regularize_layer_params(net, regularization.l2) * l2_reg_w
        # get the network output
        prediction = l.get_output(net)

        if multilabelb:
            loss = LasagneTrainingFunctionBuilder.calc_loss_multi(prediction, targets) + l2_reg
        else:
            loss = LasagneTrainingFunctionBuilder.calc_loss(prediction, targets) + l2_reg

        # we use dynamic learning rates which change after some epochs
        lr_dynamic = T.scalar(name='learning_rate')

        # get all trainable parameters (weights) of our net
        params = l.get_all_params(net, trainable=True)

        param_updates = update_func('adam', loss, params, lr_dynamic, update_b1, update_b2, update_epsilon, update_rho, update_momentum)

        return theano.function([l.get_all_layers(net)[0].input_var, targets, lr_dynamic], loss, updates=param_updates)
