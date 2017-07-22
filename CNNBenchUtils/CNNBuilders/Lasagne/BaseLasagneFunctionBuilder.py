from lasagne import layers
from lasagne import objectives
import theano.tensor as t

from CNNBenchUtils.CNNBuilders.BaseFunctionBuilder import BaseFunctionBuilder


class BaseLasagneFunctionBuilder(BaseFunctionBuilder):
    def __init__(self, net=None, func_desc=None, tensors=None):
        super(BaseLasagneFunctionBuilder, self).__init__(net, func_desc, tensors)

    @staticmethod
    def calc_loss(prediction, targets):
        '''
        taken straight from Stefan Kahl's code. see https://github.com/kahst/BirdCLEF2017
        :param prediction:
        :param targets:
        :return: returns loss function(?) as a tensor
        '''
        # categorical crossentropy is the best choice for a multi-class softmax output -Stefan Kahl
        loss = t.mean(objectives.categorical_crossentropy(prediction, targets))
        return loss

    @staticmethod
    def calc_loss_multi(prediction, targets):
        '''
        taken straight from Stefan Kahl's code. see https://github.com/kahst/BirdCLEF2017
        :param prediction:
        :param targets:
        :return: returns loss function(?) as a tensor
        '''
        # we need to clip predictions when calculating the log-loss -Stefan Kahl
        prediction = t.clip(prediction, 0.0000001, 0.9999999)
        # binary crossentropy is the best choice for a multi-class sigmoid output -Stefan Kahl
        loss = t.mean(objectives.binary_crossentropy(prediction, targets))
        return loss

    @staticmethod
    def calc_accuracy(prediction, targets):
        '''
        taken straight from Stefan Kahl's code. see https://github.com/kahst/BirdCLEF2017
        :param prediction:
        :param targets:
        :return: returns loss function(?) as a tensor
        '''
        # we can use the lasagne objective categorical_accuracy to determine the top1 single label accuracy -Stefan Kahl
        a = t.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
        return a

    @staticmethod
    def calc_accuracy_multi(prediction, targets):
        '''
        taken straight from Stefan Kahl's code. see https://github.com/kahst/BirdCLEF2017
        :param prediction:
        :param targets:
        :return: returns loss function(?) as a tensor
        '''
        # we can use the lasagne objective binary_accuracy to determine the multi label accuracy -Stefan Kahl
        a = t.mean(objectives.binary_accuracy(prediction, targets))
        return a

    def build(self, net=None, func_desc=None, tensors=None, stage=0):
        '''
        Just some basic input parameter validation to be used by sub-classes.
        Also see BaseFunctionBuilder
        :raises TypeError
        :raises ValueError
        :param net:
        :param func_desc:
        :param tensors:
        :param stage:
        :return:
        '''
        if net is not None:
            self.net = net
        if self.net is None:
            raise ValueError('No "net" given!')
        elif not issubclass(type(self.net), layers.Layer):
            raise TypeError('Argument "net" must be a Lasagne Layer(any type of Layer derived from lasagne.layer.Layer)!')

        if func_desc is not None:
            self.func_desc = func_desc
        if self.func_desc is None:
            raise ValueError('No function description!')
        elif not isinstance(self.func_desc, dict):
            raise TypeError('Argument "func_desc" must be a dict!')

        if tensors is not None and isinstance(tensors, dict):
            self.tensors = tensors
        if self.tensors is None:
            self.tensors = {}

        return None, None
