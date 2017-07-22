from lasagne import objectives
import theano.tensor as t

from CNNBenchUtils.CNNBuilders.BaseFunctionBuilder import BaseFunctionBuilder


class BaseLasagneFunctionBuilder(BaseFunctionBuilder):
    def __init__(self):
        super(BaseLasagneFunctionBuilder, self).__init__()
        pass

    @staticmethod
    def calc_loss(prediction, targets):
        # categorical crossentropy is the best choice for a multi-class softmax output
        loss = t.mean(objectives.categorical_crossentropy(prediction, targets))
        return loss

    @staticmethod
    def calc_loss_multi(prediction, targets):
        # we need to clip predictions when calculating the log-loss
        prediction = t.clip(prediction, 0.0000001, 0.9999999)
        # binary crossentropy is the best choice for a multi-class sigmoid output
        loss = t.mean(objectives.binary_crossentropy(prediction, targets))
        return loss

    @staticmethod
    def calc_accuracy(prediction, targets):
        # we can use the lasagne objective categorical_accuracy to determine the top1 single label accuracy
        a = t.mean(objectives.categorical_accuracy(prediction, targets, top_k=1))
        return a

    @staticmethod
    def calc_accuracy_multi(prediction, targets):
        # we can use the lasagne objective binary_accuracy to determine the multi label accuracy
        a = t.mean(objectives.binary_accuracy(prediction, targets))
        return a
