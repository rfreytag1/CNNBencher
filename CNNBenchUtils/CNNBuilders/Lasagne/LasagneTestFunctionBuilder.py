from lasagne import regularization
from lasagne import layers as l

import theano
import theano.tensor as t

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneFunctionBuilder import BaseLasagneFunctionBuilder


class LasagneTestFunctionBuilder(BaseLasagneFunctionBuilder):
    def __init__(self):
        super(LasagneTestFunctionBuilder, self).__init__()

    def build(self, net, func, tensors=None, stage=0):
        self.func_desc = func
        self.net = net
        if tensors is not None and isinstance(tensors, dict):
            self.tensors = tensors
        l2_reg_w = LasagneTestFunctionBuilder.getdval_float(func['params'].get('regularization.l2_weight', 1e-4), stage)
        multilabel = LasagneTestFunctionBuilder.getdval_str(func['params'].get('loss.multilabel'), stage, 'false').lower()
        multilabelb = True if multilabel == 'true' else False

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
                loss = LasagneTestFunctionBuilder.calc_loss_multi(prediction, targets) + l2_reg
            else:
                loss = LasagneTestFunctionBuilder.calc_loss(prediction, targets) + l2_reg
            self.tensors['loss'] = loss

        accuracy = self.tensors.get('accuracy')
        if accuracy is None:
            if multilabelb:
                accuracy = LasagneTestFunctionBuilder.calc_accuracy_multi(prediction, targets)
            else:
                accuracy = LasagneTestFunctionBuilder.calc_accuracy(prediction, targets)
            self.tensors['accuracy'] = accuracy

        net_output = l.get_output(net, deterministic=True)

        self.func = theano.function([l.get_all_layers(net)[0].input_var, targets], [net_output, loss, accuracy])

        return self.func

    def rebuild(self, stage=0):
        return self.build(self.net, self.func_desc, stage)
