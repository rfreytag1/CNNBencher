from lasagne import updates

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneUpdatesBuilder import BaseLasagneUpdatesBuilder


class AdamUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)

        if learning_rate is None:
            learning_rate = 0.001

        return updates.adam(loss_or_grads, params, learning_rate, beta1, beta2, epsilon)


class AdamaxUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)

        if learning_rate is None:
            learning_rate = 0.002

        return updates.adamax(loss_or_grads, params, learning_rate, beta1, beta2, epsilon)


class AdadeltaUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        epsilon = kwargs.get('epsilon', 1e-6)
        rho = kwargs.get('rho', 0.95)

        if learning_rate is None:
            learning_rate = 1.0

        return updates.adadelta(loss_or_grads, params, learning_rate, epsilon, rho)


class AdagradUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        epsilon = kwargs.get('epsilon', 1e-6)
        if learning_rate is None:
            learning_rate = 1.0

        return updates.adagrad(loss_or_grads, params, learning_rate, epsilon)


class MomentumUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        momentum = kwargs.get('momentum', 0.9)

        if learning_rate is None:
            learning_rate = 1.0

        return updates.momentum(loss_or_grads, params, learning_rate, momentum)


class NesterovMomentumUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        momentum = kwargs.get('momentum', 0.9)

        if learning_rate is None:
            learning_rate = 1.0

        return updates.nesterov_momentum(loss_or_grads, params, learning_rate, momentum)


class SGDUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        if learning_rate is None:
            learning_rate = 1.0

        return updates.sgd(loss_or_grads, params, learning_rate)


class RMSUpdatesBuilder(BaseLasagneUpdatesBuilder):
    @staticmethod
    def build(loss_or_grads=None, params=None, learning_rate=None, **kwargs):
        rho = kwargs.get('rho', 0.9)
        epsilon = kwargs.get('epsilon', 1e-6)

        if learning_rate is None:
            learning_rate = 1.0

        return updates.rmsprop(loss_or_grads, params, learning_rate, rho, epsilon)
