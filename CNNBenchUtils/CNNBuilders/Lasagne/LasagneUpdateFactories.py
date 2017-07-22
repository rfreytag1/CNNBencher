from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneUpdatesFactory import BaseLasagneUpdatesFactory


class AdamUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'adam'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 0.001

        if beta1 is None:
            beta1 = 0.9

        if beta2 is None:
            beta2 = 0.999

        if epsilon is None:
            epsilon = 1e-08

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, beta1, beta2, epsilon)


class AdamaxUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'adamax'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 0.002

        if beta1 is None:
            beta1 = 0.9

        if beta2 is None:
            beta2 = 0.999

        if epsilon is None:
            epsilon = 1e-08

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, beta1, beta2, epsilon)


class AdadeltaUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'adadelta'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 1.0

        if epsilon is None:
            epsilon = 1e-06

        if rho is None:
            rho = 0.95

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, epsilon, rho)


class AdagradUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'adagrad'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 1.0

        if epsilon is None:
            epsilon = 1e-06

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, epsilon)


class MomentumUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'momentum'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 1.0

        if momentum is None:
            momentum = 0.9

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, momentum)


class NesterovMomentumUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'nesterov'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 1.0

        if momentum is None:
            momentum = 0.9

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, momentum)


class SGDUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'sgd'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 1.0

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate)


class RMSUpdatesFactory(BaseLasagneUpdatesFactory):
    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        actual_type = 'rms'
        if str(utype).lower() != actual_type:
            return None

        if learning_rate is None:
            learning_rate = 1.0

        if rho is None:
            rho = 0.9

        if epsilon is None:
            epsilon = 1e-6

        return BaseLasagneUpdatesFactory.available_updates.get(actual_type)(loss_or_grads, params, learning_rate, rho, epsilon)
