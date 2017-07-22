from lasagne import updates


class BaseLasagneUpdatesFactory:
    available_updates = {
        'adam': updates.adam,
        'adamax': updates.adamax,
        'adadelta': updates.adadelta,
        'adagrad': updates.adagrad,
        'momentum': updates.momentum,
        'nesterov': updates.nesterov_momentum,
        'sgd': updates.sgd,
        'rms': updates.rmsprop
    }

    @staticmethod
    def instance(utype, loss_or_grads=None, params=None, learning_rate=None, beta1=None, beta2=None, epsilon=None, rho=None, momentum=None):
        pass
