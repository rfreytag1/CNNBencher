from lasagne import init


class BaseLasagneWeightInitFactory:
    standard_weights_init = {
        'constant': init.Constant,
        'uniform': init.Uniform,
        'normal': init.Normal,
        'henormal': init.HeNormal,
        'heuniform': init.HeUniform,
        'glorotnormal': init.GlorotNormal,
        'glorotuniform': init.GlorotUniform,
        'sparse': init.Sparse,
        'ortho': init.Orthogonal
    }

    '''
    create instance of desired weight initializer with sane defaults if no parameters were passed 
    '''
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        pass