from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneWeightInitFactory import BaseLasagneWeightInitFactory


class ConstantWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'constant'
        if str(wtype).lower() != actual_type:
            return None
        if value is None:
            value = 0.0

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(value)


class UniformWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'uniform'
        if str(wtype).lower() != actual_type:
            return None

        nrange = nrange if nrange is not None else 0.01
        mean = mean if mean is not None else 0.0

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(nrange, stddev, mean)


class NormalWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'normal'
        if str(wtype).lower() != actual_type:
            return None

        stddev = stddev if stddev is not None else 0.01
        mean = mean if mean is not None else 0.0

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(stddev, mean)


class HeNormalWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'henormal'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(gain)


class HeUniformWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'heuniform'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(gain)


class GlorotNormalWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'glorotnormal'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(gain)


class GlorotUniformWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'glorotuniform'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(gain)


class SparseWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'sparse'
        if str(wtype).lower() != actual_type:
            return None

        stddev = stddev if stddev is not None else 0.01
        sparsity = sparsity if sparsity is not None else 0.1

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(sparsity, stddev)


class OrthoWeightInitFactory(BaseLasagneWeightInitFactory):
    @staticmethod
    def instance(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'ortho'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitFactory.standard_weights_init.get(actual_type)(gain)
