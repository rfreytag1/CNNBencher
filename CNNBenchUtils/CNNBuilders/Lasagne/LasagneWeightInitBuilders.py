from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneWeightInitBuilder import BaseLasagneWeightInitBuilder


class ConstantWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'constant'
        if str(wtype).lower() != actual_type:
            return None
        if value is None:
            value = 0.0

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(value)


class UniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'uniform'
        if str(wtype).lower() != actual_type:
            return None

        nrange = nrange if nrange is not None else 0.01
        mean = mean if mean is not None else 0.0

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(nrange, stddev, mean)


class NormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'normal'
        if str(wtype).lower() != actual_type:
            return None

        stddev = stddev if stddev is not None else 0.01
        mean = mean if mean is not None else 0.0

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(stddev, mean)


class HeNormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'henormal'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(gain)


class HeUniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'heuniform'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(gain)


class GlorotNormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'glorotnormal'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(gain)


class GlorotUniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'glorotuniform'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(gain)


class SparseWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'sparse'
        if str(wtype).lower() != actual_type:
            return None

        stddev = stddev if stddev is not None else 0.01
        sparsity = sparsity if sparsity is not None else 0.1

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(sparsity, stddev)


class OrthoWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(wtype, value=None, gain=None, stddev=None, mean=None, nrange=None, sparsity=None):
        actual_type = 'ortho'
        if str(wtype).lower() != actual_type:
            return None

        gain = gain if gain is not None else 0.01

        return BaseLasagneWeightInitBuilder.standard_weights_init.get(actual_type)(gain)
