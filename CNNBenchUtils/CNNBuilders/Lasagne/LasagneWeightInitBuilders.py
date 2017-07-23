from lasagne import init

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneWeightInitBuilder import BaseLasagneWeightInitBuilder


class ConstantWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        value = kwargs.get('value', 0.0)
        return init.Constant(value)


class UniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        nrange = kwargs.get('nrange', 0.01)
        mean = kwargs.get('mean', 0.0)
        stddev = kwargs.get('stddev')
        return init.Uniform(nrange, stddev, mean)


class NormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        stddev = kwargs.get('stddev', 0.01)
        mean = kwargs.get('mean', 0.0)
        return init.Normal(stddev, mean)


class HeNormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', 0.01)
        return init.HeNormal(gain)


class HeUniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', 0.01)
        return init.HeUniform(gain)


class GlorotNormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', 0.01)
        return init.GlorotNormal(gain)


class GlorotUniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', 0.01)
        return init.GlorotUniform(gain)


class SparseWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        stddev = kwargs.get('stddev', 0.01)
        sparsity = kwargs.get('sparsity', 0.1)
        return init.Sparse(sparsity, stddev)


class OrthoWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', 0.01)
        return init.Orthogonal(gain)
