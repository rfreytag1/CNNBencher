from lasagne import init

from CNNBenchUtils.CNNBuilders.Lasagne.BaseLasagneWeightInitBuilder import BaseLasagneWeightInitBuilder


WINIT_DEFAULT_VALUE = 0.0
WINIT_DEFAULT_GAIN = 0.01
WINIT_DEFAULT_MEAN = 0.0
WINIT_DEFAULT_STDDEV = 0.01
WINIT_DEFAULT_NRANGE = 0.0
WINIT_DEFAULT_SPARSITY = 0.1


class ConstantWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        value = kwargs.get('value', WINIT_DEFAULT_VALUE)
        return init.Constant(value)


class UniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        nrange = kwargs.get('nrange', WINIT_DEFAULT_NRANGE)
        mean = kwargs.get('mean', WINIT_DEFAULT_MEAN)
        stddev = kwargs.get('stddev', WINIT_DEFAULT_STDDEV)
        return init.Uniform(nrange, stddev, mean)


class NormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        stddev = kwargs.get('stddev', WINIT_DEFAULT_STDDEV)
        mean = kwargs.get('mean', WINIT_DEFAULT_MEAN)
        return init.Normal(stddev, mean)


class HeNormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', WINIT_DEFAULT_GAIN)
        return init.HeNormal(gain)


class HeUniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', WINIT_DEFAULT_GAIN)
        return init.HeUniform(gain)


class GlorotNormalWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', WINIT_DEFAULT_GAIN)
        return init.GlorotNormal(gain)


class GlorotUniformWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', WINIT_DEFAULT_GAIN)
        return init.GlorotUniform(gain)


class SparseWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        stddev = kwargs.get('stddev', WINIT_DEFAULT_STDDEV)
        sparsity = kwargs.get('sparsity', WINIT_DEFAULT_SPARSITY)
        return init.Sparse(sparsity, stddev)


class OrthoWeightInitBuilder(BaseLasagneWeightInitBuilder):
    @staticmethod
    def build(**kwargs):
        gain = kwargs.get('gain', WINIT_DEFAULT_GAIN)
        return init.Orthogonal(gain)
