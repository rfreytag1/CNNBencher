from CNNBenchUtils.CNNBuilders.BaseBuilder import BaseBuilder


class BaseFunctionBuilder(BaseBuilder):
    def __init__(self, net=None, func_desc=None, tensors=None):
        self.func = None
        self.func_desc = func_desc
        self.net = net
        self.tensors = tensors
        if self.tensors is None:
            self.tensors = {}

    def build(self, net=None, func_desc=None, tensors=None, stage=0):
        '''
        Builds a neural net function based on the NN given as "net" for the current benchmark stage
        :param net: neural net to build function for
        :param func_desc: dict of parameters used for building the function
        :param tensors: dict from which tensors are taken or if not yet available they're stored in there. Since this is mutable the caller will get the updated dict for use in future function builds
        :param stage: current benchmark stage for which this should be configured
        :return: returns a callable and the tensors dict
        '''
        pass
