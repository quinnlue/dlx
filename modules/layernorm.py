from ..nn.tensor import Tensor
from ..nn.module import Module
from ..utils.backend import xp

class LayerNorm(Module):
    def __init__(self, shape: int, axis: int, module_dict, layer_dict, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.axis = axis
        self.shape = shape
        self.gamma = Tensor(xp.ones(shape), requires_grad=True, name='gamma')
        self.beta = Tensor(xp.zeros(shape), requires_grad=True, name='beta')
        self.register_parameter(param=self.gamma, module_dict=module_dict, layer_type=layer_dict["type"], layer_dict=layer_dict, name="gamma")
        self.register_parameter(param=self.beta, module_dict=module_dict, layer_type=layer_dict["type"], layer_dict=layer_dict, name="beta")

    def forward(self, x: Tensor, axis=-1):
        mean = x.mean(axis=axis, keepdims=True)
        var = ((x - mean).pow(2)).mean(axis=axis, keepdims=True)
        out = (x - mean) / (var + self.eps).pow(0.5)
        return out * self.gamma + self.beta
    
    def __call__(self, x):
        return self.forward(x)