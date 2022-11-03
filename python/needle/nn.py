"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(
            fan_in=in_features, fan_out=out_features, requires_grad=True)
        if bias == True:
            self.bias = init.kaiming_uniform(
                fan_in=out_features, fan_out=1, requires_grad=True).reshape((1, out_features))
        # END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        x_shape = X.shape  # (batch_size, in_features)
        # BEGIN YOUR SOLUTION
        a = ops.matmul(X, self.weight)  # (batch_size,out_features)
        if self.bias == None:
            return a

        broadcast_size = [i for i in x_shape]
        broadcast_size[-1] = self.out_features
        b = ops.broadcast_to(self.bias, shape=tuple(broadcast_size))
        return a+b
        # END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.relu(x)
        # END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        # END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION
        num_classes = logits.shape[1]
        onehot = init.one_hot(num_classes, y)

        # might be overflow
        z_y = ops.multiply(onehot, ops.exp(logits))
        exps_up = ops.log(ops.summation(z_y, axes=(1)))

        exps_down = ops.logsumexp(logits, axes=(1,))

        return ops.summation(exps_down-exps_up)/logits.shape[0]
        # END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        batch_size, in_features = x.shape  # (batch_size, in_features)
        mean = (ops.summation(x, axes=(1,)) /
                in_features).reshape((batch_size, 1))
        broadcast_mean = ops.broadcast_to(mean, (batch_size, in_features))
        var = (ops.summation(ops.power_scalar(
            x-broadcast_mean, 2), axes=(1,))/in_features).reshape((batch_size, 1))
        broadcast_var = ops.broadcast_to(var, (batch_size, in_features))
        new_x = ops.divide((x-broadcast_mean),
                           ops.power_scalar((broadcast_var+self.eps), 0.5))  # (batch_size, in_features)

        broadcast_weight = ops.broadcast_to(
            self.weight, (batch_size, in_features))
        broadcast_bias = ops.broadcast_to(self.bias, (batch_size, in_features))
        return ops.multiply(new_x, broadcast_weight)+broadcast_bias

        # END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION
