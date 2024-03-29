"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        #print(f'EWiseAdd: {out_grad}')
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        #print(f'multiply grad: {out_grad * rhs},{out_grad * lhs}')
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return (node.inputs[0]**Tensor(self.scalar-1)) * out_grad*Tensor(self.scalar)
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        left, right = node.inputs

        tmp1 = Tensor(1, dtype="float32") / right
        tmp2 = Tensor(-1, dtype="float32")*left/(right*right)

        return out_grad*tmp1, out_grad*tmp2
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a/self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        res = out_grad*Tensor(1)/self.scalar
        #print(f'div scalar: {res}')
        return res
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is not None:
            axes[self.axes[0]] = self.axes[1]
            axes[self.axes[1]] = self.axes[0]
        else:
            tmp = axes[-1]
            axes[-1] = axes[-2]
            axes[-2] = tmp

        return array_api.transpose(a, axes=axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        res = array_api.broadcast_to(a, self.shape)
        return res

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        #print(f'broadcast_to outgrad: {out_grad}')

        broadcast_shape = list(self.shape)
        broadcast_shape.reverse()
        input_shape = list(node.inputs[0].shape)
        input_shape.reverse()

        broad_axes = []
        final_index = len(broadcast_shape)-1
        for i, v in enumerate(broadcast_shape):
            if i < len(input_shape):
                if input_shape[i] == 1 and v > 1:
                    broad_axes.append(final_index-i)
            else:
                broad_axes.append(final_index-i)

        grad = out_grad.sum(axes=tuple(broad_axes)).reshape(node.inputs[0].shape)
        #print(f'broadcast_to grad: {grad}')

        return grad

        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        if self.axes is None:
            return out_grad.broadcast_to(node.inputs[0].shape)

        new_shape = list(node.inputs[0].shape)
        if self.axes is not None:
            if type(self.axes) is not tuple:
                new_shape[self.axes] = 1
            else:
                for i in self.axes:
                    new_shape[i] = 1
        grad = out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        #print(f'summation grad: {grad}')
        # BEGIN YOUR SOLUTION
        return grad
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        left, right = node.inputs
        right_shape_len = len(right.shape)
        left_shape_len = len(left.shape)

        right_index = list(range(right_shape_len))
        left_index = list(range(left_shape_len))
        right_transposed = right.transpose(right_index[-2:])
        left_transposed = left.transpose(left_index[-2:])

        grad_left = out_grad.matmul(right_transposed)
        grad_right = left_transposed.matmul(out_grad)

        left_extend_len = len(grad_left.shape)-left_shape_len
        right_extend_len = len(grad_right.shape)-right_shape_len

        if left_extend_len > 0:
            axes = list(range(left_extend_len))
            grad_left = grad_left.sum(tuple(axes))

        if right_extend_len > 0:
            axes = list(range(right_extend_len))
            grad_right = grad_right.sum(tuple(axes))
        #print(f'matmul grad: {grad_left},{grad_right}')
        return grad_left, grad_right
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.negative(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad*Tensor(-1)
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad*Tensor(1)/node.inputs[0]
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * Tensor(array_api.exp(node.inputs[0].realize_cached_data()))
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        # maybe use view or something like that, copy is too expensive
        data = array_api.array(a, copy=True)
        data[data < 0] = 0
        return data
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()
        data[data < 0] = 0
        data[data > 0] = 1
        return out_grad*Tensor(data)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes)

        if self.axes is not None:
            reshape_size = [1]*len(Z.shape)
            for i in range(len(Z.shape)):
                if i not in self.axes:
                    reshape_size[i] = Z.shape[i]
            resize_max_z = max_z.reshape(reshape_size)
            new_z = Z-array_api.broadcast_to(resize_max_z, Z.shape)
        else:
            new_z = Z-array_api.broadcast_to(max_z, Z.shape)

        res = array_api.log(array_api.sum(
            array_api.exp(new_z), axis=self.axes))+max_z

        return res
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()

        max_z = array_api.max(data, axis=self.axes)

        if self.axes is not None:
            reshape_size = [1]*len(data.shape)
            for i in range(len(data.shape)):
                if i not in self.axes:
                    reshape_size[i] = data.shape[i]
            resize_max_z = max_z.reshape(reshape_size)
            max_z = array_api.broadcast_to(resize_max_z, data.shape)

        exps = array_api.exp(data-max_z)

        exps_down = array_api.sum(exps, axis=self.axes)
        if self.axes is not None:
            reshape_size = [1]*len(data.shape)
            for i in range(len(data.shape)):
                if i not in self.axes:
                    reshape_size[i] = data.shape[i]
            resize_exps_down = exps_down.reshape(reshape_size)
            exps_down = array_api.broadcast_to(resize_exps_down, data.shape)
            resize_out_grad = out_grad.reshape(reshape_size)
            out_grad = resize_out_grad.broadcast_to(data.shape)

        res = Tensor(array_api.multiply(
            out_grad.realize_cached_data(), exps/exps_down))

        #print(f'logsumexp grad: {res}')
        return res
        # END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
