"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional, Union

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Negation forward"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Negation backward"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Inverse forward"""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Inverse backward"""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Addition forward"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Addition backward"""
        return (
            grad_output,
            grad_output,
        )  # can prove that it's the same as taking independent values, like this


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Elementwise multiplication forward"""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(
            t1, t2
        )  # t2.f would be the same as t1.f, but the reason for tensor.functional is so you can change based on hardware, like GPU or CPU

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Elementwise multiplication backward"""
        # grad_output.f.mul_zip(grad_output, ctx.saved_values[1]), grad_output.f.mul_zip(grad_output, ctx.saved_values[0])
        t1, t2 = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Sigmoid tensor Forward"""
        sigmoid = a.f.sigmoid_map(a)
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Sigmoid tensor Backward"""
        (sigmoid,) = ctx.saved_values
        neg = grad_output.f.neg_map(sigmoid)
        diff = grad_output.f.add_zip(grad_output._ensure_tensor(1), neg)
        mult = grad_output.f.mul_zip(sigmoid, diff)
        return grad_output.f.mul_zip(grad_output, mult)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Relu tensor Forward"""
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Relu tensor Backward"""
        (a,) = ctx.saved_values
        # we don't have gtzip so let's make a negative then do lt zip
        aneg = grad_output.f.neg_map(a)
        g0 = grad_output.f.lt_zip(
            aneg, grad_output._ensure_tensor(0)
        )  # find which are greater than 0, sets that to 1 independently
        return grad_output.f.mul_zip(grad_output, g0)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Log tensor Forward"""
        ctx.save_for_backward(a)
        return a.f.log_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Log tensor Backward"""
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)
        # ainv = grad_output.f.inv_map(a)
        # return grad_output.f.mul_zip(grad_output, ainv)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Exp tensor Forward"""
        ctx.save_for_backward(a)
        return a.f.exp_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Exp tensor Backward"""
        (a,) = ctx.saved_values
        exp = grad_output.f.exp_map(a)
        return grad_output.f.mul_zip(grad_output, exp)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Sum tensor Forward, specify dimension"""
        # ctx.save_for_backward(dim)
        if dim is not None:
            return a.f.add_reduce(a, int(dim.item()))
        else:
            return a.f.add_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Sum tensor Backward"""
        # (dim,) = ctx.saved_values
        return grad_output, 0.0

    #     ctx.save_for_backward(dim)
    #     if dim is None:
    #         return a.f.add_reduce(
    #             a.contiguous().view(int(operators.prod(a.shape))), 0
    #         )  # same logic as from All
    #     return a.f.add_reduce(a, int(dim.item()))

    # @staticmethod
    # def backward(
    #     ctx: Context, grad_output: Tensor
    # ) -> Union[Tensor, Tuple[Tensor, int]]:
    #     """Sum tensor Backward"""
    #     (dim,) = ctx.saved_values
    #     if dim is None:
    #         return grad_output  # expand is automatically called and expands it along the right dimension
    #     else:
    #         return (
    #             grad_output,
    #             0,
    #         )  # need to return same number of items as there are inputs! no derivative for dim tho


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Less than tensor Forward"""
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[int, int]:
        """Less than tensor Backward"""
        return 0, 0


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Equal tensor Forward"""
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[int, int]:
        """Equal tensor Backward"""
        return 0, 0


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """IsClose tensor Forward"""
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[int, int]:
        """IsClose tensor Backward"""
        return 0, 0


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, *dims: Tensor) -> Tensor:
        """Permute tensor Forward"""
        ctx.save_for_backward(dims)
        dims2 = [int(dim.item()) for dim in dims]
        temptensor = a._tensor.permute(*dims2)
        ts = minitorch.Tensor.make(
            temptensor._storage,
            temptensor.shape,
            strides=temptensor.strides,
            backend=a.backend,
        )
        return ts

    @staticmethod
    def backward(
        ctx: Context, grad_output: Tensor
    ) -> Tuple[
        Union[Tensor, int], ...
    ]:  # ideally would be Tuple[Tensor, int, ...] or Tuple[Tensor, *tuple[int]] but not allowed for github classroom python version
        """Permute tensor Backward"""
        (dims,) = ctx.saved_values
        dims2 = [int(dim.item()) for dim in dims]
        inv_dims = [0] * len(dims2)
        out_grad = [0] * len(dims2)
        for i, d in enumerate(dims2):
            inv_dims[d] = i
        # print(dims,inv_dims)
        # now we want to do the reverse of permute
        grad_tensor = grad_output._tensor.permute(*inv_dims)
        ts = minitorch.Tensor.make(
            grad_tensor._storage,
            grad_tensor.shape,
            strides=grad_tensor.strides,
            backend=grad_output.backend,
        )
        return (
            ts,
            *out_grad,
        )  # returning the same number of items as there are inputs, out grad is all 0s for the axes


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """View tensor Forward"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )  # actually a bit complex, i'll trust the code


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference for a function, used for gradient checking."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
