from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Given a function we apply it, return it as a scalar and keep it's history
        Importantly, we make sure to keep the context and to define the inputs that went in
        This is the most important part for autodifferentiation.

        Args:
        ----
            vals: The input values to the function.

        Returns:
        -------
            Scalar: The output scalar

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # print(c, ctx)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivatives are $f'_x(x, y) = 1$ and $f'_y(x, y) = 1$"""
        # return dz/dx * d_output, dz/dy * d_output
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Logarithm of a number"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is $f'_x(x) = 1/x$"""
        (a,) = ctx.saved_values  # returns a tuple, so unpack it
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two numbers"""
        # save values
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivatives are $f'_x(x, y) = y$ and $f'_y(x, y) = x$"""
        (a, b) = ctx.saved_values
        # now return dz/da * d_output, dz/db * d_output
        # dz/da is b, because function is a*b
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Inverse of a number, throws error if x is zero"""
        if x == 0:
            raise ValueError("Cannot compute inverse of zero")
        # and save the value
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is $f'_x(x) = -1/x^2$, but that's in the inv_back function"""
        (x,) = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Negate a number"""
        return operators.neg(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is $f'_x(x) = -1$"""
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Sigmoid of a number"""
        ctx.save_for_backward(x)
        return operators.sigmoid(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is $f'_x(x) = f(x) * (1 - f(x)) * d_output$"""
        (x,) = ctx.saved_values
        return (
            operators.sigmoid(x) * (1 - operators.sigmoid(x)) * d_output
        )  # from wikipedia


class ReLU(ScalarFunction):
    """Rectified Linear Unit function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """ReLU of a number"""
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is $f'_x(x) = 0$ if $x < 0$ else $d_output$"""
        (x,) = ctx.saved_values
        return 0 if x < 0 else d_output


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Exponential of a number"""
        ctx.save_for_backward(x)
        return operators.exp(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is $f'_x(x) = e^x * d_output$"""
        (x,) = ctx.saved_values
        return operators.exp(x) * d_output


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less than comparison of two numbers"""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """No derivative, so return 0"""
        # raise NotImplementedError("LT is not differentiable")
        return 0, 0


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equality comparison of two numbers"""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """No derivative, so return 0"""
        # raise NotImplementedError("EQ is not differentiable")
        return 0, 0
