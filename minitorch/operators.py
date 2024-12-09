"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiplication of 2 numbers

    Args:
    ----
        x: the first number
        y: the second numbers

    Returns:
    -------
        the product of the two numbers

    """
    return x * y


def id(x: float) -> float:
    """Identity function

    Args:
    ----
        x: the input number

    Returns:
    -------
        the input number

    """
    return x


def add(x: float, y: float) -> float:
    """Addition of 2 numbers

    Args:
    ----
        x: the first number
        y: the second number

    Returns:
    -------
        the sum of the two numbers

    """
    return x + y


def neg(x: float) -> float:
    """Negation of a number

    Args:
    ----
        x: the input number

    Returns:
    -------
        the negation of the input number

    """
    return float(-x)


def lt(x: float, y: float) -> float:
    """Check if the first number is less than the second number

    Args:
    ----
        x: the first number
        y: the second number

    Returns:
    -------
        True if the first number is less than the second number, False otherwise

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if the two numbers are equal

    Args:
    ----
        x: the first number
        y: the second number

    Returns:
    -------
        True if the two numbers are equal, False otherwise

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of the two numbers

    Args:
    ----
        x: the first number
        y: the second number

    Returns:
    -------
        the maximum of the two numbers

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if the two numbers are close

    Args:
    ----
        x: the first number
        y: the second number

    Returns:
    -------
        True if the two numbers are close, False otherwise

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0
    # note that this might be wrong if your precision is low and you're using small numbers


def sigmoid(x: float) -> float:
    """Compute the sigmoid of a number

    Args:
    ----
        x: the input number

    Returns:
    -------
        the sigmoid of the input number

    """
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))  # more numerically stable


def relu(x: float) -> float:
    """Compute the ReLU of a number

    Args:
    ----
        x: the input number

    Returns:
    -------
        the ReLU of the input number

    """
    return x if x > 0 else 0.0


EPS = 1e-50


def log(x: float) -> float:
    """Compute the natural logarithm of a number

    Args:
    ----
        x: the input number

    Returns:
    -------
        the natural logarithm of the input number

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential of a number

    Args:
    ----
        x: the input number

    Returns:
    -------
        the exponential of the input number

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the inverse of a number

    Args:
    ----
        x: the input number

    Returns:
    -------
        the inverse of the input number

    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log * second arg

    Args:
    ----
        x: the input number whose derviative is computed
        y: the second number that is multiplied by the derivative

    Returns:
    -------
        the derivative of log * second arg

    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of inv * second arg

    Args:
    ----
        x: the input number whose derviative is computed
        y: the second number that is multiplied by the derivative

    Returns:
    -------
        the derivative of inv * second arg

    """
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu * second arg

    Args:
    ----
        x: the input number whose derviative is computed
        y: the second number that is multiplied by the derivative

    Returns:
    -------
        the derivative of relu * second arg

    """
    if x > 0:
        return y
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each individual float in the list

    Args:
    ----
        fn: the function to apply to each float

    Returns:
    -------
        a function which applies the function to each float in the list

    """

    # return [fn(x) for x in arr]
    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines iterables from two lists using a provided function

    Args:
    ----
        fn: the function to combine the two floats
        a: the first list of floats
        b: the second list of floats

    Returns:
    -------
        a list of floats where the function has been applied to each pair of floats

    """

    # return [fn(x, y) for x, y in zip(a, b)]
    def _zipWith(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(a, b):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce a list of floats to a single float

    Args:
    ----
        fn: the function to reduce the list of floats
        arr: the list of floats
        start: the starting value for the reduction

    Returns:
    -------
        the reduced float

    """

    # ensure arr is a list
    # arr = list(arr)
    # if len(arr) == 0:
    #     return 0  # nothing to reduce
    # tempresult = arr[0]
    # if len(arr) == 1:
    #     return tempresult
    # for i in range(1, len(arr)):
    #     tempresult = fn(tempresult, arr[i])
    # return tempresult
    def _reduce(arr: Iterable[float]) -> float:
        val = start
        for x in arr:
            val = fn(val, x)
        return val

    return _reduce


def negList(arr: Iterable[float]) -> Iterable[float]:
    """Negate each float in the list

    Args:
    ----
        arr: the list of floats

    Returns:
    -------
        a list of floats where each float has been negated

    """
    return map(neg)(arr)


def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    """Add each float in the two lists together

    Args:
    ----
        arr1: the first list of floats
        arr2: the second list of floats

    Returns:
    -------
        a list of floats where each pair of floats has been added together

    """
    return zipWith(add)(arr1, arr2)


def sum(arr: Iterable[float]) -> float:
    """Sum all the floats in the list

    Args:
    ----
        arr: the list of floats

    Returns:
    -------
        the sum of all the floats in the list

    """
    return reduce(add, 0)(arr)


def prod(arr: Iterable[float]) -> float:
    """Take the product of all the floats in the list

    Args:
    ----
        arr: the list of floats

    Returns:
    -------
        the product of all the floats in the list

    """
    return reduce(mul, 1)(arr)
