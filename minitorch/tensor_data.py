from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    # so we are given strides and it is a np array essentially
    # so we want to multiply elements and then sum them, which is np.dot
    # idx = np.dot(index, strides)
    # but do it with for loops
    # idx = 0
    # for i in range(len(index)):
    #     idx += index[i] * strides[i]
    # return idx

    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # here we want to get the index given an ordinal
    # easy option is to do a loop and keep dividing by the shape and taking the remainder
    # the idea is we just use the shape, not the stride to find the index
    # if we permute, it changes the shape, that's why don't need stride

    # for i in range(len(shape)):
    #     out_index[i] = ordinal % shape[i]  # modify the out_index in place
    #     ordinal = ordinal // shape[i]
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh

    return None


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    # we can think of this as we broadcast the small tensor all over the big tensor
    # then we want to find if we did the broadcasting what the element would be
    # we could either just broadcast, or the smarter thing is just find the index, more efficient

    # first find the union shape
    # bigshape = shape_broadcast(big_shape, shape)

    # and extend out_index to the big shape
    # out_index = [0] * len(shape)

    # find how many padded 1s are there
    # num_pad = len(bigshape) - len(shape)

    # now we loop through, start at the right so we can skip when we get to num_pad
    # for i in range(-1, -len(shape) - 1, -1):
    #     if (
    #         big_shape[i] == 1
    #     ):  # this is actually bad because big shape is smaller in a dim so we never aaccess those elements!! But rarely the case for like zip, even not for map
    #         # but there are arbitrary examples where it's true, and this doesn't make sense...
    #         out_index[i] = 0
    #     elif (
    #         shape[i] == 1
    #     ):  # if the shape dimension is 1, then we also have index be 1, this is like it's padded
    #         out_index[i] = 0
    #     else:
    #         out_index[i] = big_index[
    #             i
    #         ]  # if not 1, then just use the big index (assume it broadcasts)

    # TODO: Implement for Task 2.2.
    # raise NotImplementedError("Need to implement for Task 2.2")
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    return None


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    # TODO: Implement for Task 2.2.
    # raise NotImplementedError("Need to implement for Task 2.2")

    # so we have the 2 shapes, we need to broadcast them
    """rules
    1. dimensions of size 1 broadcast to the other shape
    2. if the dimensions are different, pad the smaller one with 1s on the left, padded with view
    3. zip will add the starting dims of 1 (not in this function)"""

    # first check if the shapes are compatible
    # if len(shape1) > len(shape2):
    #     shape1, shape2 = shape2, shape1  # better to have the smaller shape first

    # shape1tuple = tuple(shape1)
    # # now we pad the smaller shape with 1s on the left using view
    # shape1 = (1,) * (len(shape2) - len(shape1)) + shape1tuple

    # # now simply loop through and if the shapes are different and not 1, raise an error
    # out_shape = []
    # for s1, s2 in zip(shape1, shape2):
    #     if s1 == s2:
    #         out_shape.append(s1)
    #     elif s1 == 1:
    #         out_shape.append(s2)
    #     elif s2 == 1:
    #         out_shape.append(s1)
    #     else:
    #         raise IndexingError(f"Shapes {shape1} and {shape2} are not compatible.")

    # return tuple(out_shape)

    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Shapes {shape1} and {shape2} are not compatible.")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Shapes {shape1} and {shape2} are not compatible.")
    return tuple(reversed(c_rev))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a : first shape
            shape_b : second shape

        Returns:
        -------
            broadcasted shape

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Converts a multidimensional tensor `index` into a single-dimensional position in the storage

        Args:
        ----
            index : index tuple of ints for multidimensional tensor

        Returns:
        -------
            Position in storage

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Iterate over all indices in the tensor."""
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get a value from the tensor by indexing the storage."""
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set a value in the tensor to `val` by altering the storage."""
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # we simply permute by swaping the shape and strides
        # new_shape = tuple(
        #     self.shape[i] for i in order
        # )  # order is like 0,2,1 and goes through this
        # new_strides = tuple(self.strides[i] for i in order)
        # return TensorData(
        #     self._storage, new_shape, new_strides
        # )  # makes a new tensor with the new shape and strides

        return TensorData(
            self._storage,
            tuple(self.shape[i] for i in order),
            tuple(self._strides[i] for i in order),
        )

        # TODO: Implement for Task 2.1.
        # raise NotImplementedError("Need to implement for Task 2.1")

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
