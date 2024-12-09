from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

from typing_extensions import Protocol

import numpy as np

from . import operators
from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce
            start (float): starting value for reduction

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication
        for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))
        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out
        # raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """My explanation of the logic

        so first of all, map will be given out, out_shape, out_strides, in_storage, in_shape, in_strides

        first we make a out_index that is [0,0,0]. then we loop through, I'll stick to the 000 element for now.

        we first do to index, so if i is 0, we use the shape to find that we are at element 0,0,0, so for the first part of the loop, out_index is unchanged.
        Then we set in_index as [0,0,0] then use broadcasting, in this case the shape is the same so in_index is also set to [0,0,0].
        Then we use index to position to find the element in storage based on the strides and the multidimensional index. For the first element that is still 0.
        Then we apply our fn to in storage at that element in storage and assign it to the output

        Next you repeat but go to element 1, find its index, find its position in storage and continue repeating until we get through all the outputs
        """
        assert len(in_shape) <= len(
            out_shape
        ), "in_shape must be smaller than out_shape"
        out_index = np.array(
            [0] * len(out_shape), dtype=np.int32
        )  # a list with 0 the same amount as out shape times, so that's [0,0,0] for a 3d tensor
        for i in range(len(out)):  # loop through storage, the output one
            to_index(
                i, out_shape, out_index
            )  # get the index for the output matrix which is the larger one, out_index is modified in place

            in_index = np.array(
                [0] * len(in_shape), dtype=np.int32
            )  # first set to 0s again, could be smaller than out_index

            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # broadcast the index to the smaller one

            in_position = index_to_position(
                in_index, in_strides
            )  # now finds the position based on the strides and index
            out_position = index_to_position(out_index, out_strides)  # same for output

            out[out_position] = fn(
                in_storage[in_position]
            )  # and now in the storage we apply it

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # similar idea, the key difference is zip is with 2 tensors, so we need to get the index for both
        # like before we don't need a separate broadcast vs non broadcast shape
        out_index = np.array([0] * len(out_shape), dtype=np.int32)
        # assert(shape_broadcast(a_shape, b_shape) == out_shape) #make sure they can be broadcasted, this is already ensured tho!
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            a_index = np.array([0] * len(a_shape), dtype=np.int32)
            broadcast_index(
                out_index, out_shape, a_shape, a_index
            )  # outshape must be bigger than a and b as it is the broadcast of it
            b_index = np.array([0] * len(b_shape), dtype=np.int32)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)

            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # same idea again, but here we reduce it as well, like summing along one dimension
        # so if a is 3x3 and reduce_dim is 0 then out_shape is 1x3
        # good news is no broadcasting here

        # also fn takes in 2 elements at a time for parallelism, and it's how hardware does it, that's why not pass in independent vectors

        # same starting
        # out_index = np.array([0] * len(out_shape), dtype=np.int32)
        # for i in range(len(out)):  # loop through the output storage
        #     to_index(
        #         i, out_shape, out_index
        #     )  # get the index for output matrix, get multidimensional index

        #     a_index = np.array(
        #         list(out_index), dtype=np.int32
        #     )  # copy the index, we will modify it
        #     # a_index.insert(reduce_dim, 0)  # This is needed for pytorch implementation, but for minitorch we provide the 1 in output dimension
        #     a_position = index_to_position(
        #         a_index, a_strides
        #     )  # finds the corresponding position in the input, but need to move along the row still
        #     accumulator = a_storage[a_position]  # this is where we start

        #     # this index now means for example summing across the removed dimension, which is reduce_dim
        #     for j in range(
        #         1, a_shape[reduce_dim]
        #     ):  # loop through the removed dimension, but we already have the first value, need 2 points minimum, so skip the first element. else doesn't generalize between + and * which need 0 and 1 respectively
        #         a_index[reduce_dim] = (
        #             j  # set it to the current value along how far we have moved in that dimension that's reduced
        #         )
        #         a_position = index_to_position(
        #             a_index, a_strides
        #         )  # get the new position
        #         accumulator = fn(
        #             accumulator, a_storage[a_position]
        #         )  # combine it with the previous one!

        #     # and now we store it
        #     out_position = index_to_position(
        #         out_index, out_strides
        #     )  # just like we get the point for a, also get it for out
        #     out[out_position] = accumulator  # and now assign it based on accumulator
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return _reduce


def tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # this is a slow implementation that doesn't use numba, so it's about as slow as you can be
    # but it does let us do matmul, I manually verified several outputs and it seems similar to the
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch_size = max(a_shape[0], b_shape[0])  # Broadcasting in batch dimension
    m, n = a_shape[-2], b_shape[-1]
    k = a_shape[-1]

    for batch in range(batch_size):
        for i in range(m):
            for j in range(n):
                # Compute output index
                out_index = (
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )

                # Initialize the output element
                out[out_index] = 0

                for p in range(k):
                    # Compute indices for `a` and `b`
                    a_index = (
                        (batch * a_batch_stride if a_batch_stride else 0)
                        + i * a_strides[-2]
                        + p * a_strides[-1]
                    )
                    b_index = (
                        (batch * b_batch_stride if b_batch_stride else 0)
                        + p * b_strides[-2]
                        + j * b_strides[-1]
                    )

                    # Perform the multiplication and accumulate
                    out[out_index] += a_storage[a_index] * b_storage[b_index]

    return None


SimpleBackend = TensorBackend(SimpleOps)
