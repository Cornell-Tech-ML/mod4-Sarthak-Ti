from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile functions with NUMBA."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

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
        # Make these always be a 3 dimensional multiply
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

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

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
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")
        # Ensure that in_shape is smaller than or equal to out_shape dimensions
        assert len(in_shape) <= len(
            out_shape
        ), "in_shape must be smaller than out_shape"
        strides_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(
            out_shape, in_shape
        )

        # the first key idea is we check if the strides and the shapes are the same, if they are we can avoid indexing
        # then we use prange to parallelize the loop
        # regardless, the idea is find the index of the input, apply function and store in output

        # Iterate over each element in `out` in parallel
        # print(out.size)
        if strides_aligned:
            # If strides are aligned, we can avoid indexing, simply apply the function
            for i in prange(out.size):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(out.size):  # using prange for parallel looping
                out_index = np.empty(len(out_shape), dtype=np.int32)
                in_index = np.empty(len(in_shape), dtype=np.int32)
                # Get the multi-dimensional index for the current flat index `i`
                to_index(i, out_shape, out_index)

                # Broadcast out_index to in_index based on the input shape
                broadcast_index(out_index, out_shape, in_shape, in_index)

                # Calculate flat positions in input and output based on strides
                in_position = index_to_position(in_index, in_strides)
                out_position = index_to_position(out_index, out_strides)

                # Apply the function and store the result in the output storage
                out[out_position] = fn(in_storage[in_position])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

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
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")
        strides_aligned = (
            np.array_equal(b_strides, a_strides)
            and np.array_equal(a_shape, b_shape)
            and np.array_equal(a_shape, out_shape)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
        )

        # exact same idea as tensor_map, we check if strides and shapes are the same, if they are we can avoid indexing
        # and the only difference is have an a and b not just an a, so check all of those
        # Then simply find a and b index, apply function and store in output

        if strides_aligned:
            # If strides are aligned, we can avoid indexing
            for i in prange(out.size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(out.size):
                out_index = np.empty(len(out_shape), dtype=np.int32)
                a_index = np.empty(len(a_shape), dtype=np.int32)
                b_index = np.empty(len(b_shape), dtype=np.int32)
                # Get the multi-dimensional index for the current flat index `i`
                to_index(i, out_shape, out_index)

                # Broadcast out_index to a_index and b_index based on their respective shapes
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                # Calculate flat positions in a, b, and output based on strides
                a_position = index_to_position(a_index, a_strides)
                b_position = index_to_position(b_index, b_strides)
                out_position = index_to_position(out_index, out_strides)

                # Apply the function element-wise on both input tensors and assign to output
                out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

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
        # a bit more complex than map and zip, but we can still parallelize it for each output
        # this means if do .reduce() without specifying a dimension it won't actually be parallel
        # but if we have a 100x100 tensor and do .sum(0) it will parallelize across up to 100 cpu cores!
        # the basic ideas is we find the index of the output, then one core will fully compute that output and store it
        # so in the 100x100 example, a core would take the first row, sum it and store it. each core does its own row

        reduce_size = a_shape[reduce_dim]  # size of the dimension we are reducing

        assert len(out_shape) < MAX_DIMS, "out_shape must be less than MAX_DIMS"

        for i in prange(
            len(out)
        ):  # Parallelized loop, if output is 1 dimension doesn't actually parallelize, but if reduce along a dimension it will
            out_index = np.empty(MAX_DIMS, dtype=np.int32)  # Index buffer for output
            to_index(i, out_shape, out_index)  # get the index
            base_position = index_to_position(
                out_index, out_strides
            )  # get the position of the value that will remain after reduction

            # Inner loop: use precomputed offsets for reduction
            o = index_to_position(out_index, out_strides)
            result = out[o]  # Identity for the reduction, 0 for sum, 1 for product
            out_index[reduce_dim] = 0  # initialize to 0
            temp_position = index_to_position(
                out_index, a_strides
            )  # the starting position of the values to reduce
            for s in range(reduce_size):
                result = fn(
                    result, float(a_storage[temp_position + s * a_strides[reduce_dim]])
                )  # apply the function to the result and the next value and keep applying it until it's done

            out[base_position] = result

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
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

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


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
    # these simply let us know the strides for the batch, i, j, k
    # the i is the row, j is the column, k is the inner dimension that is shared between a and b
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    a_i_stride = a_strides[1] if a_shape[1] > 1 else 0
    b_j_stride = b_strides[2] if b_shape[2] > 1 else 0
    a_k_stride = a_strides[2] if a_shape[2] > 1 else 0
    b_k_stride = b_strides[1] if b_shape[1] > 1 else 0
    # batch_max = max(a_shape[0], b_shape[0])

    # we have to do loops like this because we can't have index buffers
    # also parallelizing over the batch dimension doesn't speed up a large matrix multiply, only when have many batches
    # the basic idea here is we take a batch and assign it to a core. Then each core simply computes the matrix multiply for that batch
    for b in prange(out_shape[0]):
        # now we iterate over the other dimensions, pretend we have 2 2x2 matrices
        for i in range(out_shape[1]):  # basic matrix multiply loop
            for j in range(out_shape[2]):
                tmp = 0.0
                # we also calculate the indices so we don't have to do multiplies in the for loop which only allows 1 multiply
                a_index = b * a_batch_stride + i * a_i_stride
                b_index = b * b_batch_stride + j * b_j_stride
                for k in range(a_shape[-1]):
                    tmp += a_storage[a_index] * b_storage[b_index]
                    a_index += a_k_stride  # add the indices so we don't have to multiply, basically moves by the stride dimensions
                    b_index += b_k_stride

                out_index = (
                    b * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )  # now find the out index and write it out
                out[out_index] = tmp
                # now loop and find the next element. compute the full product and move on!!

    return None


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
