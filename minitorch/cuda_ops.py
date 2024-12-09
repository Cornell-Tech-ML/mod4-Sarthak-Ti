# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to create a jitted version of a function that runs on the GPU."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to create a jitted version of a function that runs on the GPU."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Given 2 matrices that are 2 or 3d, return their matrix product
        Computes the matrix product of two tensors by first ensuring they both have batch size of at least 1
        Handles broadcasting by calling the `shape_broadcast` function
        Operation performed by the `tensor_matrix_multiply` kernel using a GPU
        This ensure that there are enough blocks to handle the output across all dimensions

        Args:
        ----
            a (Tensor): First tensor.
            b (Tensor): Second tensor.

        Returns:
        -------
            Tensor: Matrix product of `a` and `b`.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        # this is easy as it can be fully parallel!
        # first find the broadcasted index
        # for k in range(MAX_DIMS):
        #     out_index[k] = 0
        #     in_index[k] = 0

        # we don't care about blocks or anything, each thread can do it's own thing, just take one value, compute something, and put it in the output
        if (
            i < out_size
        ):  # guard to make sure thread corresponds to a value in the matrix
            to_index(i, out_shape, out_index)  # get the index of the output
            broadcast_index(
                out_index, out_shape, in_shape, in_index
            )  # broadcast the index to the input
            in_position = index_to_position(
                in_index, in_strides
            )  # get the position of the input and output with broadcasted index
            out_position = index_to_position(out_index, out_strides)

            out[out_position] = fn(
                in_storage[in_position]
            )  # simply apply the function to the input and put it in the output

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # very similar to the map function, but take values from array a and b, broadcast to output then apply the function
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)

            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # first we do the sum in each block, so basically do the tree in each block, then after the blocks, we sum along the block dimension
    # first move data to the cache for the block to speed thigns up
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0
    cuda.syncthreads()  # sync threads to make sure all threads have loaded the data and cache is fully loaded
    stride = cuda.blockDim.x // 2  # start with half the block size
    while stride > 0:  # just a while loop, could do a for loop
        if pos < stride:  # only do the computation if the thread is within the stride
            cache[pos] += cache[
                pos + stride
            ]  # think of this as a generalization of what I did in gpu puzzle 10!
        stride = (
            stride // 2
        )  # nowo divide by 2 so that we apply the function on the partially reduced array
        cuda.syncthreads()  # also have to sync so that we know have the partial reduction
    if pos == 0:  # only write it out if it's the first thread
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Reduction practice function, takes in a tensor that is larger than 32, and returns block sums."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData(
        [0.0 for i in range(2)], (2,)
    )  # only do the first 2, if bigger than 64, it won't work!
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # so the hard part will be reduction if the dim is None or not, then it's a bitharder. Use strategy in sum practice to do the reduction
        # first we determine along which dimension we are reducing and then assign blocks and threads accordingly
        # we are also given a reduce value which is nice so we know what to start with whether it's 0 or 1
        # we also don't compute i because we need to align the block to reduce_dim and put as many blocks along that as we need

        # so the first step is find the reduction dimension and how many elements it has

        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        to_index(out_pos, out_shape, out_index)

        # Copy the output index to the input index, again to speed things up and not read from global memory
        for i in range(len(out_shape)):
            a_index[i] = out_index[i]

        # Get the size of the dimension to reduce
        reduce_dim_size = a_shape[reduce_dim]
        num_threads = cuda.blockDim.x

        # Initialize accumulator with the identity value. 0 for sum, but 1 for product
        acc = reduce_value

        # Each thread processes multiple elements along the reduce dimension
        # basically, we loop through the reduce dimension and add the values to the accumulator
        # so the 1 thread fully computes it's part of the reduction
        # so if we have 4 threads, thread 0 does 0,4,8,12, thread 1 does 1,5,9,13, etc.
        # the first partial reduction, then do the rest of the reduction (if necessary) like we did in the practice
        for idx in range(pos, reduce_dim_size, num_threads):
            a_index[reduce_dim] = idx
            a_pos = index_to_position(a_index, a_strides)
            val = a_storage[a_pos]
            acc = fn(acc, val)

        # Store the partial results in shared memory
        cache[pos] = acc

        # Synchronize threads within the block
        cuda.syncthreads()

        # Now finalize reduction using shared memory
        stride = num_threads // 2
        while stride > 0:
            cuda.syncthreads()
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            stride = stride // 2

        # The first thread writes the result to the output
        if pos == 0:
            out_pos = index_to_position(out_index, out_strides)
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # this is easy because size is < 32 which means we can do it in one block

    # however, getting it to read only each cell once is tricky... but we can read it into shared memory!
    # have to align it so that the block reads in the proper rows and columns! In this case can just read the whole thing in
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i < size and j < size:  # guard against out of bounds
        shared_a[i, j] = a[i * size + j]  # add it to shared memory
        shared_b[i, j] = b[i * size + j]
        cuda.syncthreads()

        temp = 0  # now we can simply compute the full dot product, each thread computes one value
        for k in range(size):
            temp += shared_a[i, k] * shared_b[k, j]
        out[i * size + j] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Matrix multiply practice function, takes in two 2d tensors which are smaller than 32x32."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # note that this approach requires blocks as specified above
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # out_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64) #not needed

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # The basic idea with this code is you are given a block that corresponds to a portion of the output matrix, and it's the same grid size as the output so you can map it 1-1
    # the key idea is you copy parts of a and b into the block, then loop through the parts of B that are necessary to compute those elements of C
    # for any C[i,j] you need to loop through the k values of A[i,k] and B[k,j] to compute the dot product
    # it's further complicated by the use of blocks, but what you can do is think about it like this, let's say you have a 64 x 64 matrix A and B.
    # if the block is 32x32, you can think of breaking down your matrix into 2x2 blocks, and then each block can compute the 32x32 matrix product
    # this matrix product is identical to what we did in the practice problem, so you can use that as a reference
    # Also a key idea is if the shapes aren't divisible by 32, you pad the extra values as 0 (rather obvious)
    # so the key idea is you copy a block of a and b into memory, compute the dot product of each of the vectors with each other (each thread comptues 1 dot product)
    # then you move on to the next block and keep looping until you have fully computed every element for that block of c
    # I drew this out to fully understand the problem, but that's th ehigh level idea

    # also I should note this works because we assign a single block to every portion of c and there's no overlap, which allows it to be fast!

    # the i represents the row in c which corresponds to the row in a, the j represents the column in c which corresponds to the column in b
    # but we use pi and pj as the way to move along the row an dcol of a and b respectively

    temp = 0  # this is the value for the thread. Since the dot product can be broken into many partial products, ekep adding to this variable in local memory

    for k in range(0, a_shape[2], BLOCK_DIM):
        # this is the k loop where we have to move the block shared memory down along the b dimension and right along the a dimension to tell us the true values of c by combining partial dot products
        # we incrememt k by blockdim, because each thread covers each value in the block, so then we move to the right or down by block dim to make sure each thread reads each blocks values once
        # first we need to grab the values from global memory and move it to shared memory for quick computation
        # we first need to check if the thread value is within the bounds of the a matrix, since we move only to rhte right in the a matrix (because dot product is col of b * row of a, so move right in a down in b)
        # so we check the a dimension if the thread is within the bounds in the row, but for the column we move to the right, so have to add this k value to it
        if i < a_shape[1] and k + pj < a_shape[2]:  # check to see if it's in bounds
            # if it is in bounds, we have to find the corresponding value from the thread in that block. Basically
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + (k + pj) * a_strides[2]
            ]  # sometimes it's not contiguous, so need to even multiply the last element by the stride
        else:
            a_shared[pi, pj] = (
                0  # else we add 0. Can think of padding a with zeros such that it's divisible by 32 in rows and columns
            )
        # now we do the same for b, but we move down in b and j represents the column in b, so the k and pi is rows. Otherwise it's the exact same
        if k + pi < b_shape[1] and j < b_shape[2]:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + (k + pi) * b_strides[1] + j * b_strides[2]
            ]
        else:
            b_shared[pi, pj] = 0

        # now we sync threads to make sure that the shared memory is fully loaded before we start the computation, as values for each thread can rely on the values of other threads
        cuda.syncthreads()

        # now that we have leaded the memory this is exaclty the same computation as we did in the practice matrix multiply. This is just part of the product tho
        for kk in range(BLOCK_DIM):
            temp += (
                a_shared[pi, kk] * b_shared[kk, pj]
            )  # we simply accumulate the value in temp
            # we can do this because for thread i,j we compute c[i,j] = sum(a[i,k] * b[k,j]) for all k
            # we simply compute part of the sum each time and so we can just keep adding to the temp variable
        cuda.syncthreads()
        # sync again after it's done because after this step we modify the shared memory again, and don't want to start modifying before all threads are done
        # now loop thorugh to get the full value for a single output value in c for each thread!

    # now we have the data, it's just writing each thread's value to the output in global memory
    out_index = (
        batch * out_strides[0] + i * out_strides[1] + j
    )  # get the index of the output, out will always be contiguous, but could multiply j by out_strides[2] if it's not
    if (
        i < out_shape[1] and j < out_shape[2]
    ):  # guard to make sure it's only writing if it's in bounds
        out[out_index] = temp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
