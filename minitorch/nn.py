from typing import Tuple, Optional

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # this tells us we need no padding, it's set up to tile perfectly
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    # the function basically applies this tiling, just need to gather the data so we can apply pooling
    new_height = height // kh
    new_width = width // kw

    # reshape the tensor to have the new dimensions
    # but it's not so simple as a reshape. first have to group the kernel dimensions together

    # the first step is to make it contiguous so we can view
    input = input.contiguous()

    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)

    # now we have to reshape so that we can get the kernel dimensions together, this way we can easily combine them then later pool
    permuted = reshaped.permute(
        0, 1, 2, 4, 3, 5
    )  # (batch, channel, new_height, new_width, kh, kw)

    # now we reshape to combine the kernel dimensions together
    # now make it contiguous again so we can view
    permuted = permuted.contiguous()
    tiled = permuted.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D. Function argument and parameters taken from the minitorch github

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape

    # first tile
    tiled, new_height, new_width = tile(input, kernel)
    # print(tiled)
    # now do the average operation over that last dimension
    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)

# #all of these function arguments and parameters are taken from the minitorch github as well as the docstrings


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax


    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        # in the forward pass we first find the max value use
        maxed = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, maxed, dim)
        return maxed

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        # a bit complex, the basic idea is that we set the value of max to 1 and the rest to 0
        # to set the value of max to 1, we can
        # interestingly, we see that pytorch distributes the gradient evenly for all values that share when multiple are maxed
        # but we will maybe do that, maybe just choose one randomly? idk dude. Slides say to just pick one, so let's do that!!
        # remember we already are doing it only along an axis, so can think about it as a 1d or 2d thing but we know the axis it's reduced on
        # first we get the argmax
        input, maxed, dim = ctx.saved_tensors
        args = argmax(input, int(dim.item()))
        # now check to see if there's any extras, and add noise
        # note that by doing this noise approach, even if we had a bunch of zeros if we do .max and a row is all 0s, one random 0 will get a gradient of 1, the rest 0
        # so if average the gradient many times will see 2 peaks and the rest noise, because random!
        if args.sum(int(dim.item())) > 1:
            # first add some random noise
            args = args + rand(args.shape)
            # now we redo the argmax
            args = argmax(args, int(dim.item()))
        # now simply multiply by the gradient
        return args * grad_output, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Apply max reduction to a tensor along a dimension."""
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))
    # return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    # rather straightforward to implement, we have the functions, just call it
    # note we could make it numerically stable by subtracting the max value from each value, but we will not do that here because it's likely unnecessary
    # if we get an overflow error, we will add that in
    assert dim is not None
    exp = input.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
    ----
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
    -------
         log of softmax tensor

    """
    # here we want to use the logsumexp trick and do a log of the softmax
    # could literally do softmax(input, dim).log() but that isn't numerically stable
    # so basically what you do is use this principle, log(x) = log(2x*0.5) = log(2x) + log(0.5)
    # what we have is something like this: log(sum(exp(xi)))
    # First you factor out the max value of x, log(exp(xmax)*sum(exp(xi)*exp(-xmax)))
    # then simplify to get xmax + log(sum(exp(xi-xmax)))
    # the only difference here is we do it for tensors and apply it to every element along the axis of choice
    xmax = max(input, dim)
    diff = input - xmax
    return diff - diff.exp().sum(dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    batch, channel, height, width = input.shape
    # this is very similar to the average pooling, just with a different function
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with random positions dropped out

    """
    # we will do dropout, but also we include areas where we don't dropout
    # if we don't dropout, we will just return the same tensor
    if ignore:
        return input
    # otherwise we will dropout, here we randomly sample everything with a specific probability
    # could do like a bernoulli distribution, or another option is just create a random tensor, if it's over the rate, we drop it out
    rand_vals = rand(input.shape)
    return input * (rand_vals > rate)
