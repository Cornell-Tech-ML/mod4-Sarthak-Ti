import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # we will be testing the max function, both the forward and the backward, and we already get a tensor
    # for this purpose we will define a mini max and argmax function
    t.requires_grad_(True)

    # first the forward pass
    out = minitorch.nn.max(t, 0)
    for i in range(3):
        for j in range(4):
            assert_close(out[0, i, j], max([t[k, i, j] for k in range(2)]))

    # repeat for the other 2 dims
    out = minitorch.nn.max(t, 1)
    for i in range(2):
        for j in range(4):
            assert_close(out[i, 0, j], max([t[i, k, j] for k in range(3)]))

    out = minitorch.nn.max(t, 2)
    for i in range(2):
        for j in range(3):
            assert_close(out[i, j, 0], max([t[i, j, k] for k in range(4)]))

    # and now if we don't have a dimension
    out = minitorch.nn.max(t)
    assert_close(
        out[0], max([t[i, j, k] for i in range(2) for j in range(3) for k in range(4)])
    )

    # now the backward pass is a bit harder, but we can use the same idea of using the max function in python to get argmax
    # so we could normally use central_difference to check the gradient, but I don't think we should, because when there are multiple max values it's weird
    # so let's isntead just manually check the gradient. again check for all 4 conditions
    # we can manually loop over all the values and ensure that only 1 is non 0
    t.zero_grad_()
    out = minitorch.nn.max(t, 0).view(3, 4)
    # here we have an output that is shape 3,4. we can simply provide a random tensor as the d output values to manually easily verify it
    d_out = minitorch.rand((3, 4)) + 1  # just to ensure no values are 0 for this test
    out.backward(d_out)
    # now manually verify the values in t.grad
    for i in range(3):
        for j in range(4):
            vals = [t[k, i, j] for k in range(2)]
            max_val = max(vals)
            # now ensure that the gradient is d_out[i,j] if it's a max and the rest are 0
            # now find the indices that ar emax
            max_indices = [k for k in range(2) if t[k, i, j] == max_val]

            # now the hard part is to verify that only one of the values matches and exaclty only 1 value, no others
            # so we will loop over things, and we first check to see if it is a max value, if not we obviously know the gradient shoudl be 0 ther
            # if it's not, then we check to see if the gradient is the same as d_out[i,j], if it is we increment a counter to ensure only 1 value has a gradient
            # otherwise we ensure the gradient is 0. a relatively basic and straightforward test

            grad_assigned = 0  # this lets us ensure only 1 value is non 0!
            for k in range(2):
                if k in max_indices:
                    # now we know it is a max value
                    if t.grad[k, i, j] == d_out[i, j]:
                        grad_assigned += 1
                        assert_close(t.grad[k, i, j], d_out[i, j])
                    else:
                        assert_close(t.grad[k, i, j], 0)
                else:
                    assert_close(t.grad[k, i, j], 0)
            assert (
                grad_assigned == 1
            )  # we ensure that we found a max value, else that's a problem, no gradients were established

    # now we can repeat for the other 2 dimensions
    t.zero_grad_()
    out = minitorch.nn.max(
        t, 1
    ).view(
        2, 4
    )  # we need this view here. previously it was ok because broadcasting, but basically the d out and this need to be the same shape. here it wont' be without view
    d_out = minitorch.rand((2, 4)) + 1
    out.backward(d_out)
    for i in range(2):
        for j in range(4):
            vals = [t[i, k, j] for k in range(3)]
            max_val = max(vals)
            max_indices = [k for k in range(3) if t[i, k, j] == max_val]
            grad_assigned = 0
            for k in range(3):
                if k in max_indices:
                    if t.grad[i, k, j] == d_out[i, j]:
                        grad_assigned += 1
                        assert_close(t.grad[i, k, j], d_out[i, j])
                    else:
                        assert_close(t.grad[i, k, j], 0)
                else:
                    assert_close(t.grad[i, k, j], 0)
            assert grad_assigned == 1

    t.zero_grad_()
    out = minitorch.nn.max(t, 2).view(2, 3)
    d_out = minitorch.rand((2, 3)) + 1
    out.backward(d_out)
    for i in range(2):
        for j in range(3):
            vals = [t[i, j, k] for k in range(4)]
            max_val = max(vals)
            max_indices = [k for k in range(4) if t[i, j, k] == max_val]
            grad_assigned = 0
            for k in range(4):
                if k in max_indices:
                    if t.grad[i, j, k] == d_out[i, j]:
                        grad_assigned += 1
                        assert_close(t.grad[i, j, k], d_out[i, j])
                    else:
                        assert_close(t.grad[i, j, k], 0)
                else:
                    assert_close(t.grad[i, j, k], 0)
            assert grad_assigned == 1

    # and now if we don't have a dimension it's a little more complex, just more loops but the exact same process
    t.zero_grad_()
    out = minitorch.nn.max(t)  # don't need view as max automatically reduces to shape 1
    d_out = (
        minitorch.rand((1, 1)) + 1
    )  # we can't make it shape 1 with a tensor, so better to do 1x1
    d_out = d_out.view(1)  # now view it to shape 1
    out.backward(d_out)
    vals = [
        t[i, j, k] for i in range(2) for j in range(3) for k in range(4)
    ]  # only 1 list of vals
    max_val = max(vals)
    max_indices = [
        (i, j, k)
        for i in range(2)
        for j in range(3)
        for k in range(4)
        if t[i, j, k] == max_val
    ]  # and only 1 list of indices
    grad_assigned = 0
    for i in range(2):
        for j in range(3):
            for k in range(4):
                if (i, j, k) in max_indices:
                    if t.grad[i, j, k] == d_out[0]:
                        grad_assigned += 1
                        assert_close(t.grad[i, j, k], d_out[0])
                    else:
                        assert_close(t.grad[i, j, k], 0)
                else:
                    assert_close(t.grad[i, j, k], 0)

    # we have successfully tested the max function, both forward and backward, and we have verified the gradients manually, so we can assume if it passses these tests, it must work!


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
