from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # first we add epsilong and subtract epsilon from our value at that arg
    plusvals = list(vals)
    minusvals = list(vals)
    plusvals[arg] += epsilon
    minusvals[arg] -= epsilon
    print(plusvals, minusvals)
    # then we get the value of the function at x plus epsilon and minus epsilon
    fp = f(*plusvals)
    fm = f(*minusvals)
    # print(fp, fm)
    # now return derivative using the definition
    return (fp - fm) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """A class that simply is a placeholder for the `Variable` class.
    It is used to type hint the `Variable` class before it is defined.
    """

    def accumulate_derivative(self, x: Any) -> None:
        """Adds the value to the derivative if it exists or sets it to that if it doesn't"""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable"""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for the inputs of this Scalar."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.
    Lets you know the order of every node that was called in the forward pass.
    Reverse order so we call backward in that order

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # Use DFS in my implementation
    # sorted_nodes = []
    # visited = set()
    # # we start at the rightmost one and go backwards, let's keep track of the depth using a depth_history dict
    # depth_history = {}  # this doubles as keeping track of the nodes we have visited

    # def dfs(node: Variable, depth: int = 0) -> None:
    #     if depth not in depth_history:
    #         depth_history[depth] = []
    #     if node.unique_id in visited or node.is_constant():
    #         return
    #     visited.add(node.unique_id)
    #     depth_history[depth].append(node)

    #     if not node.is_leaf():  # only if the node has a history so it's a parent node
    #         for input_node in node.parents:
    #             dfs(input_node, depth + 1)

    # dfs(variable)
    # # now go through the depth history and add the nodes in reverse order
    # for depth in sorted(depth_history.keys()):
    #     sorted_nodes.extend(depth_history[depth])

    # return list(sorted_nodes)

    # their implementation, not needed
    order = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    visit(parent)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    # my implementation below, but has a minor bug
    # sorted_nodes = topological_sort(variable)

    # # create dict of scalars and current derivatives
    # derivatives = {node.unique_id: 0 for node in sorted_nodes}
    # derivatives[variable.unique_id] = (
    #     deriv  # set the derivative of the variable to the derivative we want to backpropagate
    # )

    # # now loop over the nodes and update the derivataives
    # for node in sorted_nodes:
    #     if node.is_leaf():
    #         node.accumulate_derivative(
    #             derivatives[node.unique_id]
    #         )  # derivative should already be calculated now just set it
    #     else:
    #         local_grad = derivatives[node.unique_id]
    #         for parent, derivative in node.chain_rule(local_grad):
    #             derivatives[parent.unique_id] += derivative

    # their implementation, works well
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0)
                derivatives[v.unique_id] += d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values"""
        return self.saved_values
