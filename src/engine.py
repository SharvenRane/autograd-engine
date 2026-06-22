"""A micrograd style reverse mode automatic differentiation engine.

Each Value wraps a single scalar. Operations build a directed acyclic graph
recording how outputs depend on inputs. Calling backward() on a scalar output
runs reverse mode autodiff, accumulating the partial derivative of that output
with respect to every Value in the graph into the .grad attribute.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Set, Tuple, Union

Number = Union[int, float]


class Value:
    """A scalar node in the autodiff graph."""

    def __init__(
        self,
        data: Number,
        _children: Tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        # Function that propagates this node's grad into its parents' grads.
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Value"] = set(_children)
        self._op: str = _op
        self.label: str = label

    # ----- core arithmetic -------------------------------------------------

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: Number) -> "Value":
        assert isinstance(other, (int, float)), "power must be a constant scalar"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ----- nonlinearities --------------------------------------------------

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ----- derived / convenience operators ---------------------------------

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: Number) -> "Value":
        return self + other

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other: Number) -> "Value":
        return (-self) + other

    def __rmul__(self, other: Number) -> "Value":
        return self * other

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other: Number) -> "Value":
        return Value(other) * self ** -1

    # ----- reverse mode autodiff -------------------------------------------

    def backward(self) -> None:
        """Run reverse mode autodiff from this node.

        Builds a topological order of the graph, seeds this node's grad to 1,
        then walks the order in reverse calling each node's local backward.
        """
        topo: list[Value] = []
        visited: Set[Value] = set()

        def build(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradients on this node and every ancestor in its graph."""
        visited: Set[Value] = set()

        def visit(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                node.grad = 0.0
                for child in node._prev:
                    visit(child)

        visit(self)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"


def topological_order(root: Value) -> Iterable[Value]:
    """Return the nodes of root's graph in dependency order (inputs first)."""
    topo: list[Value] = []
    visited: Set[Value] = set()

    def build(node: Value) -> None:
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

    build(root)
    return topo
