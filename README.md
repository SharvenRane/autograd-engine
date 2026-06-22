# autograd-engine

A small reverse mode automatic differentiation engine that works on scalar values, in the spirit of micrograd. Every number you compute with is wrapped in a `Value` object. As you combine values with arithmetic and nonlinearities, the engine quietly records a graph of how each result depends on its inputs. When you call `backward()` on a final scalar, it walks that graph in reverse and fills in the derivative of the output with respect to every value that fed into it.

The goal here is to make the mechanics of backpropagation concrete. There is no array library doing the heavy lifting and no hidden magic. Each operation knows two things: how to compute its output, and how to push gradient from its output back onto its inputs.

## What it supports

The `Value` type implements:

* Addition, subtraction, multiplication, and division
* Raising to a constant power
* `tanh`, `relu`, `exp`, and `log`
* `backward()` for reverse mode gradient computation
* `zero_grad()` to reset gradients across the whole graph

Operations accept plain Python numbers as well as other `Value` objects, so you can write expressions like `3 * x ** 2 + 2 * x + 1` and read them the way you would on paper.

## How it works

Each operation creates a new output `Value` that remembers its parents and stores a local `_backward` closure. That closure knows the chain rule contribution for the specific op. For example a multiply node `z = x * y` stores the rule that the gradient flowing into `z` lands on `x` scaled by `y.data`, and on `y` scaled by `x.data`.

Calling `backward()` does three things. First it builds a topological order of the graph so that every node comes after all of its inputs. Then it seeds the gradient of the output node to one. Finally it visits the nodes in reverse topological order, running each local rule, which accumulates partial derivatives into the `.grad` field of every node. Accumulation matters: when a single value feeds into several places, its gradient is the sum of every path, and the `+=` in each rule handles that correctly.

## Usage

```python
from src.engine import Value

x = Value(3.0)
f = 3 * x ** 2 + 2 * x + 1   # f = 3x^2 + 2x + 1
f.backward()

print(f.data)   # 34.0
print(x.grad)   # 32.0, which is 6x + 2 at x = 3
```

A tiny single neuron looks like this:

```python
from src.engine import Value

x1, x2 = Value(0.5), Value(-1.5)
w1, w2 = Value(1.1), Value(-0.7)
b = Value(0.05)

out = (x1 * w1 + x2 * w2 + b).tanh()
out.backward()

print(w1.grad, w2.grad, b.grad)  # gradients for a gradient descent step
```

## Tests

The test suite checks three kinds of things. Forward values are compared against ordinary arithmetic. Gradients are compared against derivatives worked out by hand for cases like products, polynomials, reused variables, `tanh`, `relu`, and division. The remaining tests build mixed expressions and a small neuron, then confirm that both the forward values and the gradients match `torch.autograd` to within floating point tolerance. PyTorch acts as an independent reference, so agreement is strong evidence the engine is correct.

Run them with the project interpreter:

```
python -m pytest tests/ -q
```

On the local run all 16 tests passed.

## Layout

```
autograd-engine/
  src/
    __init__.py
    engine.py        the Value type and the autodiff logic
  tests/
    test_engine.py   forward, hand computed, and torch cross checks
  requirements.txt
  README.md
```

## Requirements

Python 3.10 or newer, plus `pytest` and `torch`. PyTorch is used only by the tests as a reference; the engine itself depends on nothing beyond the standard library.
