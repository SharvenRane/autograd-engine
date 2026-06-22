"""Behavior tests for the scalar autodiff engine.

The tests fall into three groups:

1. Forward values match plain arithmetic.
2. Backward gradients match hand computed derivatives.
3. Both forward values and gradients match torch.autograd on the same
   expressions, which is an independent reference implementation.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import Value, topological_order  # noqa: E402


# --------------------------------------------------------------------------
# forward value checks
# --------------------------------------------------------------------------

def test_forward_add_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a
    assert c.data == pytest.approx(8.0)


def test_forward_pow_and_div():
    a = Value(4.0)
    assert (a ** 0.5).data == pytest.approx(2.0)
    assert (a / 2.0).data == pytest.approx(2.0)


def test_relu_and_tanh_forward():
    assert Value(-3.0).relu().data == 0.0
    assert Value(2.5).relu().data == pytest.approx(2.5)
    assert Value(0.7).tanh().data == pytest.approx(math.tanh(0.7))


# --------------------------------------------------------------------------
# hand computed gradient checks
# --------------------------------------------------------------------------

def test_grad_simple_product():
    # f = a * b ; df/da = b, df/db = a
    a = Value(-2.0)
    b = Value(3.0)
    f = a * b
    f.backward()
    assert a.grad == pytest.approx(3.0)
    assert b.grad == pytest.approx(-2.0)


def test_grad_polynomial():
    # f = 3 * x**2 + 2 * x + 1 ; df/dx = 6x + 2
    x = Value(5.0)
    f = 3 * x ** 2 + 2 * x + 1
    f.backward()
    assert f.data == pytest.approx(3 * 25 + 2 * 5 + 1)
    assert x.grad == pytest.approx(6 * 5 + 2)


def test_grad_reused_variable():
    # f = x * x + x ; a variable feeding several ops must accumulate grad.
    # df/dx = 2x + 1
    x = Value(3.0)
    f = x * x + x
    f.backward()
    assert x.grad == pytest.approx(2 * 3 + 1)


def test_grad_tanh():
    # d/dx tanh(x) = 1 - tanh(x)**2
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    assert x.grad == pytest.approx(1.0 - math.tanh(0.5) ** 2)


def test_grad_relu_branches():
    pos = Value(2.0)
    pos.relu().backward()
    assert pos.grad == pytest.approx(1.0)

    neg = Value(-2.0)
    neg.relu().backward()
    assert neg.grad == pytest.approx(0.0)


def test_grad_division():
    # f = a / b ; df/da = 1/b, df/db = -a / b**2
    a = Value(6.0)
    b = Value(3.0)
    f = a / b
    f.backward()
    assert f.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(1.0 / 3.0)
    assert b.grad == pytest.approx(-6.0 / 9.0)


def test_grad_exp_log():
    # f = log(exp(x)) = x ; df/dx = 1
    x = Value(1.3)
    f = x.exp().log()
    f.backward()
    assert f.data == pytest.approx(1.3)
    assert x.grad == pytest.approx(1.0)


# --------------------------------------------------------------------------
# topological order sanity
# --------------------------------------------------------------------------

def test_topological_order_inputs_before_outputs():
    a = Value(1.0)
    b = Value(2.0)
    out = a * b + a
    order = list(topological_order(out))
    assert order[-1] is out
    assert order.index(a) < order.index(out)
    assert order.index(b) < order.index(out)


def test_zero_grad_resets_graph():
    x = Value(2.0)
    y = x * x
    y.backward()
    assert x.grad != 0.0
    y.zero_grad()
    assert x.grad == 0.0
    assert y.grad == 0.0


# --------------------------------------------------------------------------
# cross check against torch.autograd
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _torch_scalar(v):
    t = torch.tensor(float(v), dtype=torch.double, requires_grad=True)
    return t


def test_matches_torch_mixed_expression():
    # f = (a * b + b**3) * relu(c) - (a / b)
    a_, b_, c_ = -4.0, 2.0, 3.0

    a = Value(a_)
    b = Value(b_)
    c = Value(c_)
    f = (a * b + b ** 3) * c.relu() - (a / b)
    f.backward()

    ta = _torch_scalar(a_)
    tb = _torch_scalar(b_)
    tc = _torch_scalar(c_)
    tf = (ta * tb + tb ** 3) * tc.relu() - (ta / tb)
    tf.backward()

    assert f.data == pytest.approx(tf.item(), rel=1e-9, abs=1e-9)
    assert a.grad == pytest.approx(ta.grad.item(), rel=1e-9, abs=1e-9)
    assert b.grad == pytest.approx(tb.grad.item(), rel=1e-9, abs=1e-9)
    assert c.grad == pytest.approx(tc.grad.item(), rel=1e-9, abs=1e-9)


def test_matches_torch_tanh_chain():
    x_, w_, bias_ = 0.7, -1.2, 0.3

    x = Value(x_)
    w = Value(w_)
    bias = Value(bias_)
    out = (x * w + bias).tanh()
    out.backward()

    tx = _torch_scalar(x_)
    tw = _torch_scalar(w_)
    tb = _torch_scalar(bias_)
    tout = torch.tanh(tx * tw + tb)
    tout.backward()

    assert out.data == pytest.approx(tout.item(), rel=1e-9, abs=1e-9)
    assert x.grad == pytest.approx(tx.grad.item(), rel=1e-9, abs=1e-9)
    assert w.grad == pytest.approx(tw.grad.item(), rel=1e-9, abs=1e-9)
    assert bias.grad == pytest.approx(tb.grad.item(), rel=1e-9, abs=1e-9)


def test_matches_torch_deep_chain():
    # A longer chain exercising add, mul, pow, div, exp, tanh together.
    x_ = 0.9
    x = Value(x_)
    y = ((x * 2.0 + 1.0) ** 2) / (x.exp() + 1.0)
    y = y.tanh()
    y.backward()

    tx = _torch_scalar(x_)
    ty = ((tx * 2.0 + 1.0) ** 2) / (torch.exp(tx) + 1.0)
    ty = torch.tanh(ty)
    ty.backward()

    assert y.data == pytest.approx(ty.item(), rel=1e-9, abs=1e-9)
    assert x.grad == pytest.approx(tx.grad.item(), rel=1e-9, abs=1e-9)


def test_matches_torch_small_mlp_neuron():
    # A single neuron with two inputs followed by a tanh, gradients on all params.
    xs = [0.5, -1.5]
    ws = [1.1, -0.7]
    bias_ = 0.05

    vx = [Value(v) for v in xs]
    vw = [Value(v) for v in ws]
    vb = Value(bias_)
    act = vx[0] * vw[0] + vx[1] * vw[1] + vb
    out = act.tanh()
    out.backward()

    tx = [_torch_scalar(v) for v in xs]
    tw = [_torch_scalar(v) for v in ws]
    tb = _torch_scalar(bias_)
    tact = tx[0] * tw[0] + tx[1] * tw[1] + tb
    tout = torch.tanh(tact)
    tout.backward()

    assert out.data == pytest.approx(tout.item(), rel=1e-9, abs=1e-9)
    for vwi, twi in zip(vw, tw):
        assert vwi.grad == pytest.approx(twi.grad.item(), rel=1e-9, abs=1e-9)
    for vxi, txi in zip(vx, tx):
        assert vxi.grad == pytest.approx(txi.grad.item(), rel=1e-9, abs=1e-9)
    assert vb.grad == pytest.approx(tb.grad.item(), rel=1e-9, abs=1e-9)
