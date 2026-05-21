## `jones-tensor`

`jones-tensor` is a Python program to evaluate the Jones polynomial of a link at a given set of points in the complex plane. 

It works by defining a tensor network where the structure encodes the structure of the link, and fully contracting the network yields the value of the polynomial. The construction of the network is originally due to Kauffman [[Knots and Physics, p.125]](https://www.worldscientific.com/worldscibooks/10.1142/4256), but the version presented here uses modern computational methods, like hyper-optimized contraction pathfinding from [`cotengra`](https://cotengra.readthedocs.io), and GPU acceleration with [`autoray`](https://autoray.readthedocs.io).

### Setup

You can use `pip` to install directly from git:
```
pip install git+https://github.com/tlaakkonen/jones-tensor.git
```

### Usage

See the `examples/` directory for specifics, including GPU acceleration, and an example of a large link with over 300 crossings. `jones-tensor` supports defining links from a planar diagram code, or a braid representative.

As a basic example, let's evaluate the Jones polynomial $V_L(q)$ of the trefoil at $q = 2$:
```python
>>> import jones_tensor
>>> trefoil = jones_tensor.MorseLink.from_braid([-1, -1, -1])
>>> jones_tensor.evaluate_jones(2, trefoil)
(0.5625+0j)
```

### Tests

You can run `tests/test_db_repro.py` to check that `jones-tensor` agrees with the tabulated polynomials in the Knot Atlas, KnotInfo, and LinkInfo databases (this might take a few minutes).

### Background

This method is based on the $R$-matrix formulation of the bracket polynomial, and other similar implementations exist (for example, the [`REngine` module](https://katlas.org/wiki/R-Matrix_Invariants) of the `KnotTheory` package). The main difference between those and this package is that, because we only evaluate the polynomial at a fixed point rather than computing the whole polynomial, we can use existing sophisticated tensor network techniques, such as GPU acceleration, and better pathfinding (rather than contracting 'in-order' as in REngine). This lets us tackle larger braids, up to several hundred crossings. This formulation can also be generalized to other Reshtikhin-Tuarev invariants, including the coloured Jones polynomial, but this is not implemented here yet (see the [`QuantumGroups` package](https://katlas.org/wiki/Quantum_knot_invariants) for an implementation).