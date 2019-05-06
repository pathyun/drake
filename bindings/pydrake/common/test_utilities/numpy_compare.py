"""
Provides consistent utilities for comparing NumPy matrices and scalars.

Prefer comparisons in the following order:

 - Methods from this module.
 - Methods from `np.testing.*`, if the dtypes are guaranteed to be NumPy
   builtins and the API definitely won't support other scalar types.
"""

# TODO(eric.cousineau): Make custom assert-vectorize which will output
# coordinates and stuff.

from collections import namedtuple
from itertools import product

import numpy as np


class _UnwantedEquality(AssertionError):
    pass


class _Registry(object):
    # Scalar comparator.
    # `assert_eq` will be vectorized; it should raise an assertion error upon
    # first inequality.
    # `assert_ne` will stay as scalar; this should raise `_UnwantedEquality` to
    # make intent explicit.
    # TODO(eric.cousineau): Add `assert_near` when it's necessary.
    AssertComparator = namedtuple(
        'AssertComparator', ['assert_eq', 'assert_ne'])

    def __init__(self):
        self._comparators = {}
        self._to_float = {}

    def register_comparator(self, cls_a, cls_b, assert_eq, assert_ne=None):
        key = (cls_a, cls_b)
        assert key not in self._comparators, key
        assert_eq = np.vectorize(assert_eq)
        self._comparators[key] = self.AssertComparator(assert_eq, assert_ne)

    def get_comparator_from_arrays(self, a, b):
        # Ensure all types are homogeneous.
        a_type, = {type(np.asarray(x).item()) for x in a.flat}
        b_type, = {type(np.asarray(x).item()) for x in b.flat}
        key = (a_type, b_type)
        return self._comparators[key]

    def register_to_float(self, cls, func):
        assert cls not in self._to_float, cls
        self._to_float[cls] = func

    def get_to_float(self, cls):
        return self._to_float[cls]


@np.vectorize
def to_float(x):
    """Converts scalar or array to floats."""
    x = np.asarray(x)
    if x.dtype == object:
        x = x.item()
        cls = type(x)
        return _registry.get_to_float(cls)(x)
    else:
        return np.float64(x)


def assert_equal(a, b):
    """Compare scalars or arrays directly, requiring equality."""
    a, b = map(np.asarray, (a, b))
    if a.size == 0 and b.size == 0:
        return
    if a.dtype != object and b.dtype != object:
        np.testing.assert_equal(a, b)
    else:
        _registry.get_comparator_from_arrays(a, b).assert_eq(a, b)


def _raw_ne(a, b):
    if a == b:
        raise _UnwantedEquality(str((a, b)))


def assert_not_equal(a, b):
    """Compare scalars or arrays directly, requiring inequality."""
    a, b = map(np.asarray, (a, b))
    assert not (a.size == 0 and b.size == 0)
    if a.dtype != object and b.dtype != object:
        assert_ne = _raw_ne
    else:
        assert_ne = _registry.get_comparator_from_arrays(a, b).assert_ne
    # For this to fail, all items must have failed.
    br = np.broadcast(a, b)
    errs = []
    for ai, bi in br:
        e = None
        try:
            assert_ne(ai, bi)
        except _UnwantedEquality as e:
            errs.append(str(e))
    all_equal = len(errs) == br.size
    if all_equal:
        raise AssertionError("Unwanted equality: {}".format(errs))


def _str_eq(a, b):
    # b is a string, a is to be converted.
    a = str(a)
    assert a == b, (a, b)


def _str_ne(a, b):
    # b is a string, a is to be converted.
    a = str(a)
    if a == b:
        raise _UnwantedEquality(str((a, b)))


def _register_autodiff():
    from pydrake.autodiffutils import AutoDiffXd

    def autodiff_eq(a, b):
        assert a.value() == b.value(), (a.value(), b.value())
        np.testing.assert_equal(a.derivatives(), b.derivatives())

    def autodiff_ne(a, b):
        if (a.value() == b.value() and
                (a.derivatives() == b.derivatives()).all()):
            raise _UnwantedEquality(str(a.value(), b.derivatives()))

    _registry.register_to_float(AutoDiffXd, AutoDiffXd.value)
    _registry.register_comparator(
        AutoDiffXd, AutoDiffXd, autodiff_eq, autodiff_ne)


def _register_symbolic():
    from pydrake.symbolic import (
        Expression, Formula, Monomial, Polynomial, Variable)

    def sym_struct_eq(a, b):
        assert a.EqualTo(b), (a, b)

    def sym_struct_ne(a, b):
        assert not a.EqualTo(b), (a, b)

    _registry.register_to_float(Expression, Expression.Evaluate)
    _registry.register_comparator(Formula, str, _str_eq, _str_ne)
    lhs_types = [Variable, Expression, Polynomial, Monomial]
    rhs_types = lhs_types + [float]
    for lhs_type in lhs_types:
        _registry.register_comparator(lhs_type, str, _str_eq, _str_ne)
    for lhs_type, rhs_type in product(lhs_types, rhs_types):
        _registry.register_comparator(
            lhs_type, rhs_type, sym_struct_eq, sym_struct_ne)


# Globals.
_registry = _Registry()
_register_autodiff()
_register_symbolic()
