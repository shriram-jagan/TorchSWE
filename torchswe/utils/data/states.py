#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Data models.
"""
# imports related to type hinting
from __future__ import annotations as _annotations  # allows us not using quotation marks for hints
from typing import TYPE_CHECKING as _TYPE_CHECKING  # indicates if we have type checking right now
if _TYPE_CHECKING:  # if we are having type checking, then we import corresponding classes/types
    from torchswe.nplike import ndarray
    from torchswe.utils.config import Config
    from torchswe.utils.data.grid import Domain

# pylint: disable=wrong-import-position, ungrouped-imports
from logging import getLogger as _getLogger
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

from pydantic import validator as _validator
from pydantic import root_validator as _root_validator
from torchswe import nplike as _nplike
from torchswe import _dummy_function
from torchswe.utils.config import BaseConfig as _BaseConfig
from torchswe.utils.io import read_block as _read_block
from torchswe.utils.misc import DummyDict as _DummyDict
from torchswe.utils.misc import interpolate as _interpolate
from torchswe.utils.data.grid import Domain as _Domain
from torchswe.utils.data.grid import get_domain as _get_domain


_logger = _getLogger("torchswe.utils.data.states")


def _pydantic_val_nan_inf(val, field):
    """Validates if any elements are NaN or inf."""
    assert not _nplike.any(_nplike.isnan(val)), f"Got NaN in {field.name}"
    assert not _nplike.any(_nplike.isinf(val)), f"Got Inf in {field.name}"
    return val


def _shape_val_factory(shift: _Union[_Tuple[int, int], int]):
    """A function factory creating a function to validate shapes of arrays."""

    def _core_func(val, values):
        """A function to validate the shape."""
        try:
            if isinstance(shift, int):
                target = (values["n"]+shift,)
            else:
                target = (values["ny"]+shift[0], values["nx"]+shift[1])
        except KeyError as err:
            raise AssertionError("Validation failed due to other validation failures.") from err

        assert val.shape == target, f"Shape mismatch. Should be {target}, got {val.shape}"
        return val

    return _core_func


class FaceOneSideModel(_BaseConfig):
    """Data model holding quantities on one side of cell faces normal to one direction.

    Attributes
    ----------
    q : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        The fluid elevation (i.e., h + b), h * u, and h * v.
    p : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        The fluid depth, u-velocity, and depth-v-velocity.
    a : nplike.ndarray of shape (ny+1, nx) or (3, ny, nx+1)
        The local speed.
    f : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An array holding discontinuous fluxes.
    """

    q: _nplike.ndarray
    p: _nplike.ndarray
    a: _nplike.ndarray
    f: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("q", "p", "a", "f", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate the consistency of the arrays shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            q, p, a, f = values["q"], values["p"], values["a"], values["f"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        n1, n2 = a.shape
        assert q.shape == (3, n1, n2), f"q shape mismatch. Should be {(3, n1, n2)}. Got {q.shape}."
        assert p.shape == (3, n1, n2), f"p shape mismatch. Should be {(3, n1, n2)}. Got {p.shape}."
        assert f.shape == (3, n1, n2), f"f shape mismatch. Should be {(3, n1, n2)}. Got {f.shape}."
        assert q.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {q.dtype}."
        assert p.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {p.dtype}."
        assert f.dtype == a.dtype, f"dtype mismatch. Should be {a.dtype}. Got {f.dtype}."
        return values


class FaceTwoSideModel(_BaseConfig):
    """Date model holding quantities on both sides of cell faces normal to one direction.

    Attributes
    ----------
    plus, minus : FaceOneSideModel
        Objects holding data on one side of each face.
    cf : nplike.ndarray of shape (3, ny+1, nx) or (3, ny, nx+1)
        An object holding common flux (i.e., continuous or numerical flux)
    """

    plus: FaceOneSideModel
    minus: FaceOneSideModel
    cf: _nplike.ndarray

    # validator
    _val_valid_numbers = _validator("cf", allow_reuse=True)(_pydantic_val_nan_inf)

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            plus, minus, cf = values["plus"], values["minus"], values["cf"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert plus.q.shape == minus.q.shape, f"Shape mismatch: {plus.q.shape} and {minus.q.shape}."
        assert plus.q.dtype == minus.q.dtype, f"dtype mismatch: {plus.q.dtype} and {minus.q.dtype}."
        assert plus.q.shape == cf.shape, f"Shape mismatch: {plus.q.shape} and {cf.shape}."
        assert plus.q.dtype == cf.dtype, f"dtype mismatch: {plus.q.dtype} and {cf.dtype}."
        return values


class FaceQuantityModel(_BaseConfig):
    """Data model holding quantities on both sides of cell faces in both x and y directions.

    Attributes
    ----------
    x, y : FaceTwoSideModel
        Objects holding data on faces facing x and y directions.
    """

    x: FaceTwoSideModel
    y: FaceTwoSideModel

    @_root_validator(pre=False, skip_on_failure=True)
    def _val_arrays(cls, values):  # pylint: disable=no-self-argument, no-self-use
        """Validate shapes and dtypes."""
        # pylint: disable=invalid-name

        try:
            x, y = values["x"], values["y"]
        except KeyError as err:
            raise AssertionError("Fix other fields first.") from err

        assert (x.plus.a.shape[1] - y.plus.a.shape[1]) == 1, "Incorrect nx size."
        assert (y.plus.a.shape[0] - x.plus.a.shape[0]) == 1, "Incorrect ny size."
        assert x.plus.a.dtype == y.plus.a.dtype, "Mismatched dtype."
        return values


class States(_BaseConfig):
    """A jumbo data model of all arrays on a mesh patch.

    A brief overview of the structure in this jumbo model (ignoring scalars):
    State: {
        q: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        p: ndarray                                          # shape: (3, ny+2*ngh, nx+2*ngh)
        s: ndarray                                          # shape: (3, ny, nx)
        ss: ndarray                                         # shape: (3, ny, nx)
        face: {
            x: {
                plus: {
                    q: ndarray                              # shape: (3, ny, nx+1)
                    p: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    f: ndarray                              # shape: (3, ny, nx+1)
                },
                minus: {
                    q: ndarray                              # shape: (3, ny, nx+1)
                    p: ndarray                              # shape: (3, ny, nx+1)
                    a: ndarray                              # shape: (ny, nx+1)
                    f: ndarray                              # shape: (3, ny, nx+1)
                },
                cf: ndarray                                  # shape: (3, ny, nx+1)
            },
            y: {                                            # shape: (ny+1, nx)
                plus: {
                    q: ndarray                              # shape: (3, ny+1, nx)
                    p: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    f: ndarray                              # shape: (3, ny+1, nx)
                },
                minus: {
                    q: ndarray                              # shape: (3, ny+1, nx)
                    p: ndarray                              # shape: (3, ny+1, nx)
                    a: ndarray                              # shape: (ny+1, nx)
                    f: ndarray                              # shape: (3, ny+1, nx)
                },
                cf: ndarray                                  # shape: (3, ny+1, nx)
            }
        },
        slpx: ndarray                                       # shape: (3, ny, nx+2)
        slpy: ndarray                                       # shape: (3, ny+2, nx)
    }

    Attributes
    ----------
    domain : torchswe.utils.data.Domain
        The domain associated to this state object.
    q : nplike.ndarray of shape (3, ny+2*nhalo, nx+2*nhalo)
        The conservative quantities defined at cell centers.
    p : nplike.ndarray of shape (3, ny+2*nhalo, nx+2*nhalo)
        The non-conservative quantities defined at cell centers.
    s : nplike.ndarray of shape (3, ny, nx)
        The explicit right-hand-side terms when during time integration. Defined at cell centers.
    ss : nplike.ndarray of shape (3, ny, nx)
        The stiff right-hand-side term that require semi-implicit handling. Defined at cell centers.
    face : torchswe.utils.data.FaceQuantityModel
        Holding quantites defined at cell faces, including continuous and discontinuous ones.
    """

    # associated domain
    domain: _Domain

    # quantities defined at cell centers and faces
    q: _nplike.ndarray
    p: _nplike.ndarray
    s: _nplike.ndarray
    ss: _Optional[_nplike.ndarray]
    face: FaceQuantityModel

    # TODO: Remove this nasty piece of code that is being added to avoid using futures
    gravity_x: _nplike.ndarray
    gravity_y: _nplike.ndarray

    gravity2_x: _nplike.ndarray
    gravity2_y: _nplike.ndarray

    p0_shape: _Tuple
    x_plus_p0_shape: _Tuple 
    y_plus_p0_shape: _Tuple

    # intermediate quantities that we want to pre-allocate memory to save time allocating memory
    slpx: _nplike.ndarray
    slpy: _nplike.ndarray



def get_empty_states(config: Config, domain: Domain = None):
    """Get an empty (i.e., zero arrays) States.

    Arguments
    ---------
    domain : torchswe.utils.data.Domain
    ngh : int

    Returns
    -------
    A States with zero arrays.
    """

    # to hold data for initializing a States instance
    data = _DummyDict()

    # if domain is not provided, get a new one
    if domain is None:
        data.domain = domain = _get_domain(config)
    else:
        data.domain = domain

    # aliases
    ny, nx = data.domain.shape
    dtype = data.domain.dtype
    ngh = data.domain.nhalo

    # cell-centered arrays
    data.q = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=dtype)
    data.p = _nplike.zeros((3, ny+2*ngh, nx+2*ngh), dtype=dtype)
    data.s = _nplike.zeros((3, ny, nx), dtype=dtype)
    data.ss = _nplike.zeros((3, ny, nx), dtype=dtype) if config.friction is not None else None
    data.slpx = _nplike.zeros((3, ny, nx+2), dtype=dtype)
    data.slpy = _nplike.zeros((3, ny+2, nx), dtype=dtype)

    # quantities on faces
    data.face = FaceQuantityModel(
        x=FaceTwoSideModel(
            plus=FaceOneSideModel(
                q=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                p=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                a=_nplike.zeros((ny, nx+1), dtype=dtype),
                f=_nplike.zeros((3, ny, nx+1), dtype)
            ),
            minus=FaceOneSideModel(
                q=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                p=_nplike.zeros((3, ny, nx+1), dtype=dtype),
                a=_nplike.zeros((ny, nx+1), dtype=dtype),
                f=_nplike.zeros((3, ny, nx+1), dtype)
            ),
            cf=_nplike.zeros((3, ny, nx+1), dtype)
        ),
        y=FaceTwoSideModel(
            plus=FaceOneSideModel(
                q=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                p=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                a=_nplike.zeros((ny+1, nx), dtype=dtype),
                f=_nplike.zeros((3, ny+1, nx), dtype)
            ),
            minus=FaceOneSideModel(
                q=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                p=_nplike.zeros((3, ny+1, nx), dtype=dtype),
                a=_nplike.zeros((ny+1, nx), dtype=dtype),
                f=_nplike.zeros((3, ny+1, nx), dtype)
            ),
            cf=_nplike.zeros((3, ny+1, nx), dtype)
        ),
    )

    # SJ: Added to avoid using futures

    data.x_plus_p0_shape = data.face.x.plus.p[0].shape
    data.y_plus_p0_shape = data.face.y.plus.p[0].shape

    gravity = config.params.gravity
    data.gravity_x = _nplike.full(data.x_plus_p0_shape, gravity, dtype=dtype)
    data.gravity_y = _nplike.full(data.y_plus_p0_shape, gravity, dtype=dtype)

    data.gravity2_x = _nplike.full(data.x_plus_p0_shape, gravity/2, dtype=dtype)
    data.gravity2_y = _nplike.full(data.y_plus_p0_shape, gravity/2, dtype=dtype)

    data.p0_shape = data.p[0].shape

    return States(**data)


def get_initial_states(config: Config, domain: Domain = None):
    """Get a States instance filled with initial conditions.

    Arguments
    ---------

    Returns
    -------
    torchswe.utils.data.States

    Notes
    -----
    When x and y axes have different resolutions from the x and y in the NetCDF file, an bi-cubic
    spline interpolation will take place.
    """

    # get an empty states
    states = get_empty_states(config, domain)

    # rebind; aliases
    domain = states.domain

    # special case: constant I.C.
    if config.ic.values is not None:
        states.q[(slice(None),)+domain.nonhalo_c] = _nplike.array(config.ic.values).reshape((3, 1, 1))
        states.check()
        return states

    # otherwise, read data from a NetCDF file
    data = _read_block(config.ic.file, config.ic.xykeys, config.ic.keys, domain.lextent_c, domain)

    # see if we need to do interpolation
    try:
        interp = not (_nplike.allclose(domain.x.c, data.x) and _nplike.allclose(domain.y.c, data.y))
    except ValueError:  # assume thie excpetion means a shape mismatch
        interp = True

    # unfortunately, we need to do interpolation in such a situation
    if interp:
        _logger.warning("Grids do not match. Doing spline interpolation.")
        for i in range(3):
            states.q[(i,)+domain.nonhalo_c] = _nplike.array(
                _interpolate(data.x, data.y, data[config.ic.keys[i]].T, domain.x.c, domain.y.c).T
            )
    else:
        for i in range(3):
            states.q[(i,)+domain.nonhalo_c] = data[config.ic.keys[i]]

    states.check()
    return states
