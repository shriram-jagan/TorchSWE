
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

from torchswe import nplike as _nplike

class ConstValBC:
    def __init__(self):
        """..."""
        self.qbcm1 = None
        self.qbcm2 = None
        self.val = None

    def __call__(self):
        self.qbcm1[...] = self.val
        self.qbcm2[...] = self.val


def _const_val_bc_set_west(bc, Q, B, Bx, val, ngh, comp):
    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]
    bc.val = _nplike.full_like(bc.qbcm1, val, dtype=Q.dtype)

    B[ngh:B.shape[0]-ngh, ngh-1] = Bx[0:B.shape[0]-2*ngh, 0]
    B[ngh:B.shape[0]-ngh, ngh-2] = Bx[0:B.shape[0]-2*ngh, 0]


def _const_val_bc_set_east(bc, Q, B, Bx, val, ngh, comp):
    bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
    bc.val = _nplike.full_like(bc.qbcm1, val, dtype=Q.dtype)

    B[ngh:B.shape[0]-ngh, B.shape[1]-ngh]   = Bx[0:B.shape[0]-2*ngh, Bx.shape[1]-1]
    B[ngh:B.shape[0]-ngh, B.shape[1]-ngh+1] = Bx[0:B.shape[0]-2*ngh, Bx.shape[1]-1]


def _const_val_bc_set_south(bc, Q, B, By, val, ngh, comp):
    bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]
    bc.val = _nplike.full_like(bc.qbcm1, val, dtype=Q.dtype)

    B[ngh-1, ngh:B.shape[1]-ngh] = By[0, 0:B.shape[1]-2*ngh]
    B[ngh-2, ngh:B.shape[1]-ngh] = By[0, 0:B.shape[1]-2*ngh]


def _const_val_bc_set_north(bc, Q, B, By, val, ngh, comp):
    bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
    bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]
    bc.val = _nplike.full_like(bc.qbcm1, val, dtype=Q.dtype)

    B[B.shape[0]-ngh,   ngh:B.shape[1]-ngh] = By[By.shape[0]-1, 0:B.shape[1]-2*ngh]
    B[B.shape[0]-ngh+1, ngh:B.shape[1]-ngh] = By[By.shape[0]-1, 0:B.shape[1]-2*ngh]

def _const_val_bc_factory(bc, Q, B, Bx, By, val, ngh, comp, ornt):

    assert Q.shape[1] == B.shape[0]
    assert Q.shape[2] == B.shape[1]
    assert Q.shape[1] == Bx.shape[0] + 2 * ngh
    assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
    assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
    assert Q.shape[2] == By.shape[1] + 2 * ngh

    if ornt == 0:  # west
        _const_val_bc_set_west(bc, Q, B, Bx, val, ngh, comp)
    elif ornt == 1:  # east
        _const_val_bc_set_east(bc, Q, B, Bx, val, ngh, comp)
    elif ornt == 2:  # south
        _const_val_bc_set_south(bc, Q, B, By, val, ngh, comp)
    elif ornt == 3:  # north
        _const_val_bc_set_north(bc, Q, B, By, val, ngh, comp)
    else:
        raise ValueError(f"orientation id {ornt} not accepted.")

def const_val_bc_factory(ornt, comp, states, topo, val, *args, **kwargs):
    """Factory to create a constant-valued boundary condition callable object.
    """

    # aliases
    Q = states.q
    B = topo.c
    Bx = topo.xf
    By = topo.yf
    ngh = states.domain.nhalo

    if isinstance(ornt, str):
        if ornt in ["w", "west"]:
            ornt = 0
        elif ornt in ["e", "east"]:
            ornt = 1
        elif ornt in ["s", "south"]:
            ornt = 2
        elif ornt in ["n", "north"]:
            ornt = 3

    bc = ConstValBC()
    _const_val_bc_factory(bc, Q, B, Bx, By, val, ngh, comp, ornt)

    return bc
