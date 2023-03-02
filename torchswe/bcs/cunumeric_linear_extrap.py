#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""This subpackage contain boundary-condition-related functions.
"""
from torchswe import nplike as _nplike

class LinearExtrapBC:
    """ Linear extrapolation boundary condition."""
    def __init__(self):

        # these will be updated later
        self.qc0 = None
        self.qc1 = None
        self.qbcm1 = None
        self.qbcm2 = None
        
    def __call__(self):
        """ Implementation of the boundary condition."""

        delta = self.qc0 - self.qc1;
        self.qbcm1[...] = self.qc0 + delta;
        self.qbcm2[...] = self.qbcm1 + delta;

def _linear_extrap_bc_set_west(bc, Q, B, Bx, ngh, comp):
    if comp < 0:
        bc.qc0      = Q[0:3, ngh:Q.shape[1]-ngh, ngh]
        bc.qc1      = Q[0:3, ngh:Q.shape[1]-ngh, ngh+1]
        bc.qbcm1    = Q[0:3, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qbcm2    = Q[0:3, ngh:Q.shape[1]-ngh, ngh-2]
    else:
        bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, ngh]
        bc.qc1      = Q[comp, ngh:Q.shape[1]-ngh, ngh+1]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, ngh-1]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, ngh-2]

    B[ngh:B.shape[0]-ngh, ngh-1] = Bx[0:B.shape[0]-2*ngh, 0] * 2.0 - B[ngh:B.shape[0]-ngh, ngh]
    B[ngh:B.shape[0]-ngh, ngh-2] = Bx[0:B.shape[0]-2*ngh, 0] * 4.0 - B[ngh:B.shape[0]-ngh, ngh] * 3.0

def _linear_extrap_bc_set_east(bc, Q, B, Bx, ngh, comp):
    if comp < 0:
        bc.qc0      = Q[0:3, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qc1      = Q[0:3, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
        bc.qbcm1    = Q[0:3, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qbcm2    = Q[0:3, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]
    else:
        bc.qc0      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-1]
        bc.qc1      = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh-2]
        bc.qbcm1    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh:Q.shape[1]-ngh, Q.shape[2]-ngh+1]

    B[ngh:B.shape[0]-ngh, B.shape[1]-ngh]   = Bx[0:B.shape[0]-2*ngh, Bx.shape[1]-1] * 2.0 - B[ngh:B.shape[0]-ngh, B.shape[1]-ngh-1]
    B[ngh:B.shape[0]-ngh, B.shape[1]-ngh+1] = Bx[0:B.shape[0]-2*ngh, Bx.shape[1]-1] * 4.0 - B[ngh:B.shape[0]-ngh, B.shape[1]-ngh-1] * 3.0


def _linear_extrap_bc_set_south(bc, Q, B, By, ngh, comp):
    if comp < 0:
        bc.qc0      = Q[0:3, ngh,      ngh:Q.shape[2]-ngh]
        bc.qc1      = Q[0:3, ngh+1,    ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[0:3, ngh-1,    ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[0:3, ngh-2,    ngh:Q.shape[2]-ngh]
    else:
        bc.qc0      = Q[comp, ngh,      ngh:Q.shape[2]-ngh]
        bc.qc1      = Q[comp, ngh+1,    ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[comp, ngh-1,    ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, ngh-2,    ngh:Q.shape[2]-ngh]

    # modify the topography elevation in ghost cells
    B[ngh-1, ngh:B.shape[1]-ngh] = By[0, 0:B.shape[1]-2*ngh] - B[ngh, ngh:B.shape[1]-ngh]
    B[ngh-2, ngh:B.shape[1]-ngh] = By[0, 0:B.shape[1]-2*ngh] * 4.0 - B[ngh, ngh:B.shape[1]-ngh] * 3.0


def _linear_extrap_bc_set_north(bc, Q, B, By, ngh, comp):
    if comp < 0:
        bc.qc0      = Q[0:3, Q.shape[1]-ngh-1,     ngh:Q.shape[2]-ngh]
        bc.qc1      = Q[0:3, Q.shape[1]-ngh-2,     ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[0:3, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[0:3, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]
    else:
        bc.qc0      = Q[comp, Q.shape[1]-ngh-1,     ngh:Q.shape[2]-ngh]
        bc.qc1      = Q[comp, Q.shape[1]-ngh-2,     ngh:Q.shape[2]-ngh]
        bc.qbcm1    = Q[comp, Q.shape[1]-ngh,       ngh:Q.shape[2]-ngh]
        bc.qbcm2    = Q[comp, Q.shape[1]-ngh+1,     ngh:Q.shape[2]-ngh]

    B[B.shape[0]-ngh, ngh:B.shape[1]-ngh]   = By[By.shape[0]-1, 0:B.shape[1]-2*ngh] * 2.0 - B[B.shape[0]-ngh-1, ngh:B.shape[1]-ngh]
    B[B.shape[0]-ngh+1, ngh:B.shape[1]-ngh] = By[By.shape[0]-1, 0:B.shape[1]-2*ngh] * 4.0 - B[B.shape[0]-ngh-1, ngh:B.shape[1]-ngh] * 3.0


def _linear_extrap_bc_factory(bc, Q, B, Bx, By, ngh, comp, ornt):
    assert Q.shape[1] == B.shape[0]
    assert Q.shape[2] == B.shape[1]
    assert Q.shape[1] == Bx.shape[0] + 2 * ngh
    assert Q.shape[2] == Bx.shape[1] + 2 * ngh - 1
    assert Q.shape[1] == By.shape[0] + 2 * ngh - 1
    assert Q.shape[2] == By.shape[1] + 2 * ngh

    if ornt == 0:  # west
        _linear_extrap_bc_set_west(bc, Q, B, Bx, ngh, comp)
    elif ornt == 1:  # east
        _linear_extrap_bc_set_east(bc, Q, B, Bx, ngh, comp)
    elif ornt == 2:  # south
        _linear_extrap_bc_set_south(bc, Q, B, By, ngh, comp)
    elif ornt == 3:  # north
        _linear_extrap_bc_set_north(bc, Q, B, By, ngh, comp)
    else:
        raise ValueError(f"orientation id {ornt} not accepted.")


def linear_extrap_bc_factory(ornt, comp, states, topo, *args, **kwargs):
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

    bc = LinearExtrapBC()

    # this function does the one-time initialization and assertion checks
    # and then updates the arrays in the bc class 
    _linear_extrap_bc_factory(bc, Q, B, Bx, By, ngh, comp, ornt)

    return bc

