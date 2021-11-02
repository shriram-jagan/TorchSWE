#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Install TorchSWE.

Notes
-----
Configuration of the package/project has been migrated to setup.cfg. Only extension modules (i.e.,
those require being compiled first) remain here.
"""
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize, build_ext

exts = [
    Extension(
        name="torchswe.kernels.cython.minmod", sources=["torchswe/kernels/cython/minmod.pyx"],
        include_dirs=[numpy.get_include()], language="c++"),
    Extension(
        name="torchswe.kernels.cython.flux", sources=["torchswe/kernels/cython/flux.pyx"],
        include_dirs=[numpy.get_include()], language="c++"),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        exts,
        compiler_directives={"language_level": "3"},
    )
)
