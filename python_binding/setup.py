#!/usr/bin/env python

import sys
import os
from os import path

from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

here = path.abspath(path.dirname(__file__))

cuda_path = os.environ.get('CUDA_PATH', '')  # Nvidia default on Windows
if not cuda_path and path.exists('/usr/local/cuda'):
    cuda_path = '/usr/local/cuda'

extensions = [
    Extension("*", ["*.pyx"],
        language='c++', 
        include_dirs = [path.join(cuda_path, 'include'), path.abspath(path.join(here, '..', 'include'))],
        libraries = ['warpctc', 'cudart'],
        library_dirs = [path.join(cuda_path, 'lib64'), path.abspath(path.join(here, '..', 'build'))]),
]

setup(
    name='warpctc',
    
    url='https://github.com/baidu-research/warp-ctc', # ToDo
    #packages=['chainer',
    #          'chainer.functions'],

    install_requires=['numpy'],
    tests_require=['nose'],
	
    ext_modules=cythonize(extensions),


)
