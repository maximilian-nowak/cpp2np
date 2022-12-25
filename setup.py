import os
import numpy

# if available use setuptools, otherwise distutils (deprecated)
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

os.environ["CC"] = "g++"
os.environ["CFLAGS"] = "-std=c++2a"

setup(name='cpp2np',
    version='1.3',
    description='Numpy extension for wrapping continuous C/C++ style arrays inside a numpy array using the same memory buffer.',
    ext_modules=[
        Extension('cpp2np',
            sources = ['cpp2np.cpp'],
            include_dirs = [numpy.get_include()],
        )
    ],
    author='Maxmilian Nowak',
    author_email='maximilian.nowak@hm.edu',
    url='https://gitlab.lrz.de/nowak/cpp2np'
)
