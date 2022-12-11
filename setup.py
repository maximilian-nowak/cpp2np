#build the modules
import os
import numpy

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


os.environ["CC"] = "g++"
os.environ["CFLAGS"] = "-std=c++2a"

setup(name='cpp2np', version='1.0', 
    ext_modules=[
        Extension('cpp2np', ['cpp2np.cpp'], include_dirs=[numpy.get_include()])
        ]
)
