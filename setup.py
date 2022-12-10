#build the modules
import os
from distutils.core import setup, Extension
import numpy

os.environ["CC"] = "g++"
os.environ["CFLAGS"] = "-std=c++2a"

setup(name='cpp2np', version='1.0', 
      ext_modules=[
        Extension('cpp2np', ['cpp2np.cpp'], include_dirs=[numpy.get_include()]
            )])
