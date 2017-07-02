from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='minhash',
    ext_modules=cythonize("minhash_numpy.pyx"),
    include_dirs=[numpy.get_include()]
)
