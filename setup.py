import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(name='fwht',ext_modules=cythonize("fwht.pyx",annotate=True),
include_dirs=[np.get_include()])

setup(name='exp',ext_modules=cythonize("exp.pyx",annotate=True),
include_dirs=[np.get_include()])

setup(name='aqc_c',ext_modules=cythonize("aqc_c.pyx",annotate=True),
include_dirs=[np.get_include()])
