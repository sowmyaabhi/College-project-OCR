from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_module = Extension(
    "TextDetection",
    ["TextDetection.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
	cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)

