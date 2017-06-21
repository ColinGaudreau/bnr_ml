from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'

setup(
	ext_modules = cythonize(Extension(
		'ap_nms',
		sources=['ap_nms.pyx'],
		extra_compile_args=['-v', '-I', '/usr/local/python/bnr_ml/venv/lib/python2.7',
				   '-I', '/usr/local/python/bnr_ml/venv/lib/python2.7/site-packages/numpy/core/include',
				   '-mmacosx-version-min=10.11'],
		)
	)
)
