import os
import shutil
from setuptools import Extension, setup
from setuptools.command import build_ext


os.environ["CC"] = "g++"
module = Extension('il',
                   sources = [
                       'src/interactivelearning/PyInterface.cpp',
                       'src/interactivelearning/HnswHelper.cpp',
                       'src/interactivelearning/AnnoyHelper.cpp',
                       'src/interactivelearning/IVFHelper.cpp',
                       'src/interactivelearning/exq/ExqClassifier.cpp'
                   ],
                   include_dirs=[
                        '/usr/local/include/opencv4',
                        '/usr/local/hdf5/include',
                        '/usr/include/python3.10',
                        '/usr/include/faiss'
                    ],
                   library_dirs=[
                        '/usr/local/lib', #opencv and faiss
                        '/usr/local/hdf5/lib',
                        '/usr/lib/python3.10',
                    ],
                   libraries=['python3.10', 'hdf5', 'opencv_core', 'opencv_ml', 'faiss'],
                   extra_compile_args=['-O3', '-Wall', '-std=c++20'],
                   language='c++'
         )


setup(
    ext_modules=[module],
)