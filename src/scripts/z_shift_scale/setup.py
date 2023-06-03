from distutils.core import setup, Extension
import numpy

mapping = Extension('mapping',
                  ['mapping.c'],
                  include_dirs=[numpy.get_include()]
)

setup(name='mapping',
      ext_modules=[mapping]
)