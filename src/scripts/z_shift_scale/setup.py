from distutils.core import setup, Extension
import numpy

shift_scale = Extension('mapping',
                  ['mapping.cc'],
                  include_dirs=[numpy.get_include()]
)

setup(name='mapping',
      ext_modules=[shift_scale]
)