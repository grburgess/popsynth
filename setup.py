from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


import os

# Create list of data files
def find_data_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join('..', path, filename))

    return paths

#extra_files = find_data_files('pyspi/data')

setup(

    name="popsynth",
    packages=[
        'popsynth',
        'popsynth/utils'
    ],
    version='v1.0a',
    license='BSD',
    description='A population synth code',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',
    #   url = 'https://github.com/grburgess/pychangcooper',
 #   download_url='https://github.com/grburgess/pychangcooper/archive/1.1.2.tar.gz',

#    package_data={'': extra_files, },
#    include_package_data=True,

#    ext_modules=cythonize('popsynth/broken_powerlaw_sampler.pyx'),
    install_requires=[
        'numpy',
        'scipy',
        'ipython',
        'matplotlib',
        'h5py',
        'pandas',
        'seaborn',
        'astropy',
        'ipywidgets',
        'Cython'
    ],
)
