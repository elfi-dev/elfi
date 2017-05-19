import os
from setuptools import setup, find_packages
from io import open


with open('docs/readme.rst', 'r', encoding='utf-8') as f:
    long_description = f.read()

packages = ['elfi'] + ['elfi.' + p for p in find_packages('elfi')]

# include C++ examples
package_data = {'elfi.examples': ['cpp/Makefile', 'cpp/*.txt', 'cpp/*.cpp']}

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

optionals = {
    'doc': ['Sphinx'],
    'graphviz': ['graphviz>=0.7.1']
}

# read version number
__version__ = open('elfi/__init__.py').readlines()[-1].split(' ')[-1].strip().strip("'\"")

setup(
    name='elfi',
    keywords='abc likelihood-free statistics',
    packages=packages,
    package_data=package_data,
    version=__version__,
    author='ELFI authors',
    author_email='elfi-support@hiit.fi',
    url='http://elfi.readthedocs.io',

    install_requires=requirements,
    extras_require=optionals,

    description='Modular ABC inference framework for python',
    long_description=long_description,

    license='BSD',

    classifiers=['Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Operating System :: OS Independent',
                 'Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License'],
    zip_safe = False)
