import os
from setuptools import setup, find_packages
from io import open


with open('docs/readme.rst', 'r', encoding='utf-8') as f:
    long_description = f.read()

packages = ['elfi'] + ['elfi.' + p for p in find_packages('elfi')]

requirements = [
                'numpy>=1.8',
                'scipy>=0.16.1',
                'matplotlib>=1.1',
                'GPy>=1.0.9',
                'networkX>=1.11',
                'ipyparallel>=6.0'
                ]

optionals = {
    'doc': ['Sphinx'],
    'graphviz': ['graphviz>=0.5']
}

# read version number
__version__ = open('elfi/__init__.py').readlines()[-1].split(' ')[-1].strip().strip("'\"")

setup(
    name='elfi',
    keywords='abc likelihood-free statistics',
    packages=packages,
    version=__version__,
    author='HIIT',
    author_email='elfi-support@hiit.fi',
    url='http://elfi.readthedocs.io',

    install_requires=requirements,
    extras_require=optionals,

    description='Modular ABC inference framework for python',
    long_description=long_description,

    license='BSD',

    classifiers=['Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Bio-Informatics',
                 'Operating System :: OS Independent',
                 'Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License'],
    zip_safe = False)
