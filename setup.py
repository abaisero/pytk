#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools import setup


# def read(fname):
#     """Utility function to read the README file."""
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='pytk',
    version='0.0.1',
    description='pytk - python toolkit',
    # long_description=read('README.rst'),
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/pytk',
    download_url='https://github.com/abaisero/pytk',

    packages=['pytk'],
    package_dir={'':'src'},

    install_requires=['numpy', 'scipy'],
    license='MIT',
    test_suite='tests',
)
