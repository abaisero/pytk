#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools import setup


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='pytk',
    description='pytk - toolkit of convenient python methods',
    long_description=read('README.rst'),
    author='Andrea Baisero',
    url='https://github.com/bigblindbais/pytk',
    download_url='https://github.com/bigblindbais/pytk',
    author_email='andrea.baisero@gmail.com',
    version='0.0.1',
    install_requires=['numpy', ],
    packages=['pytk'],
    license='MIT',
    test_suite='tests',
)
