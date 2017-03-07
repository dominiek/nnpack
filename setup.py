from setuptools import setup
from distutils.extension import Extension
from setuptools import find_packages

setup(
    name="nnpack",
    version="0.1.0",
    description='Packaging and Data Portability for Neural Networks',
    url='https://github.com/dominiek/nnpack',
    packages=find_packages()
)
