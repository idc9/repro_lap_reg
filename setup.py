from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

# time, abc, numbers, copy, textwrap
# os, json, argparase, re


setup(name='repro_lap_reg',
      version='0.0.0',
      description='Reproduces the results of The folded concave Laplacian spectral penalty learns block diagonal sparsity patterns with the strong oracle property.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
