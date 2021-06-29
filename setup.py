#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(os.path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of requirements
requirements = []
with open(os.path.join(HERE, 'requirements.txt'), 'r') as fh:
    for line in fh:
        if not line.startswith('#'):
            requirements.append(line.strip())

setup(name='fuse',
      version='0.0.1',
      description='Open-source PyTorch based framework designed to facilitate deep learning R&D in medical imaging',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='TBD',
      author='Moshe Raboh',
      author_email='moshiko.raboh@ibm.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='LICENSE.txt',
      install_requires=requirements
      )
