#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# list of requirements
requirements = []
with open(os.path.join(HERE, "requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            requirements.append(line.strip())

setup(
    name="fuse-med-ml-examples",
    description="examples package for 'https://github.com/BiomedSciAI/fuse-med-ml/'",
    long_description="examples package for 'https://github.com/BiomedSciAI/fuse-med-ml/'",
    long_description_content_type="text/markdown",
    url="https://github.com/BiomedSciAI/fuse-med-ml/",
    author="IBM Research - Machine Learning for Healthcare and Life Sciences",
    author_email="moshiko.raboh@ibm.com",
    packages=find_packages(),
    license="Apache License 2.0",
    install_requires=requirements,
)
