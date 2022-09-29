#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_namespace_packages


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(os.path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of requirements for core packages
fuse_requirements = []
with open(os.path.join(HERE, "fuse/requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            fuse_requirements.append(line.strip())

# list of requirements for core packages for development
fuse_requirements_dev = []
with open(os.path.join(HERE, "fuse/requirements_dev.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            fuse_requirements_dev.append(line.strip())

# list of requirements for fuseimg
fuseimg_requirements = []
with open(os.path.join(HERE, "fuseimg/requirements.txt"), "r") as fh:
    for line in fh:
        if not line.startswith("#"):
            fuseimg_requirements.append(line.strip())

# all extra requires
all_requirements = fuseimg_requirements + fuse_requirements_dev
# version
from fuse.version import __version__  # noqa

version = __version__

setup(
    name="fuse-med-ml",
    version=version,
    description="Open-source PyTorch based framework designed to facilitate deep learning R&D in medical imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiomedSciAI/fuse-med-ml/",
    author="IBM Research - Machine Learning for Healthcare and Life Sciences",
    author_email="moshiko.raboh@ibm.com",
    packages=find_namespace_packages(),
    license="Apache License 2.0",
    install_requires=fuse_requirements,
    extras_require={"fuseimg": fuseimg_requirements, "dev": fuse_requirements_dev, "all": all_requirements},
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
