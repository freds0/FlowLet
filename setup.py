#!/usr/bin/env python
import os

from setuptools import find_packages, setup

# Read README for long description
with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

# Read Version from flowlet/VERSION
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "flowlet", "VERSION"), encoding="utf-8") as fin:
    version = fin.read().strip()

# Read requirements from requirements.txt
def get_requires():
    requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements, encoding="utf-8") as reqfile:
        return [str(r).strip() for r in reqfile if r.strip() and not r.startswith("#")]

setup(
    name="flowlet",
    version=version,
    description="FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching",
    long_description=README,
    long_description_content_type="text/markdown",
    author="FlowLet Team",
    author_email="",
    url="https://github.com/freds0/FlowLet",
    install_requires=get_requires(),
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),

    # Console scripts
    entry_points={
        "console_scripts": [
            "flowlet-train=flowlet.train:main",
            "flowlet-sample=scripts.sample:main",
        ]
    },
    python_requires=">=3.9.0",
)
