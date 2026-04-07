import os

from setuptools import find_packages, setup

__version__ = None

# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="cheminformatics_similarity",
    version=__version__,
    author="Kevin Spiekermann",
    description="This codebase contains functions to calculate similarity for cheminformatics workflows.",
    url="https://github.com/kspieks/cheminformatics_similarity/",
    packages=find_packages(),
    long_description=read('README.md'),
)
