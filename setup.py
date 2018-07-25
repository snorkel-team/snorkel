# This setup script is provided as an unofficial alternative to installing Snorkel using Conda. (See README.md for
# the recommended installation instructions.) This script will not work on all systems. If you encounter errors, you
# might be able to resolve them by installing any packages that prefer Conda, such as PyTorch or Numba, before running.
import os
import re

import setuptools
from pkg_resources import Requirement

directory = os.path.dirname(os.path.abspath(__file__))

# Extract version information
path = os.path.join(directory, 'snorkel', '__init__.py')
with open(path) as read_file:
    text = read_file.read()
pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.MULTILINE)
version = pattern.search(text).group(1)

# Extract long_description
path = os.path.join(directory, 'README.md')
with open(path) as read_file:
    long_description = read_file.read()

# Extract package requirements from Conda environment.yml
install_requires = []
dependency_links = []
path = os.path.join(directory, 'environment.yml')
with open(path) as read_file:
    state = "PREAMBLE"
    for line in read_file:
        line = line.rstrip().lstrip(" -")
        if line == "dependencies:":
            state = "CONDA_DEPS"
        elif line == "pip:":
            state = "PIP_DEPS"
        elif state == "CONDA_DEPS":
            # PyTorch requires substituting the recommended pip dependencies
            requirement = Requirement(line)
            if requirement.key == "pytorch":
                install_requires.append(line.replace("pytorch", "torch", 1))
                install_requires.append("torchvision")
            else:
                # Appends to dependencies
                install_requires.append(line)
        elif state == "PIP_DEPS":
            # Appends to dependency links
            dependency_links.append(line)
            # Adds package name to dependencies
            install_requires.append(line.split("/")[-1].split("@")[0])

setuptools.setup(
    name='snorkel',
    version=version,
    url='https://github.com/HazyResearch/snorkel',
    description='A system for quickly generating training data with weak supervision',
    long_description_content_type='text/markdown',
    long_description=long_description,
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,

    keywords='machine-learning weak-supervision information-extraction',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],

    project_urls={  # Optional
        'Homepage': 'http://snorkel.stanford.edu',
        'Source': 'https://github.com/HazyResearch/snorkel/',
        'Bug Reports': 'https://github.com/HazyResearch/snorkel/issues',
        'Citation': 'https://doi.org/10.14778/3157794.3157797',
    },
)
