import os
import re

import setuptools

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

    keywords='machine-learning ai information-extraction weak-supervision',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],

    project_urls={  # Optional
        'Homepage': 'https://hazyresearch.github.io/snorkel/',
        'Source': 'https://github.com/HazyResearch/snorkel/',
        'Bug Reports': 'https://github.com/HazyResearch/snorkel/issues',
        'Citation': 'https://doi.org/10.14778/3157794.3157797',
    },
)
