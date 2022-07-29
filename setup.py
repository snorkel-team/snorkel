from typing import Dict

from setuptools import find_packages, setup

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import snorkel.
VERSION: Dict[str, str] = {}
with open("snorkel/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# Use README.md as the long_description for the package
with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="snorkel",
    version=VERSION["VERSION"],
    url="https://github.com/snorkel-team/snorkel",
    description="A system for quickly generating training data with weak supervision",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="Apache License 2.0",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    project_urls={
        "Homepage": "https://snorkel.org",
        "Source": "https://github.com/snorkel-team/snorkel/",
        "Bug Reports": "https://github.com/snorkel-team/snorkel/issues",
        "Citation": "https://doi.org/10.14778/3157794.3157797",
    },
    packages=find_packages(exclude=("test*",)),
    include_package_data=True,
    install_requires=[
        "munkres>=1.0.6",
        "numpy>=1.16.5",
        "scipy>=1.2.0",
        "pandas>=1.0.0",
        "tqdm>=4.33.0",
        "scikit-learn>=0.20.2",
        "torch>=1.2.0",
        "tensorboard>=2.9.1",
        "networkx>=2.2",
    ],
    python_requires=">=3.6",
    keywords="machine-learning ai weak-supervision",
)
