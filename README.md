<img src="figs/logo_01.png" width="150"/>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/snorkel)
![PyPI](https://img.shields.io/pypi/v/snorkel)
![Conda](https://img.shields.io/conda/v/conda-forge/snorkel)
[![build](https://travis-ci.com/snorkel-team/snorkel.svg?branch=master)](https://travis-ci.com/snorkel-team/snorkel?branch=master)
[![docs](https://readthedocs.org/projects/snorkel/badge/?version=master)](https://snorkel.readthedocs.io/en/master)
[![coverage](https://codecov.io/gh/snorkel-team/snorkel/branch/master/graph/badge.svg)](https://codecov.io/gh/snorkel-team/snorkel/branch/master)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Join the community on Spectrum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/snorkel)


***Programmatically Build and Manage Training Data***

# Quick Links
* [Snorkel website](https://snorkel.org)
* [Snorkel tutorials](https://github.com/snorkel-team/snorkel-tutorials)
* [Snorkel documentation](https://snorkel.readthedocs.io/)
* [Snorkel community forum](https://spectrum.chat/snorkel)
* [Snorkel mailing list](https://groups.google.com/forum/#!forum/snorkel-ml)
* [Snorkel Twitter](https://twitter.com/SnorkelML)

# Getting Started
The quickest way to familiarize yourself with the Snorkel library is to walk through the [Get Started](https://snorkel.org/get-started/) page on the Snorkel website, followed by the full-length tutorials in the [Snorkel tutorials](https://github.com/snorkel-team/snorkel-tutorials) repository.
These tutorials demonstrate a variety of tasks, domains, labeling techniques, and integrations that can serve as templates as you apply Snorkel to your own applications.


# Installation

Snorkel requires Python 3.6 or later. To install Snorkel, we recommend using `pip`:

```bash
pip install snorkel
```

or `conda`:

```bash
conda install snorkel -c conda-forge
```

For information on installing from source and contributing to Snorkel, see our
[contributing guidelines](./CONTRIBUTING.md).

<details><summary><b>Details on installing with <tt>conda</tt></b></summary>
<p>

The following example commands give some more color on installing with `conda`.
These commands assume that your `conda` installation is Python 3.6,
and that you want to use a virtual environment called `snorkel-env`.

```bash
# [OPTIONAL] Activate a virtual environment called "snorkel"
conda create --yes -n snorkel-env python=3.6
conda activate snorkel-env

# We specify PyTorch here to ensure compatibility, but it may not be necessary.
conda install pytorch==1.1.0 -c pytorch
conda install snorkel==0.9.0 -c conda-forge
```

</p>
</details>

<details><summary><b>A quick note for Windows users</b></summary>
<p>

If you're using Windows, we highly recommend using Docker
(you can find an example in our
[tutorials repo](https://github.com/snorkel-team/snorkel-tutorials/blob/master/Dockerfile))
or the [Linux subsystem](https://docs.microsoft.com/en-us/windows/wsl/faq).
We've done limited testing on Windows, so if you want to contribute instructions
or improvements, feel free to open a PR!

</p>
</details>

# Discussion

## Issues
We use [GitHub Issues](https://github.com/snorkel-team/snorkel/issues) for posting bugs and feature requests — anything code-related.
Just make sure you search for related issues first and use our Issues templates.
We may ask for contributions if a prompt fix doesn't fit into the immediate roadmap of the core development team.

## Contributions
We welcome contributions from the Snorkel community! 
This is likely the fastest way to get a change you'd like to see into the library.

Small contributions can be made directly in a pull request (PR).
If you would like to contribute a larger feature, we recommend first creating an issue with a proposed design for discussion. 
For ideas about what to work on, we've labeled specific issues as [`help wanted`](https://github.com/snorkel-team/snorkel/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+).

To set up a development environment for contributing back to Snorkel, see our [contributing guidelines](./CONTRIBUTING.md).
All PRs must pass the continuous integration tests and receive approval from a member of the Snorkel development team before they will be merged.

## Community Forum
For broader Q&A, discussions about using Snorkel, tutorial requests, etc., use the [Snorkel community forum](https://spectrum.chat/snorkel) hosted on Spectrum.
We hope this will be a venue for you to interact with other Snorkel users — please don't be shy about posting!

## Announcements
To stay up-to-date on Snorkel-related announcements (e.g. version releases, upcoming workshops), subscribe to the [Snorkel mailing list](https://groups.google.com/forum/#!forum/snorkel-ml). We promise to respect your inboxes — communication will be sparse!

## Twitter
Follow us on Twitter [@SnorkelML](https://twitter.com/SnorkelML).
