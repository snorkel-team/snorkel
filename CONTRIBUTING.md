# Contributing to Snorkel

We love contributors, so first and foremost, thank you!
We're actively working on our contributing guidelines, so this document is subject to change.

First, read our [Code of Conduct](./CODE_OF_CONDUCT.md).

To set up our development environment, run `make dev`.
This will install a few additional tools that help to ensure that any commits or pull requests you submit conform with our established standards.
We use the following packages:
* [isort](https://github.com/timothycrosley/isort): import standardization
* [black](https://github.com/ambv/black): automatic code formatting
* [flake8](http://flake8.pycqa.org/en/latest/): PEP8 linting

After running `make dev` to install the necessary tools, you can run `make check` to see if any changes you've made violate the repo standards and `make fix` to fix any related to isort/black.
Fixes for flake8 violations will need to be made manually.

When submitting a PR, make sure to use the preformatted template.
