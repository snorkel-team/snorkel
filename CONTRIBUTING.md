# Contributing to Snorkel

We love contributors, so first and foremost, thank you!
We're actively working on our contributing guidelines, so this document is subject to change.
First things first: read our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Development environment

### Installing

Snorkel uses [tox](https://tox.readthedocs.io) to manage development environments.
To get started, [install tox](https://tox.readthedocs.io/en/latest/install.html),
clone Snorkel, then use `tox` to create a development environment:

```bash
git clone https://github.com/HazyResearch/snorkel
pip3 install -U tox
cd snorkel
tox -e dev
```

Running `tox -e dev` will install the required packages in `requirements-dev.txt`
and create a virtual environment with Snorkel and all of its dependencies installed
in the directory `.env`.
This can be used in a number of ways, e.g. with `conda activate`
or for [linting in VSCode](https://code.visualstudio.com/docs/python/environments#_where-the-extension-looks-for-environments).
For example, you can simply activate this environment and start using Snorkel:

```bash
source .env/bin/activate

python3 -c "import snorkel; print(dir(snorkel))"
```

### Testing and committing

There are a number of useful tox commands defined:

```bash
tox -e py36  # Run unit tests pytest in Python 3.6
tox -e check  # Check style/linting with black, isort, and flake8
tox -e type  # Run static type checking with mypy
tox -e fix  # Fix style issues with black and isort
tox -e spark  # Run Spark-based tests (marked with @pytest.mark.spark)
tox -e complex  # Run more complex, integration tests (marked with @pytest.mark.complex)
tox  # Run unit tests, style checks, linting, and type checking
```

Make sure to run `tox` before committing.
CI won't pass without `tox` succeeding.

As noted, we use a few additional tools that help to ensure that any commits or pull requests you submit conform with our established standards.
We use the following packages:
* [isort](https://github.com/timothycrosley/isort): import standardization
* [black](https://black.readthedocs.io/en/stable/): automatic code formatting
* [flake8](http://flake8.pycqa.org/en/latest/): PEP8 linting
* [mypy](http://mypy-lang.org/): static type checking
* [pydocstyle](http://mypy-lang.org/): docstring compliance

The Snorkel maintainers are big fans of [VSCode](https://code.visualstudio.com/)'s Python tooling.
Here's a `settings.json` that takes advantage of the packages above (except isort) with in-line linting:

```json
{
    "python.jediEnabled": true,
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.pydocstyleEnabled": true,
    "python.linting.pylintEnabled": false,
}
```

### Docstrings

Snorkel ♥ documentation.
We expect all PRs to add or update API documentation for any affected pieces of code.
We use [NumPy style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html), and enforce style compliance with pydocstyle as indicated above.
Docstrings can be cumbersome to write, so we encourage people to use tooling to speed up the process.
For VSCode, we like [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).
Just install the extension and add the following configuration to the `settings.json` example above.
Note that we use PEP 484 type hints, so parameter types should be removed from the docstring (although note that return types should still be included).

```json
{
    "autoDocstring.docstringFormat": "numpy",
    "autoDocstring.guessTypes": false
}
```

### PySpark tests
PySpark tests are invoked separately from the rest since they require
installing Java and the large PySpark package.
They are executed on Travis, but not by default for a local `tox` command.
If you're making changes to Spark-based operators, make sure you have
Java 8 installed locally and then run `tox -e spark`.
If you add a test that imports PySpark mark it with the 
`@pytest.mark.spark` decorator.
Add the `@pytest.mark.complex` decorator as well if it runs a Spark
action (e.g. `.collect()`).


## PRs

### Submitting PRs

When submitting a PR, make sure to use the preformatted template.


### Requesting reviews

Direct commits to master are blocked, and PRs require a approving review
to merge into master.
By convention, the Snorkel maintainers will review PRs when:
  * An initial review has been requested
  * A maintainer is tagged in the PR comments and asked to complete a review


### Merging

The PR author owns the test plan and has final say on correctness.
Therefore, it is up to the PR author to merge their PR.
