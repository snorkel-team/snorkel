# Fonduer

`Fonduer` is a framework for building KBC applications from _richy formatted data_,
and is implemented as a library on top of Snorkel. To use `Fonduer`, please first
install Snorkel as shown in the [Snorkel README](https://github.com/HazyResearch/snorkel).

## Installation / dependencies

`Fonduer` adds some additional python packages to the default Snorkel installation which can be installed using `pip`:

```bash
pip install --requirement python-package-requirement.txt
```

If a package installation fails, then all of the packages below it in `python-package-requirement.txt` will fail to install as well. This can be avoided by running the following command instead of the above:
```bash
cat python-package-requirement.txt | xargs -n 1 pip install
```

Note that you may have to run `pip2` if you have Python3 installed on your system, and that `sudo` can be prepended to install dependencies system wide if this is an option and the above does not work.
For some pointers on difficulties in using `source` in shell, see [Issue 506](https://github.com/HazyResearch/snorkel/issues/506).

By default (e.g. in the tutorials, etc.) we also use [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/) for pre-processing text; you will be prompted to install this when you run `run.sh`.
In addition, we also use [`poppler`](https://poppler.freedesktop.org/) utilities for working with PDFs along with [PhantomJS](http://phantomjs.org/). You will also be prompted to install both of these when you run `run.sh`.


## Running
After installing Snorkel, and the additional python dependencies, just run:
```
./run.sh
```
which will finish installing the external libraries we use.

## Learning how to use `Fonduer`
The [`Fonduer` tutorials](https://github.com/HazyResearch/snorkel/tree/fonduer/tutorials/fonduer) cover the `Fonduer` workflow, showing how to extract relations from hardware datasheets and scientific literature.
The tutorial is available in the following directory:
```
tutorials/
```
