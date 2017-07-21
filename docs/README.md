### Sphinx Documentation

We use [`sphinx.ext.autodoc`](http://www.sphinx-doc.org/en/stable/ext/autodoc.html)
to auto-generate documentation, and then [readthedocs](http://readthedocs.org/)
to render and host.

To test locally, just run:
```
make html
```

**Note: Most problems are caused by dependence on libraries that readthedocs can't
load (ones that rely on C libs) like `numpy` or `scipy`; just add these (and all
submodules loaded) to the `MOCK_MODULES` array in `conf.py`.**