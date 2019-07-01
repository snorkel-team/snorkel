# Snorkel contrib

tl;dr Have something fun that you think others could benefit from but isn't
ready for prime time yet? Put it here!

Any code in this directory is not officially supported, and may change or be
removed at any time without notice.

The contrib directory contains project directories, each of which has designated
owners. It is meant to contain features and contributions whose interfaces may 
change, or which require some testing to see whether they can find broader acceptance.
You may be asked to refactor code in contrib to use some feature inside core or
in another contrib project rather than reimplementing the feature.

When adding a project, please stick to the following directory structure:
Create a project directory in `contrib/`, and mirror the portions of the
Snorkel tree that your project requires underneath `contrib/my_project/`.

For example, let's say you create foo ops for labeling in two files:
`foo_ops.py` and `foo_ops_test.py`. If you were to merge those files
directly into Snorkel, they would live in `snorkel/labeling/foo_ops.py` and
`test/labeling/foo_ops_test.py`. In `contrib/`, they are part
of project `foo`, and their full paths are `contrib/foo/snorkel/labeling/foo_ops.py`
and `contrib/foo/test/labeling/foo_ops_test.py`.


*Adapted from [TensorFlow contrib](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib).*
