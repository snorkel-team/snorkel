# TreeDLib

```bash
jupyter notebook
```

## Feature Ideas Doc
Shared google doc [here](https://docs.google.com/document/d/1vD6zvj1kNY40tO9knSyzdJ5qgyGfmzEAaZUaYBW-5-o/edit?usp=sharing)

## Dependencies
* _For notebook functionality:_
  * [Jupyter notebook](http://jupyter.readthedocs.org/en/latest/install.html) (`pip install jupyter`)
  * [IPython-SQL](https://github.com/catherinedevlin/ipython-sql) (`pip install ipython-sql`)
* [lxml](http://lxml.de/)

## Binning
This will eventually be done automatically (in some more or less-sophisticated manner...), however in the mean time, to use Indicator operators that use bins- such as `LengthBin(NodeSet, bin_divs)`- you can follow a simple rough procedure:

First, generate the features table, making sure to include full-path features for the lengths of interest.  For example, for sequence and dependency tree path lengths, you would need to include:
```python
Indicator(Between(Mention(0), Mention(1)), 'word')
Indicator(SeqBetween(), 'word')
```
(these are currently implemented as `get_relation_binning_features`).  Then, you can use code such as:
```sql
SELECT * FROM genepheno_features WHERE feature LIKE '%SEQ%'
```
```python
seq_lens = [len(rs.feature.split('_')) for rs in res_seq]
n, bins, patches = plt.hist(seq_lens, 50, normed=1, facecolor='green', alpha=0.75)
print [np.percentile(seq_lens, p) for p in [25,50,75]]
```
See `treedlib.ipynb` for an example implementation.
