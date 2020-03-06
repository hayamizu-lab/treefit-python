# Treefit for Python - The first software for quantitative trajectory inference

This is an implementation of
[**Treefit**](https://hayamizu-lab.github.io/treefit/) in Python.

**Treefit** is a novel data analysis toolkit that helps you perform
two types of analysis: 1) checking the goodness-of-fit of tree models
to your single-cell gene expression data; and 2) deciding which tree
best fits your data.

## Install

```bash
pip install treefit
```

## Usage

The main functions are `treefit.treefit()` and `treefit.plot()`:

```python
fit = treefit.treefit(YOUR_SINGLE_CELL_GENE_EXPRESSION_DATA)
treefit.plot(fit)
```

See https://hayamizu-lab.github.io/treefit-python/ for details.
