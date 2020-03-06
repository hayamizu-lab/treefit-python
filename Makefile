# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -j8 -W
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOGEN ?= sphinx-autogen
SOURCEDIR     = doc
BUILDDIR      = doc/_build

VERSION = $(shell \
  grep __version__ treefit/__init__.py | \
    sed -e "s/__version__ = '\\(.*\\)'/\\1/")

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	PYTHONLIB=. $(SPHINXAUTOGEN) $(SOURCEDIR)/api/*.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

update_document:
	$(MAKE) html
	rm -rf docs/$(VERSION)
	mv $(BUILDDIR)/html docs/$(VERSION)
	sed -i.bak -e "s,URL=.*/,URL=$(VERSION)/," docs/index.html
	rm -f docs/index.html.bak

dist:
	python3 setup.py sdist

upload:
	twine --repository treefit upload dist/treefit-$(VERSION).tar.gz

tag:
	git tag -a $(VERSION) -m "treefit $(VERSION) has been released!!!"
	git push --tags

release: dist upload tag
