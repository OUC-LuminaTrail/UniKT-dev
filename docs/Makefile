# Minimal Makefile for the Sphinx i18n workflow on UniKT docs.
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
GETTEXT_BDIR  = $(BUILDDIR)/gettext
LOCALEDIR     = $(SOURCEDIR)/locale

.PHONY: help gettext update build-mo html-en html-zh clean

help:
	@echo "i18n workflow (source language: zh_CN, target: en):"
	@echo "  make gettext   extract .pot from the Chinese source"
	@echo "  make update    create/update English .po under $(LOCALEDIR)/en"
	@echo "  make build-mo  compile .po into .mo (required before html-en)"
	@echo "  make html-en   build the English site into $(BUILDDIR)/html-en"
	@echo "  make html-zh   build the Chinese site into $(BUILDDIR)/html-zh"

gettext:
	$(SPHINXBUILD) -b gettext $(SOURCEDIR) $(GETTEXT_BDIR)

update:
	sphinx-intl update -p $(GETTEXT_BDIR) -d $(LOCALEDIR) -l en

build-mo:
	sphinx-intl build -d $(LOCALEDIR)

html-en: build-mo
	$(SPHINXBUILD) -b html -D language=en $(SOURCEDIR) $(BUILDDIR)/html-en

html-zh:
	$(SPHINXBUILD) -b html -D language=zh_CN $(SOURCEDIR) $(BUILDDIR)/html-zh

clean:
	rm -rf $(BUILDDIR)
