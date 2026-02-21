# Makefile for master dataset creation

.PHONY: help install run
.DEFAULT_GOAL := help

PY ?= python

help:
	@echo "Master dataset targets:"
	@echo "  install  - Install dependencies (SentenceTransformers, FAISS, PyPDF, NumPy)"
	@echo "  run      - Build master_data.pkl and master_index.faiss from PDFs"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install sentence-transformers faiss-cpu pypdf numpy

run:
	$(PY) create_master_dataset.py
