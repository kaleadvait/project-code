# Makefile for server

.PHONY: help install run
.DEFAULT_GOAL := help

PY ?= python

help:
	@echo "Server targets:"
	@echo "  install  - Install server dependencies (Flask, SentenceTransformers, FAISS, PyPDF)"
	@echo "  run      - Run server.py (HTTPS on port 5000)"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install flask sentence-transformers faiss-cpu pypdf scikit-learn

run:
	$(PY) server.py
