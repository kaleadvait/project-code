# Makefile for training (retrain master from client datasets)

.PHONY: help install run
.DEFAULT_GOAL := help

PY ?= python

help:
	@echo "Training targets:"
	@echo "  install  - Install dependencies (SentenceTransformers, FAISS, NumPy)"
	@echo "  run      - Run trainmodel.py with defaults"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install sentence-transformers faiss-cpu numpy

run:
	$(PY) trainmodel.py
