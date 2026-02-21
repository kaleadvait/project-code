# Makefile for client

.PHONY: help install run
.DEFAULT_GOAL := help

PY ?= python

help:
	@echo "Client targets:"
	@echo "  install  - Install minimal client dependencies (requests)"
	@echo "  run      - Run client.py to start interactive chat client"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install requests

run:
	$(PY) client.py
