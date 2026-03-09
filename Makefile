.PHONY: help init lint format test paper sanity

help:
	@echo "Targets:"
	@echo "  make init     - install locked deps"
	@echo "  make sanity   - run basic import + env checks"
	@echo "  make lint     - ruff + mypy"
	@echo "  make format   - black"
	@echo "  make test     - pytest"
	@echo "  make paper    - build paper PDF"

init:
	pip install -r requirements-dev.txt

sanity:
	python scripts/00_sanity.py

lint:
	ruff check src scripts
	mypy src

format:
	black src scripts paper

test:
	pytest -q

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
