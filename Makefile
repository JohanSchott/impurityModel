.PHONY: check check_ruff check_black check_mypy

check: check_ruff check_black check_mypy

check_ruff:
	ruff check .

check_black:
	black . --check

check_mypy:
	mypy --no-incremental --config-file setup.cfg -p impurityModel
