.PHONY: check check_flake8 check_pylint check_black check_mypy check_pytype

check: check_ruff check_black check_mypy check_pytype

check_ruff:
	ruff check .

check_black:
	black . --check

check_mypy:
	mypy --no-incremental --config-file setup.cfg -p impurityModel

check_pytype:
	pytype --verbosity 0 --pythonpath . input impurityModel
