.PHONY: check check_flake8 check_pylint check_black check_mypy

check: check_flake8 check_pylint check_black check_mypy

check_flake8:
	flake8 .

check_pylint:
	pylint --persistent=no --score=no --rcfile setup.cfg impurityModel

check_black:
	black . --check

check_mypy:
	mypy --no-incremental --config-file setup.cfg -p impurityModel
