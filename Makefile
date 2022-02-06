.PHONY: check check_flake8 check_pylint check_black

check: check_flake8 check_pylint check_black

check_flake8:
	flake8 .

check_pylint:
	pylint --persistent=no --score=no --rcfile setup.cfg impurityModel

check_black:
	black . --check
