
dev:
	pip install -r requirements-dev.txt
	pre-commit install

check:
	isort -rc -c .
	black .
	flake8 .

fix:
	isort -rc .
	black .

test: check
	python3 -m pytest test/

.PHONY: dev check test