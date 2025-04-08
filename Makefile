# Makefile used to execute some commands locally

env:
	pip3 install -e "."
	pip3 install -e ".[test]"
	pip3 install -e ".[dev]"


lint:
	pre-commit run --all-files


watch-lint:
	ptw --runner "pre-commit run -a"


test:
	pytest tests -s


watch-test:
	ptw --runner "pytest tests -s"


isort:
	isort .


black:
	black . -l 119


flake8:
	flake8 .


pydocstyle:
	pydocstyle --match-dir='(?!test|project).*'


wheel:
	python3 -m build

upload:
	twine upload dist/*

clean:
	@rm -rf dist build *.egg-info
