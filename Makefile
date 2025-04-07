# Makefile used to execute some commands locally

env:
	pip3 install -r requirements/build.txt
	pip3 install -r requirements/test.txt
	pip3 install -r requirements/develop.txt
	pre-commit install


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
	python3 setup.py sdist bdist_wheel; \
	ls dist

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean-wheel:
	@rm -rf dist

clean: clean-wheel clean-doc
	@rm -r build *.egg-info
