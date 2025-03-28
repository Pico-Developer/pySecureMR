# Makefile used to execute some commands locally

OpenMR=${HOME}/code/securemr/OpenMR.python
OpenCV=$(OpenMR)/external/waterdrop/3rdparty/opencv/x86-64-ubuntu18.04/lib/
SNPE=$(OpenMR)/external/waterdrop/3rdparty/snpe/lib/x86_64-linux-clang/

build-cpp:
	cd $(OpenMR); bash cicd/ci_cmake.sh -t host -s 1
	cp $(OpenMR)/app/src/main/cpp/build/host/Pybind/_securemr.cpython-310-x86_64-linux-gnu.so securemr/
	cp $(OpenMR)/app/src/main/cpp/build/host/libopenmr-backend.so securemr/_C
	cp $(OpenCV)/libopencv_calib3d.so.3.4 securemr/_C
	cp $(OpenCV)/libopencv_core.so.3.4 securemr/_C
	cp $(OpenCV)/libopencv_imgcodecs.so.3.4 securemr/_C
	cp $(OpenCV)/libopencv_imgproc.so.3.4 securemr/_C
	cp $(OpenCV)/libopencv_flann.so.3.4 securemr/_C
	cp $(SNPE)/libSNPE.so securemr/_C

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
