
UNIT_TEST_FILES := $(wildcard test/*_test.py)
.PHONY: test
test: $(UNIT_TEST_FILES)
	$(foreach file,$(UNIT_TEST_FILES),python $(file);)

.PHONY: clean
clean:
	find . -iname '*.pyc' -delete
	rm -rf nnpack.egg-info
	rm -rf dist
	rm -rf build

.PHONY: package.build
package.build:
	rm -Rf dist
	python setup.py sdist
	python setup.py bdist_wheel

.PHONY: package.release
package.release:
	twine upload dist/*
