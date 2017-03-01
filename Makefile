
UNIT_TEST_FILES := $(wildcard test/*_test.py)
.PHONY: test
test: $(UNIT_TEST_FILES)
	$(foreach file,$(UNIT_TEST_FILES),python $(file);)
