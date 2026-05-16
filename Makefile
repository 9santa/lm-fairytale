BUILD_DIR ?= build

.PHONY: all configure clean

all:
	$(MAKE) -C $(BUILD_DIR)

configure:
	cmake -S . -B $(BUILD_DIR)

clean:
	$(MAKE) -C $(BUILD_DIR) clean
