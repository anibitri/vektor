# Common tasks. See README.md.

SOURCES := $(shell find src tests -name '*.cpp' -o -name '*.hpp')

.PHONY: build test test-asan datasets format lint clean

# Release build tuned for this machine's CPU. CI and Docker build without -march=native.
build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DVEKTOR_NATIVE=ON
	cmake --build build -j

test: build
	ctest --test-dir build --output-on-failure

# Debug build with AddressSanitizer and UndefinedBehaviorSanitizer, like CI.
test-asan:
	cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVEKTOR_SANITIZE=ON
	cmake --build build-asan -j
	ctest --test-dir build-asan --output-on-failure

datasets: build
	scripts/download_datasets.sh

format:
	clang-format -i $(SOURCES)

# On macOS, SDKROOT tells clang-tidy where the system headers are.
lint: build
	SDKROOT=$$(xcrun --show-sdk-path 2>/dev/null) clang-tidy -p build --quiet $(filter %.cpp,$(SOURCES))

# Deletes build files and downloaded datasets.
clean:
	rm -rf build build-asan data
