# Common tasks. See README.md.

SOURCES := $(shell find src tests -name '*.cpp' -o -name '*.hpp')

.PHONY: build test test-asan datasets bench format lint clean

# Results record the commit they were measured on; "-modified" means the
# measured code (src/, CMakeLists.txt) had uncommitted changes.
COMMIT = $(shell git rev-parse --short HEAD)$(shell test -z "$$(git status --porcelain -- src CMakeLists.txt)" || echo -modified)
BENCH = build/vektor-bench run --data data/sift-100k.vkd --runs 3 --commit $(COMMIT)

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

# Regenerates every file in results/. Run `make datasets` first.
bench: build
	mkdir -p results
	$(BENCH) --M 8,16,32 --out results/sift-100k.csv
	$(BENCH) --select simple,heuristic --out results/sift-100k-select.csv
	$(BENCH) --visited tags,hash --out results/sift-100k-visited.csv

format:
	clang-format -i $(SOURCES)

# On macOS, SDKROOT tells clang-tidy where the system headers are.
lint: build
	SDKROOT=$$(xcrun --show-sdk-path 2>/dev/null) clang-tidy -p build --quiet $(filter %.cpp,$(SOURCES))

# Deletes build files and downloaded datasets.
clean:
	rm -rf build build-asan data
