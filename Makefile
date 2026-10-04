# Common tasks. See README.md.

SOURCES := $(shell find src tests -name '*.cpp' -o -name '*.hpp')

.PHONY: build test test-asan test-tsan datasets bench plots up format lint clean

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
	cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVEKTOR_SANITIZE=address
	cmake --build build-asan -j
	ctest --test-dir build-asan --output-on-failure

# Debug build with ThreadSanitizer, for the server's concurrency test.
test-tsan:
	cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DVEKTOR_SANITIZE=thread
	cmake --build build-tsan -j
	TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-tsan --output-on-failure

datasets: build
	scripts/download_datasets.sh

# Regenerates every file in results/. Run `make datasets` first.
bench: build
	mkdir -p results
	$(BENCH) --M 8,16,32 --out results/sift-100k.csv
	$(BENCH) --select simple,heuristic --out results/sift-100k-select.csv
	$(BENCH) --visited tags,hash --out results/sift-100k-visited.csv

# Draws the charts in docs/figures/ from results/, using a local Python venv.
plots: .venv
	.venv/bin/python scripts/plots.py

.venv: scripts/requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r scripts/requirements.txt
	touch .venv

# Builds the server's Docker image and runs it on http://localhost:8080 (Ctrl-C
# stops it). Index files go in data/. Ollama runs natively on this machine.
# Extra server flags: make up SERVER_ARGS="--llm-model qwen2.5:1.5b"
up:
	mkdir -p data
	docker build -t vektor-server .
	docker run --rm --name vektor -p 8080:8080 -v "$(CURDIR)/data:/data" \
		--add-host=host.docker.internal:host-gateway vektor-server $(SERVER_ARGS)

format:
	clang-format -i $(SOURCES)

# On macOS, SDKROOT tells clang-tidy where the system headers are.
lint: build
	SDKROOT=$$(xcrun --show-sdk-path 2>/dev/null) clang-tidy -p build --quiet $(filter %.cpp,$(SOURCES))

# Deletes build files, downloaded datasets, index files and the Python venv.
clean:
	rm -rf build build-asan build-tsan data .venv
