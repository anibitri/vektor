#!/usr/bin/env bash
# Downloads the benchmark datasets one at a time, converts each one to a .vkd
# file in data/, then deletes the download to save disk space.
# Datasets that already have a .vkd file are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
bench=build/vektor-bench
if [[ ! -x $bench ]]; then
    echo "error: $bench not found; run 'make build' first" >&2
    exit 1
fi
mkdir -p data
tmp=$(mktemp -d data/download.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

# Downloads $1 to $2 and checks its SHA-256 hash ($3).
fetch() {
    curl -fL --retry 3 -o "$2" "$1"
    echo "$3  $2" | shasum -a 256 -c -
}

# SIFT1M (TEXMEX corpus, http://corpus-texmex.irisa.fr/): 168 MB download,
# about 690 MB of disk at peak while unpacking. We keep the first 100,000 base
# vectors and the first 1,000 queries.
if [[ ! -f data/sift-100k.vkd ]]; then
    echo "== SIFT"
    fetch ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz "$tmp/sift.tar.gz" \
        92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a
    tar -xzf "$tmp/sift.tar.gz" -C "$tmp" sift/sift_base.fvecs sift/sift_query.fvecs
    rm "$tmp/sift.tar.gz"
    "$bench" convert --base "$tmp/sift/sift_base.fvecs" --queries "$tmp/sift/sift_query.fvecs" \
        --n-base 100000 --n-queries 1000 --metric l2 --out data/sift-100k.vkd
    rm -rf "$tmp/sift"
fi

ls -lh data/*.vkd
