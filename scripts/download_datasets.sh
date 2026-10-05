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

# GloVe (glove-wiki-gigaword-100 from gensim-data: 400,000 words from Wikipedia
# and Gigaword, 100 dimensions): 134 MB download, 347 MB unpacked. The rows
# are sorted by word frequency, so we take a random sample (seed 42): 100,000
# base vectors and 1,000 other words as queries. Cosine distance.
if [[ ! -f data/glove-100k.vkd ]]; then
    echo "== GloVe"
    fetch https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz \
        "$tmp/glove.gz" 4934d4708bf3d65f28fc984dbbfa18af9cbd2505d166a7258a1a957808559b43
    gunzip "$tmp/glove.gz"
    "$bench" convert --base "$tmp/glove" --format word2vec --metric cosine \
        --n-base 100000 --n-queries 1000 --sample-seed 42 --out data/glove-100k.vkd
    rm "$tmp/glove"
fi

# Fashion-MNIST (Zalando Research): 28 x 28 grey images, 784 dimensions. 31 MB
# download. A random sample (seed 42) of 30,000 training images are the base
# vectors and 1,000 test images the queries. L2 distance.
if [[ ! -f data/fashion-mnist-30k.vkd ]]; then
    echo "== Fashion-MNIST"
    url=https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion
    fetch "$url/train-images-idx3-ubyte.gz" "$tmp/train.gz" \
        3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84
    fetch "$url/t10k-images-idx3-ubyte.gz" "$tmp/test.gz" \
        346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073
    gunzip "$tmp/train.gz" "$tmp/test.gz"
    "$bench" convert --base "$tmp/train" --queries "$tmp/test" --format idx --metric l2 \
        --n-base 30000 --n-queries 1000 --sample-seed 42 --out data/fashion-mnist-30k.vkd
    rm "$tmp/train" "$tmp/test"
fi

ls -lh data/*.vkd
