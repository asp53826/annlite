#!/usr/bin/env bash
# SIFT small: 10k base vectors, 100 queries, and exact ground truth.
# The standard tiny ANN benchmark, about 5MB.
set -euo pipefail
cd "$(dirname "$0")/../data"
if [ -d siftsmall ]; then echo "siftsmall already present"; exit 0; fi
curl -L -o siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar xzf siftsmall.tar.gz
rm -f siftsmall.tar.gz
echo "got siftsmall"
