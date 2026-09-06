#!/usr/bin/env bash

N="${1:-500}"
ENGINE="${2:-gpu}"
TILE="${3:-16}"

make -s
python3 gen_matrix.py "$N"
./matmul data/A.txt data/B.txt data/C.txt data/stats.txt "$ENGINE" "$TILE"
python3 verify.py data/A.txt data/B.txt data/C.txt
