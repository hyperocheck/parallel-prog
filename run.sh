#!/usr/bin/env bash

N="${1:-500}"

make -s
python3 gen_matrix.py "$N"
./matmul data/A.txt data/B.txt data/C.txt data/stats.txt
./venv/bin/python verify.py data/A.txt data/B.txt data/C.txt
