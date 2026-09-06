#!/usr/bin/env bash
set -euo pipefail

SIZES=(200 400 800 1200 1600 2000)
PROC_COUNTS=(1 2 4 8 16)

cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p data out logs

make -s

for n in "${SIZES[@]}"; do
    if [[ ! -f "data/A_${n}.txt" || ! -f "data/B_${n}.txt" ]]; then
        ./gen_matrix "$n"
    fi
done

for n in "${SIZES[@]}"; do
    for p in "${PROC_COUNTS[@]}"; do
        job_name="matmul_N${n}_P${p}"
        sbatch \
            --job-name="$job_name" \
            --ntasks-per-node="$p" \
            --export=ALL,N="$n",PROCS="$p" \
            startMPI.pbt
    done
done

echo "submitted $(( ${#SIZES[@]} * ${#PROC_COUNTS[@]} )) jobs"
