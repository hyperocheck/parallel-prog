#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH --time=0:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition batch
#SBATCH --output=logs/%x_%j.log

module load intel/mpi4

mpirun -r ssh -np "${PROCS}" ./matmul \
    "data/A_${N}.txt" "data/B_${N}.txt" \
    "out/C_${N}_${PROCS}.txt" "out/stats_${N}_${PROCS}.txt"
