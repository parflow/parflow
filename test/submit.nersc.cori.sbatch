#!/bin/bash -l

#SBATCH -p debug     
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell
#SBATCH -t 00:30:00  
#SBATCH -J my_job    
#SBATCH -L SCRATCH

# source $PARFLOW_DIR/setenv.sh

export OMP_NUM_THREADS=1
make check
