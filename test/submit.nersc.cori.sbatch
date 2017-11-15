#!/bin/bash -l

#SBATCH -p debug     
#SBATCH -N 2
#SBATCH -t 00:30:00  
#SBATCH -J my_job    
#SBATCH -L SCRATCH
#SBATCH -C haswell 

export OMP_NUM_THREADS=1
make check
