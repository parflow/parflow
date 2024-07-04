#!/bin/bash -l

##SBATCH -p debug     
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell
#SBATCH -t 02:00:00  
#SBATCH -J pf-regression
#SBATCH --account=m2983

source $PARFLOW_DIR/setenv.sh

export OMP_NUM_THREADS=1
make test 
