#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -A oflash 
#SBATCH -p pbatch

echo 'STARTING JOB'

make check

echo 'ALL DONE'
