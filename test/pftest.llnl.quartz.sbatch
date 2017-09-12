#!/bin/sh
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -A cescyber
#SBATCH -p pbatch

date
make check

echo 'Done'
