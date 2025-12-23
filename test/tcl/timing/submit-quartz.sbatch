#!/bin/bash
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A sss
#SBATCH -p pbatch

echo 'STARTING JOB'

for i in *.tcl
do
   tclsh $i 1 1 1
done

echo 'ALL DONE'
