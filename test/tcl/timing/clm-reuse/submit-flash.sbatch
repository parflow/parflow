#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -A oflash 
#SBATCH -p pbatch

echo 'STARTING JOB'

for i in clm-reuse.tcl
do
   tclsh $i 1 1 1
done

echo 'ALL DONE'
