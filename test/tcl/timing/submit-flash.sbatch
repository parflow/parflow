#!/bin/bash
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -A oflash 
#SBATCH -p pbatch

echo 'STARTING JOB'

for i in *.tcl
do
   tclsh $i 1 1 1
done

echo 'ALL DONE'
