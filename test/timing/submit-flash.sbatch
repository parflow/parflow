#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -A oflash 
#SBATCH -p pbatch

echo 'STARTING JOB'

# tclsh timing-richards.tcl 1 1 1

for i in *.tcl
do
   tclsh $i 1 1 1
done

echo 'ALL DONE'
