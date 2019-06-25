#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -A fabric 
#SBATCH -p pbatch

echo 'STARTING JOB'

# tclsh timing-richards.tcl 1 1 1

for i in *.tcl
do
   tclsh $i 1 1 1
done

echo 'ALL DONE'
