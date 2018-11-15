#!/bin/bash

rm _$PROBLEMNAME*.out.*
P=1
Q=2
R=1
PROBLEMNAME=hillslope_sens
echo test manually by calling $PARFLOW_DIR/bin/parflowvr/visit_loader.py
$PARFLOW_DIR/bin/parflowvr/doMPI.sh $PROBLEMNAME $P $Q $R
