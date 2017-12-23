#!/bin/bash

rm _$PROBLEMNAME*.out.*
export P=2
export Q=1
export R=1
export PROBLEMNAME=hillslope_sens
export DEBUGFLAGS=""
echo test manually by calling $PARFLOW_DIR/bin/parflowvr/visit_loader.py
$PARFLOW_DIR/bin/parflowvr/doMPI.sh
$PARFLOW_DIR/bin/parflowvr/undist.tcl veg_map.pfb


