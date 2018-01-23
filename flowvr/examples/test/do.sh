#!/bin/bash -x

export P=2
export Q=1
export R=1
export PROBLEMNAME=default_richards
export DEBUGFLAGS=""
$PARFLOW_DIR/bin/parflowvr/doMPI.sh
