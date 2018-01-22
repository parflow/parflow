#!/bin/bash
#setenv VISIT /usr/local/apps/visit/2.0.0/linux-intel
export VISIT=$HOME/programs/visit2_12_3.linux-x86_64/2.12.3/linux-x86_64
export LD_LIBRARY_PATH=$VISIT/lib
export VISITPLUGINDIR=$VISIT/plugins
$PARFLOW_DIR/bin/visit-connector

