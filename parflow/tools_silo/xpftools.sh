#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

#BHEADER***********************************************************************
# (c) 1996   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

#
# Load in the required parflow packages 
#
lappend auto_path $env(PARFLOW_DIR)/bin/

    
if [catch { package require parflow } ] {
    puts "Error: Could not find parflow TCL library"
    exit
}

if [catch { package require xparflow } ] {
    puts "Error: Could not find xparflow TK library"
    exit
}


if [catch { package require vtktcl } ] {
    set XParflow::haveVTK 0
} {
    set XParflow::haveVTK 1
}

#
# To avoid typing import the namespaces from the packages
#
namespace import Parflow::*
namespace import XParflow::*

#
# Start up the main script
#
XParflow::MainWin_Init

