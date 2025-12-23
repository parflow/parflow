#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

set P  [lindex $argv 0]
set Q  [lindex $argv 1]
set R  [lindex $argv 2]

pfset Process.Topology.P $P
pfset Process.Topology.Q $Q   
pfset Process.Topology.R $R

set NumProcs [expr $P * $Q * $R]

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                -10.0
pfset ComputationalGrid.Lower.Y                 10.0
pfset ComputationalGrid.Lower.Z                  1.0

pfset ComputationalGrid.DX	                 8.8888888888888893
pfset ComputationalGrid.DY                      10.666666666666666
pfset ComputationalGrid.DZ	                 1.0

pfset ComputationalGrid.NX                      10
pfset ComputationalGrid.NY                      10
pfset ComputationalGrid.NZ                       8

set mask [pfload correct_output/samrai.out.mask.pfb]

set top [pfcomputetop $mask]
pfsave $top -sa "top.sa"

set bottom [pfcomputebottom $mask]
pfsave $bottom -sa "bottom.sa"

set domain [pfcomputedomain $top $bottom]
puts $domain

set out [pfprintdomain $domain]

set grid_file [open samrai_grid.tcl w]
puts $grid_file $out

file copy correct_output/samrai.out.mask.pfb samrai.out.mask.pfb

pfdistondomain samrai.out.mask.pfb $domain

pfundist samrai.out.mask.pfb

source pftest.tcl
set passed 1

if ![pftestFile samrai.out.mask.pfb "Max difference in mask" $sig_digits] {
    set passed 0
}
