
set tcl_precision 17

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

set in_filename [lindex $::argv 0]
set out_filename [lindex $::argv 1]

set in_file [pfload $in_filename]

set code [catch {eval pfgetgrid $in_file} grid]

# Extract grid size information
set dimension [lindex $grid 0]
set origin    [lindex $grid 1]
set interval  [lindex $grid 2]

set nx [lindex $dimension 0]
set ny [lindex $dimension 1]
set nz [lindex $dimension 2]

set x [lindex $origin 0]
set y [lindex $origin 1]
set z [lindex $origin 2]

set dx [lindex $interval 0]
set dy [lindex $interval 1]
set dz [lindex $interval 2]

set z_zero_plane [pfgetsubbox $in_file 0 0 0 $nx $ny 1]

pfsave $z_zero_plane -pfb $out_filename

 


