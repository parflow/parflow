lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

cd results

#
# Tests
#
source ../pftest.tcl
set passed 1

if ![pftestFile default_richards.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

foreach i "00000 00001 00002 00003 00004 00005" {
    if ![pftestFile default_richards.out.press.$i.pfb "Max difference in Pressure for timestep $i" $sig_digits] {
    set passed 0
}
    if ![pftestFile default_richards.out.satur.$i.pfb "Max difference in Saturation for timestep $i" $sig_digits] {
    set passed 0
}
}

#
# PFTOOLS doesn't support netcdf yet so just see if file was created.
#

if ![file exists default_richards.out.00000.nc] {
    puts "NetCDF file was not created (00000)"
    set passed 0
}

if ![file exists default_richards.out.00001.nc] {
    puts "NetCDF file was not created (00001)"
    set passed 0
}

if $passed {
    puts "default_richards_with_netcdf : PASSED"
} {
    puts "default_richards_with_netcdf : FAILED"
}

