#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

source ../../../test/pftest.tcl
set passed 1
#set sig_digits 6

if ![pftestFile default_richards.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.press.00000.pfb "Max difference in perm_x" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.press.00001.pfb "Max difference in perm_y" $sig_digits] {
    set passed 0
}
if ![pftestFile default_richards.out.press.00002.pfb "Max difference in perm_z" $sig_digits] {
    set passed 0
}

puts "RES: $passed = 1 ? "
