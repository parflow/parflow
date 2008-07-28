#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

set new [pfloadpf llnl.out.perm.pfb]
set old [pfloadpf /home/ssmith/parflow.pre.newinput/exe.SunOS/llnl/llnl4.het.out.perm.pfb]

puts "Max difference in perm"
puts [pfmdiff $new $old 12]

#-----------------------------------------------------------------------------

set new [pfloadpf llnl.out.press.00000.pfb]
set old [pfloadpf /home/ssmith/parflow.pre.newinput/exe.SunOS/llnl/llnl4.het.out.press.00000.pfb]

puts "Max difference in press"
puts [pfmdiff $new $old 12]

set save [pfaxpy -1 $new $old]
pfsavepf $save -pfb press.diff.out.pfb


#-----------------------------------------------------------------------------

foreach i { 00000 00100 } {
    set new [pfloadpf llnl.out.concen.0.00.$i.pfsb]
    set old [pfloadpf /home/ssmith/parflow.pre.newinput/exe.SunOS/llnl/llnl4.het.out.concen.0.00.$i.pfsb]

    puts "Max difference in concen $i"
    puts [pfmdiff $new $old 12]

    set save [pfaxpy -1 $new $old]
    pfsavepf $save -pfb concenc.$i.diff.out.pfb
}



