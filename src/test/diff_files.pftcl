#
# This shows how to diff files
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

set new [pfload -pfb 1/default_single.out.perm_x.pfb]
set old [pfload -pfb 2/d2.out.perm_x.pfb]

puts "Max difference in perm"
puts [pfmdiff $new $old 12]

set new [pfload -pfsb 1/default_single.out.concen.0.00.00005.pfsb]
set old [pfload -pfsb 2/d2.out.concen.0.00.00005.pfsb]

puts "Max difference in concen"
puts [pfmdiff $new $old 12]

exit




