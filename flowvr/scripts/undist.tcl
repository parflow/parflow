lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

pfundist [lindex $argv 0]
