
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

#cd results

pfundist default_richards
