set tcl_precision 17

set runname test 

#set wrf_command $env(PARFLOW_DIR)/bin/wrf.exe
set wrf_command ./wrf.exe

set DBL_MAX 3.40282346638528859811704183485e+38
set NEG_DBL_MAX [expr -$DBL_MAX ]

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

#-----------------------------------------------------------------------------
# Read the processor topology and grid
#-----------------------------------------------------------------------------
#set dir_new ./run.d
#set dir_old ./run.nosamrai.problem

if { $::argc != 5 } {
    puts "Invalid args : compare <runname> <old_dir> <new_dir> <start_time> <stop_time>"
    exit -1
}

set runname [lindex $::argv 0]
set dir_old [lindex $::argv 1]
set dir_new [lindex $::argv 2]
set start   [lindex $::argv 3]
set stop    [lindex $::argv 4]

foreach file "press" { 

    for {set file_num $start} {$file_num <= $stop} {incr file_num} {
	puts "============== Working on $file_num ==================="

	set dir [pwd]
	cd $dir_new
	set filename [format "%s.out.%s.%05d.silo" $runname $file $file_num]
	puts "        reading $filename"
	set new  [pfload $filename $NEG_DBL_MAX]
	cd $dir



	set dir [pwd]
	cd $dir_old
	set filename [format "%s.out.%s.%05d.silo" $runname $file $file_num]
	puts "        reading $filename"
	set old  [pfload $filename $NEG_DBL_MAX]
	cd $dir

	puts "        diffing"
	set diff [pfmdiff $new $old 12]

	if {[string length $diff] != 0 } {
	    puts "FAILED : $filename"

	    set mSigDigs [lindex $diff 0]
	    set maxAbsDiff [lindex $diff 1]

	    set i [lindex $mSigDigs 0]
	    set j [lindex $mSigDigs 1]
	    set k [lindex $mSigDigs 2] 
	    puts [format "\tMinimum significant digits at (% 3d, % 3d, % 3d) = %2d"\
		      $i $j $k [lindex $mSigDigs 3]]

	    puts [format "\tMaximum absolute difference = %e" $maxAbsDiff]
	    
	    puts [format "\tOld value %e" [pfgetelt $old $i $j $k]]
	    puts [format "\tNew value %e" [pfgetelt $new $i $j $k]]

	    pfsavediff $new $old 8 -file difference.$filename.txt

	    set difference [pfaxpy -1 $new $old]
	    pfsave $difference -silo "difference.$filename"
	} {
	    puts "PASSED : files match"
        }

	pfdelete $new
	pfdelete $old

    }
}

