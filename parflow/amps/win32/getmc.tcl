#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

set num_procs [lindex $argv 0]

# If user did not specify process group assume none
if {$argc > 1} {
    set filename .amps.info.[lindex $argv 1]
} {
    set filename .amps.info
}

if [file exists $filename] {
    exec attrib -h $filename
} 


# Get the number of processor to run on
set file [open $filename w]
puts $file $num_procs
close $file

exec attrib +h $filename

