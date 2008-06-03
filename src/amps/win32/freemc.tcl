#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

# If user did not specify process group assume none
if {$argc > 0} {
    set filename .amps.info.[lindex $argv 0]
} {
    set filename .amps.info
}

if [file exists $filename] {
    exec attrib -h $filename
    file delete $filename
}

