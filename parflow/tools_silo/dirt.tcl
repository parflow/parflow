#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************
proc dirtraverse {body} {
    set f [glob -nocomplain */]
    
    foreach i $f {
	if {$i != "RCS/" } {
	    set dir [pwd]
	    cd $i
	    uplevel $body	
	    dirtraverse $body
	    cd $dir
	}	
    }
}
