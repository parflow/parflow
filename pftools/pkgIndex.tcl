# Tcl package index file, version 1.0
if {![package vsatisfies [package provide Tcl] 8.5]} {puts "ERROR : pftools requires TCL 8.5 or greater"; return}

package ifneeded parflow 1.0 [list apply {dir {
    uplevel 1 [list source [file join $dir parflow.tcl]]
    uplevel 1 [list source [file join $dir pfvtk.tcl]]
    uplevel 1 [list source [file join $dir pftformat.tcl]] 
    if { [file exists [file join $dir parflow[info sharedlibextension]]] } { 
	load [file join $dir parflow[info sharedlibextension]]
    } else {
	load [file join $dir libpftools[info sharedlibextension]] parflow
    }
}} $dir]

package ifneeded xparflow 1.0 [list apply {dir {
    package require parflow
    uplevel 1 [list source [file join $dir xpftools.tcl]]
    uplevel 1 [list source [file join $dir xpfthelp.tcl]]
    uplevel 1 [list source [file join $dir metaEListBox.tcl]]
    uplevel 1 [list source [file join $dir metaEntry.tcl]]
    uplevel 1 [list source [file join $dir xpftgeneral.tcl]]
    uplevel 1 [list source [file join $dir xpftdatadsp.tcl]]
    uplevel 1 [list source [file join $dir xpftdiffdsp.tcl]]
    uplevel 1 [list source [file join $dir xpftstatdsp.tcl]]
    uplevel 1 [list source [file join $dir xpftfunctions.tcl]]
    uplevel 1 [list source [file join $dir xpftfuncwin.tcl]]
    uplevel 1 [list source [file join $dir xpftinfowin.tcl]]
    uplevel 1 [list source [file join $dir xpftgriddsp.tcl]]
    uplevel 1 [list source [file join $dir pftformat.tcl]]
    uplevel 1 [list source [file join $dir fsb.tcl]]
    uplevel 1 [list source [file join $dir xpftpf.tcl]]
    uplevel 1 [list source [file join $dir xpftsds.tcl]]
    uplevel 1 [list source [file join $dir xpftnewgrid.tcl]]
}} $dir]


