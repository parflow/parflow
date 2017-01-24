# Tcl package index file, version 1.0

package ifneeded parflow 1.0 [list \
				  [source [file join $dir parflow.tcl]] \
				  [source [file join $dir pftformat.tcl]] \
				  [source [file join $dir pfvtk.tcl]] \
				  [if {[file exists [file join $dir parflow[info sharedlibextension]]]} \
				       {load [file join $dir parflow[info sharedlibextension]]} \
				       else \
				       {load [file join $dir libpftools[info sharedlibextension]] "parflow"}] \
			      ]

package ifneeded xparflow 1.0 [list \
	[list [source [file join $dir xpftools.tcl]]] \
	[list [source [file join $dir xpfthelp.tcl]]] \
	[list [source [file join $dir metaEListBox.tcl]]] \
	[list [source [file join $dir metaEntry.tcl]]] \
	[list [source [file join $dir xpftgeneral.tcl]]] \
	[list [source [file join $dir xpftdatadsp.tcl]]] \
	[list [source [file join $dir xpftdiffdsp.tcl]]] \
	[list [source [file join $dir xpftstatdsp.tcl]]] \
	[list [source [file join $dir xpftfunctions.tcl]]] \
	[list [source [file join $dir xpftfuncwin.tcl]]] \
	[list [source [file join $dir xpftinfowin.tcl]]] \
	[list [source [file join $dir xpftgriddsp.tcl]]] \
	[list [source [file join $dir pftformat.tcl]]] \
	[list [source [file join $dir fsb.tcl]]] \
	[list [source [file join $dir xpftpf.tcl]]] \
	[list [source [file join $dir xpftsds.tcl]]] \
	[list [source [file join $dir xpftnewgrid.tcl]]] \
	]

