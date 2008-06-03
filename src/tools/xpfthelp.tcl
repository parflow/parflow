# File xpfthelp.tcl - This file contains a global array of strings
# that contain help messages.  The messages are displayed in a pop-
# up dialog box.



# Procedure Help - This procedure creates a dialog box that
# displays help messages.
#
# Parameters - msg - The help message
#
# Return value - None

proc XParflow::Help {msg} {

   tk_dialog .help {PFTools Help} $msg questhead 0 {OK}

}
