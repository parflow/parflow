#BHEADER***********************************************************************
# (c) 1996   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.3 $
#EHEADER***********************************************************************

# Procedure GrabFocus - This procedure is called in order to give a
#           window the focus and keep it until the user performs a
#           specific action such as pressing a `Cancel' or `OK' button.
#
# Parameters - w - The name of the window that will keep the focus
#
# Variables - global done - set to true when the focus is to be released
#
# Return value - None

proc XParflow::GrabFocus {w} {

   set XParflow::done false

   update idletasks
   grab $w
   tkwait variable done
   grab release $w
   
}


# Procedure ErrorDialog - This procedure creates an error dialog
#                         box.
#
# Parameters - msg - The error message to be displayed
#
# Return value - None

proc XParflow::ErrorDialog msg {

   bell

   tk_dialog .error {Error Message} $msg error 0 {OK} 

}
