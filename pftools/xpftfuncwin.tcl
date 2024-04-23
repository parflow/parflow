#BHEADER**********************************************************************
#
#  Copyright (c) 1995-2024, Lawrence Livermore National Security,
#  LLC. Produced at the Lawrence Livermore National Laboratory. Written
#  by the Parflow Team (see the CONTRIBUTORS file)
#  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
#
#  This file is part of Parflow. For details, see
#  http://www.llnl.gov/casc/parflow
#
#  Please read the COPYRIGHT file or Our Notice and the LICENSE file
#  for the GNU Lesser General Public License.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License (as published
#  by the Free Software Foundation) version 2.1 dated February 1999.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
#  and conditions of the GNU General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
#  USA
#**********************************************************************EHEADER

# File xpftfuncwin.tcl - This file contains the functions
# necessary to maintain a standard look and feel between
# different PFTools commands/functions.  No matter what arguments
# the command/function takes, these utility procedures
# handle all of the details necessary for creating a GUI
# for the command/function.


# Procedure CreateFunctionWindow - This procedure creates a top-
# level window for the function input and output data to be
# displayed for each function.
# 
# Parameters - title - The title for the window
#
# Return value - None

proc XParflow::CreateFunctionWindow {title} {

   toplevel .func
   wm resizable .func false false
   wm title .func $title

}


# Procedure CreateStatDisplay - This procedure is used to
# place a statistics display in the function window.  These
# are placed only when the output data consists of at least
# one new data set.

proc XParflow::CreateStatDisplay {} {

   StatDisplay .func.statDisplay
   pack .func.statDisplay -fill x

}


# Procedure CreateDataDisplay - This procedure is used to 
# place a data display in the function window.  These
# are placed only when the output data consists of at least
# one new data set

proc XParflow::CreateDataDisplay {} {

   DataDisplay .func.dataDisplay
   pack .func.dataDisplay -fill x

}
   

# Procedure CreateDiffDisplay - This procedure is used to
# place a difference display in the function window.  These
# are use just with the diff function for now.

proc XParflow::CreateDiffDisplay {} {

   DiffDisplay .func.diffDisplay
   pack .func.diffDisplay -fill x

}

# Procedure FunctionEvenLoop - This procedure is used to
# grab the focus for the function window until the close
# button is pressed.

proc XParflow::FunctionEventLoop {} {

   CenterWindow .func
   GrabFocus .func

   # Clean up by destroying displays and unsetting
   # globals associated with their maintenence

   if {[info commands .func.statDisplay] != ""} {

      StatDisplay_DestroyDisplay .func.statDisplay

   }

   if {[info commands .func.dataDisplay] != ""} {

      DataDisplay_DestroyDisplay .func.dataDisplay
 
   }

   if {[info commands .func.diffDisplay] != ""} {

      DataDisplay_DestroyDisplay .func.diffDisplay

   }

   destroy .func

   if {[info vars XParflow::outputData] != ""} {

      unset XParflow::outputData

   }
}


# Procedure CreateIOFrame - This function creates a frame
# for the input and output of parameters and results.
# the frame is subdivided into two smaller frames --
# one for input parameters and one for output data
#
# Parameters inputEntries  - a list of labels and the
#                            the types of entries they will preceed
#            outputEntries -
#
# Return value - None
# 

proc XParflow::CreateIOFrame {inputEntries outputEntries} {

   frame .func.ioFrame -relief flat

   CreateInputFrame .func.ioFrame.inputFrame $inputEntries
   pack .func.ioFrame.inputFrame -side left -fill both

   if {[CreateOutputFrame .func.ioFrame.outputFrame $outputEntries]
       != ""} {

      pack .func.ioFrame.outputFrame -side left -fill both

   }

   pack .func.ioFrame -fill x

}


# Procedure CreateIOFrame - This procedure creates the acutal
# area where the parameters to the function are input.  For
# each element in the entries list, there will be an entry
# and a label created.  The entries will hold the input
# parameters.  Each entry may either be a dropdown list box
# entry or a standard entry.
#
# Parameters - w - the path of the function window
#              entries - the list of entry specifications and
#                        their labels
#
# Return value - None

proc XParflow::CreateInputFrame {w entries} {

   frame $w -relief raised -borderwidth 1
   label $w.label -text {Input Parameters}
   frame $w.centerFrame -relief flat
   pack $w.label

   set num 0

   foreach i $entries {

      set label [format "%-16s:" [lindex $i 0]] 
      set type [lindex $i 1]

      # Dropdown widgets and regular entry widgets are
      # both acceptable.  The only time a dropdown is
      # used is when the corresponding parameter is 
      # a data set.  Only data sets will be placed inside
      # the dropdown list box.

      switch $type {

         META_E_LIST_BOX {

            metaEListBox $w.entry$num -label $label -entrystate disabled
            FillInputSelectionBox $w.entry$num

         }

         META_ENTRY {

            metaEntry $w.entry$num -label $label

         }

      }

      $w.entry$num configure -relief groove -borderwidth 4 
      pack $w.entry$num -in $w.centerFrame -fill x -pady 5 -ipady 4
      incr num
   
   }

   pack $w.centerFrame -padx 4 -pady 5
   return $w

}


# Procedure CreateOutputFrame - This procedure is used to build
# the area where the output entries are displayed.  For each
# list item in the entries list, an entry with or without a
# radio button, will be created.  If it has a radio button next
# to it, then the item in the entry is a data set name and the
# data can be displayed by clicking on it.  If there is no radio
# button next to the entry, then the item in the entry is a scalar
# value.
#
# Parameters - w - The path of the function window
#              entries - List of labels to be placed
#                        before each output entry

proc XParflow::CreateOutputFrame {w entries} {

   global outputNum

   # Make sure there is output

   if {[llength $entries] != 0} {
 
      frame $w -relief raised -borderwidth 1
      label $w.label -text {Output Data}
      frame $w.centerFrame -relief flat
      pack $w.label

      set num 0

      # For each of the items in the list, get the label,
      # create an entry, and determine if there is to be
      # a radio button placed next to the entry

      foreach i $entries {

         set label [format "%-16s:" [lindex $i 0] ] 
         metaEntry $w.entry$num -label $label \
                                -state disabled
         $w.entry$num.entry configure -textvariable XParflow::outputData($num)
         $w.entry$num configure -relief groove -borderwidth 4

         switch [lindex $i 1] {

            RADIO_BUTTON {

               radiobutton $w.entry$num.rbutton  \
                           -variable outputNum   \
                           -value $num           \
                           -highlightthickness 0 \
                           -command {UpdateStatDisplay; UpdateDataDisplay}

               pack $w.entry$num.rbutton -side left

            }

         }

         pack $w.entry$num -in $w.centerFrame -fill x -pady 5 -ipady 4
         incr num

      }
  
      set outputNum 0
      pack $w.centerFrame -padx 4 -pady 5
      return $w

   }
   
   return ""

} 
   

# Procedure FillInputSelectionBox - This procedure is used to fill
# each dropdown widget with the name and label of each data set
#
# Parameters - w - the path of the function window
#
# Return value - None
#

proc XParflow::FillInputSelectionBox {w} {

   foreach i [pfgetlist] {

      set str [format "%-11s %-128s" [lindex $i 0] [lindex $i 1]]
      metaEListBox $w insert end $str

   }

}


# Procedure CreateFunctionButton - This procedure is used to create
# `Compute', `Help', and `Close' buttons
#
# Parameters - function - The callback associated with the compute
#                         button
#              helpStr  - The message to be displayed in the help
#                         dialog box
# 

proc XParflow::CreateFunctionButtons {function helpStr} {

   frame .func.btnFrame -relief raised -borderwidth 1
   frame .func.buttons -relief flat
   button .func.buttons.compute -text {Compute Function} -command $function
   button .func.buttons.help -text {Help} -command "Help $helpStr"
   button .func.buttons.close -text Close -command {set done true}
   pack .func.buttons.compute .func.buttons.help .func.buttons.close \
                                          -side left -padx 5 -pady 5
   pack .func.buttons -in .func.btnFrame
   pack .func.btnFrame -side bottom -fill x

}

   
# Procedure GetInputParameters - This procedure gets the values out of
# the entries that are in the Input Parameters area.
#
# Parameters - args - The list of parameters the user entered
#                     into each input entry.
#
# Return value - None

proc XParflow::GetInputParameters {args} {

   set n 0

   # for each entry, copy its contents to  	
   # the variable

   foreach i $args {

      upvar $i data

      switch [.func.ioFrame.inputFrame.entry$n cget -class] {

         metaEListBox {
 
            set str [metaEListBox .func.ioFrame.inputFrame.entry$n get]
            set data [lindex $str 0]

         }

         metaEntry {
 
            set str [metaEntry .func.ioFrame.inputFrame.entry$n get]
            set data [lindex $str 0]

         }

      }

      incr n

   }

}


# Procedure FillOutputEntries - This procedure places the
# items in args in the output entries.
#
# Parameters - args - variable list of output data to be
#                     placed in output entries.

proc XParflow::FillOutputEntries {args} {

   set n 0

   foreach i $args {

      set XParflow::outputData($n) $i
      incr n

   }

}


# Procedure UpdateStatDisplay - This procedure detects which radio
# button is pressed and updates the stat display.  The display
# will contain statistical data pertaining to the data set adjacent
# to the radio button that was pressed.
#
# Parameters - None
#
# Return value - None

proc XParflow::UpdateStatDisplay {} {

   global outputNum

   set dataSet [lindex [metaEntry .func.ioFrame.outputFrame.entry$outputNum get] 0]
   
   if {$dataSet != ""} {

      StatDisplay .func.statDisplay -data $dataSet

   }

}


# Procedure UpdateDataDisplay - This procedure also detects which
# radio button was pressed and updates the display to show the
# data set adjacent to the radio button pressed.
#
# Parameters - None
#
# Return value - None

proc XParflow::UpdateDataDisplay {} {

   global outputNum

   set dataSet [lindex [metaEntry .func.ioFrame.outputFrame.entry$outputNum get] 0]

   if {$dataSet != ""} {

      DataDisplay .func.dataDisplay -data $dataSet

   }

}
   

# Procedure UpdateDiffDisplay - This procedure updates the diff
# display everytime the `Compute Function' button is pressed
# on the diff display
#
# Parameters - None
#
# Return value - None

proc XParflow::UpdateDiffDisplay {dataSetA dataSetB sigDigs absZero} {

   DiffDisplay .func.diffDisplay \
      -data $dataSetA $dataSetB $sigDigs $absZero

}




