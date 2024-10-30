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

# File xpftinfowin.tcl - This file contains functions that are used
#                        to create an information window.  These
#                        windows contain a grid description, a
#                        statistics display, and a data display
#                        for a data set.


# Procedure InfoWindow - This function is called to create a
# new information window.  If the data set to be displayed in
# the info window is already being viewed by an info window,
# then a new info window will not be created.
#
# Parameters - dataSet - the data set to be viewed by the
#                        information window.


proc XParflow::InfoWindow {dataSet} {

   # Make sure the data set is not already being viewed

   if {[set w [InfoWindow_InitWidgetData $dataSet]] == ""} {

      return

   }

   InfoWindow_ConstructTopLevel $w $dataSet
   InfoWindow_ManageTopLevel    $w

   return $w

}


# Procedure InfoWindow_InitWidterData - This procedure creates
# a window path of the information window from the name of the
# data set.
#
# Parameters - dataSet - the data set to be viewed inside the
#                        information window.
#
# Return Value - The path of the info window

proc XParflow::InfoWindow_InitWidgetData {dataSet} {

   # reuturn null string if an info window already exists
   # for this data set

   if {[lsearch [info commands] .info$dataSet] != -1} {

      ErrorDialog "An information window for `$dataSet' is already open."
      return ""

   }

   return .info$dataSet

}

   
# Procedure InfoWindow_ConstructTopLevel - This procedure is used
# to create the information window.
#
# Parameters - w - the path of the information window
#              dataSet - the data set to be viewed
#
# Return value - None


proc XParflow::InfoWindow_ConstructTopLevel {w dataSet} {

   global env

   toplevel $w
   wm title $w "Information Window for `$dataSet'"
   wm resizable $w false false
   wm iconbitmap $w @$env(PARFLOW_DIR)/bin/info.xbm

   set grid      [pfgetgrid $dataSet]
   GridFrame $w.gridFrame -grid $grid -entrystates disabled

   StatDisplay $w.statDisplay -data $dataSet

   DataDisplay $w.dataDisplay -data $dataSet

   frame $w.btnFrame -relief raised -borderwidth 1
 
   button $w.btnFrame.close -text Close\
          -command "InfoWindow_DestroyWindow $w"

}


# Procedure InfoWindow_ManageTopLevel - This procedure 
# manages the widgets created above
#
# Parameters - w - The path of the info window
#
# Return value - None

proc XParflow::InfoWindow_ManageTopLevel {w} {

   pack $w.gridFrame -fill x -ipadx 10

   pack $w.statDisplay -fill x -ipadx 10

   pack $w.dataDisplay -fill x -ipady 5
   pack $w.btnFrame.close -expand 1
   pack $w.btnFrame -side bottom -ipady 5 -fill x
   

   CenterWindow $w
}


# Procedure Info_DestroyWindow - This procedure executes
# the destructors of the statistics and data displays
#
# Parameters - w - The path of the info window
#
# Return Value - None

proc XParflow::InfoWindow_DestroyWindow {w} {

   DataDisplay_DestroyDisplay $w.dataDisplay
   StatDisplay_DestroyDisplay $w.statDisplay
   destroy $w

}
   
