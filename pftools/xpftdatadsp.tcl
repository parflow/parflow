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

#------------------------------------------------------------------------------
# The DataDisplay code is used to create and manage a listbox which views
# only small pieces of parflow solution data at a time.  This allows us to
# view the data without having to load it all into the list box at once.
#------------------------------------------------------------------------------


# Procedure DataDisplay - This procedure is used to construct a
# data display if the frame w does not already exist.  If it does,
# then its configuration can be changed via command line arguments.
#
# Parameters - w - the path of the data display
#
# Return value - the path of the data display

proc XParflow::DataDisplay {w args} {

   # If the frame that is the data display does not
   # exist, create it

   if {[info commands $w] == ""} {

      DataDisplay_ConstructWidgetFrame $w
      DataDisplay_ManageWidgets        $w

   }

   DataDisplay_ParseWidgetArgs      $w $args

   return $w

}


# Procedure DataDisplay_ParseWidgetArgs - This procedure is
# used to parse the commands and configuration options.
#
# Parameters - w  - the path of the data display
#            args - The configuration options and commands
#
# Return value - None
 
proc XParflow::DataDisplay_ParseWidgetArgs {w args} {

   eval set args $args
   set i 0
   set n [llength $args]

   # Examine each argument

   while {$i < $n} {

      switch -- [lindex $args $i] {

         -data {

            incr i
            set dataSet [lindex $args $i]; incr i
            DataDisplay_InitWidgetData    $w $dataSet
            DataDisplay_SetWidgetBindings $w $dataSet
            DataDisplay_FillDisplay       $w $dataSet 0 0 0

         }

         default {

            incr i

         }

      }

   }

}


# Procedure DataDisplay_DestroyDisplay - This procedure is used to 
# unset the texvariables in the display's entries and destroy the
# display and other global variables.
#
# Parameters - w - the path of the display
#
# Return value - None

proc XParflow::DataDisplay_DestroyDisplay {w} {

   # Make sure they exist

   if {[info exist ::XParflow::dataDisp($w:nx)]} {

      unset ::XParflow::dataDisp($w:nx)
      unset ::XParflow::dataDisp($w:ny)
      unset ::XParflow::dataDisp($w:nz)
      unset ::XParflow::dataDisp($w:firstx)
      unset ::XParflow::dataDisp($w:firsty)
      unset ::XParflow::dataDisp($w:firstz)
      unset ::XParflow::dataDisp($w:lastx)
      unset ::XParflow::dataDisp($w:lasty)
      unset ::XParflow::dataDisp($w:lastz)
      unset ::XParflow::dataDisp($w:entx)
      unset ::XParflow::dataDisp($w:enty)
      unset ::XParflow::dataDisp($w:entz)

   }

   destroy $w

}


# Procedure DataDisplay_InitWidgetData - This procedure is used
# to initialize the number of points in each dimension.
#
# Parameters - w       - the path of the display
#              dataSet - the data set being displayed

proc XParflow::DataDisplay_InitWidgetData {w dataSet} {

   set dimension [lindex [pfgetgrid $dataSet] 0]

   set ::XParflow::dataDisp($w:nx) [lindex $dimension 0]
   set ::XParflow::dataDisp($w:ny) [lindex $dimension 1]
   set ::XParflow::dataDisp($w:nz) [lindex $dimension 2]

}


# Procedure DataDisplay_ConstructWidgetFrame - This procedure
# is used to create the display
#
# Parameters - w - the path of the data display
#
# Return value - None

proc XParflow::DataDisplay_ConstructWidgetFrame {w} {

   global env

   frame $w -relief raised -borderwidth 1

   frame $w.coordFrame -relief flat
   label $w.coordFrame.coord -text "View X :"
   entry $w.coordFrame.entx -width 13
   label $w.coordFrame.y -text " Y :"
   entry $w.coordFrame.enty -width 13
   label $w.coordFrame.z -text " Z :"
   entry $w.coordFrame.entz -width 13

   listbox $w.listBox -height 15 

   frame $w.btnFrame -relief flat
   button $w.btnFrame.up   -bitmap @$env(PARFLOW_DIR)/bin/up.xbm
   button $w.btnFrame.down -bitmap @$env(PARFLOW_DIR)/bin/down.xbm
   button $w.btnFrame.pgUp -bitmap @$env(PARFLOW_DIR)/bin/pgup.xbm
   button $w.btnFrame.pgDn -bitmap @$env(PARFLOW_DIR)/bin/pgdown.xbm
}


# Procedure DataDisplay_SetWidgetBindings - This procedure binds
# callbacks to the buttons and entries so that the data set can be 
# displayed when the scroll buttons are pressed.  This also allows
# the display to be changed from one data set to another.
#
# Parameters - w       - the path of the data display
#              dataSet - the data set to be displayed
#
# Return value - None

proc XParflow::DataDisplay_SetWidgetBindings {w dataSet} {

   $w.coordFrame.entx configure -textvariable ::XParflow::dataDisp($w:entx)
   $w.coordFrame.enty configure -textvariable ::XParflow::dataDisp($w:enty)
   $w.coordFrame.entz configure -textvariable ::XParflow::dataDisp($w:entz)

   $w.btnFrame.up   configure -command "DataDisplay_UpButton $w $dataSet" 
   $w.btnFrame.down configure -command "DataDisplay_DownButton $w $dataSet"
   $w.btnFrame.pgUp configure -command "DataDisplay_PgUpButton $w $dataSet"
   $w.btnFrame.pgDn configure -command "DataDisplay_PgDownButton $w $dataSet"

   bind $w.coordFrame.entx <Return> "DataDisplay_ViewCoordinate $w $dataSet"
   bind $w.coordFrame.enty <Return> "DataDisplay_ViewCoordinate $w $dataSet"
   bind $w.coordFrame.entz <Return> "DataDisplay_ViewCoordinate $w $dataSet"

}


   
# Procedure DataDisplay_ManageWidgets - This procedure manages the
# data display.
#
# Parameters - w - the path of the data display
#
# Return value - None

proc XParflow::DataDisplay_ManageWidgets {w} {

   pack $w.coordFrame.coord $w.coordFrame.entx $w.coordFrame.entx \
        $w.coordFrame.y $w.coordFrame.enty $w.coordFrame.z $w.coordFrame.entz \
        -side left
   pack $w.coordFrame -anchor w -padx 10 -pady 5 

   pack $w.listBox -padx 10 -fill both -expand 1

   pack $w.btnFrame.up $w.btnFrame.down $w.btnFrame.pgUp $w.btnFrame.pgDn \
        -side left -pady 5 -padx 5
   pack $w.btnFrame -side bottom -padx 5 -fill y

}


# Procedure DataDisplay_UpdateEntries - This procedure updates the
# coordinates given in the X, Y, and Z entries along the top of the
# data display.
#
# Parameters - None
#
# Return value - None

proc XParflow::DataDisplay_UpdateEntries {w} {

   set ::XParflow::dataDisp($w:entx) $::XParflow::dataDisp($w:firstx)
   set ::XParflow::dataDisp($w:enty) $::XParflow::dataDisp($w:firsty)
   set ::XParflow::dataDisp($w:entz) $::XParflow::dataDisp($w:firstz)

}


# Procedure DataDisplay_FillDisplay - This procedure fills the
# display with data from the top down.  The x, y, and z specify
# what coordinate is displayed at the top
#
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#              i j k   - the coordinates to be placed at the
#                        top of the display
#
# Return value - None
 
proc XParflow::DataDisplay_FillDisplay {w dataSet x y z} {

   $w.listBox delete 0 end

   set ::XParflow::dataDisp($w:firstx) $x
   set ::XParflow::dataDisp($w:firsty) $y
   set ::XParflow::dataDisp($w:firstz) $z

   DataDisplay_UpdateEntries $w

   set numDisplayed 0

   # Keep displaying data until the list box is full

   for {set k $z} {$k < $::XParflow::dataDisp($w:nz)} {incr k} {

      for {set j $y} {$j < $::XParflow::dataDisp($w:ny)} {incr j} {

         for {set i $x} {$i < $::XParflow::dataDisp($w:nx)} {incr i} {

            $w.listBox insert end [format "(%3d, %3d, %3d) = %13e" $i $j $k \
                                          [pfgetelt $dataSet $i $j $k]]
            incr numDisplayed

            # Update the coordinate entries at the top of the
            # display and return.

            if {$numDisplayed == [$w.listBox cget -height]} {

               set ::XParflow::dataDisp($w:lastx) $i
               set ::XParflow::dataDisp($w:lasty) $j
               set ::XParflow::dataDisp($w:lastz) $k

               return

            }

         }

         set x 0

      }

      set y 0

   }

   set ::XParflow::dataDisp($w:lastx) [expr $::XParflow::dataDisp($w:nx) - 1]
   set ::XParflow::dataDisp($w:lasty) [expr $::XParflow::dataDisp($w:ny) - 1]
   set ::XParflow::dataDisp($w:lastz) [expr $::XParflow::dataDisp($w:nz) - 1]

}
   

# Procedure DataDisplay_IncrCoordinate - This procedure is used
# to increment the three dimensional coordinate.
#
# Parameters - w - the path of the data display
#             c1 - The x coordinate
#             c2 - The y coordinate
#             c3 - The z coordinate
#
# Return value - 1 if the triple could be incremented
#                0 if not

proc XParflow::DataDisplay_IncrCoordinate {w c1 c2 c3} {

   upvar $c1 x; set i $x
   upvar $c2 y; set j $y 
   upvar $c3 z; set k $z

   if {[incr i] >= $::XParflow::dataDisp($w:nx)} {

      set i 0

      if {[incr j] >= $::XParflow::dataDisp($w:ny)} {

         set j 0

         if {[incr k] >= $::XParflow::dataDisp($w:nz)} {

            return 0

         }

      }

   }

   set x $i
   set y $j
   set z $k

   return 1

}


# Procedure DataDisplay_DecrCoordinte - This procedure is used
# to decrement a three dimensional coordinate.
#
# Parameters - w - the path of the data display
#             c1 - the x coordinate
#             c2 - the y coordinate
#             c3 - the z coordinate
#
# Return value - 1 if the triple could be decremented
#                0 if not

proc XParflow::DataDisplay_DecrCoordinate {w c1 c2 c3} {

   upvar $c1 x; set i $x
   upvar $c2 y; set j $y 
   upvar $c3 z; set k $z

   if {[incr i -1] < 0} {
 
      set i [expr $::XParflow::dataDisp($w:nx) - 1] 
 
      if {[incr j -1] < 0} {

         set j [expr $::XParflow::dataDisp($w:ny) - 1] 

         if {[incr k -1] < 0} {

            return 0

         }

      }

   }
   
   set x $i
   set y $j
   set z $k

   return 1

}

   
# Procedure DataDisplay_DownButton - This procedure is used to
# scroll the display down one value.
#
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#
# Return value - None

proc XParflow::DataDisplay_DownButton {w dataSet} {

   # If the coordinate at the bottom of the display cannot be incremented
   # the the display can be scrolled down no further.

   if {![DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:lastx) ::XParflow::dataDisp($w:lasty) \
                                     ::XParflow::dataDisp($w:lastz)]} {
      return

   }

   if {[$w.listBox size] == [$w.listBox cget -height]} {

       $w.listBox delete 0

   }

   $w.listBox insert end [format "(%3d, %3d, %3d) = %13e" $::XParflow::dataDisp($w:lastx) \
        $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz) [pfgetelt $dataSet            \
        $::XParflow::dataDisp($w:lastx) $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz)]]

   DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:firstx) ::XParflow::dataDisp($w:firsty) \
                                  ::XParflow::dataDisp($w:firstz) 
   DataDisplay_UpdateEntries $w

}


# Procedure DataDisplay_UpButton - This procedure is used to
# scroll the display up one value.
#
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#
# Return value - None

proc XParflow::DataDisplay_UpButton {w dataSet} {

   # If the coordinate at the top cannot be decremented, the the display
   # cannot be scrolled up
 
   if {![DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:firstx) ::XParflow::dataDisp($w:firsty)         ::XParflow::dataDisp($w:firstz)]} {
      return
   
   }

   # If the, list box is full of data, then decr the coordinate at the bottom
   # also.

   if {[$w.listBox size] == [$w.listBox cget -height]} {
     
      $w.listBox delete end
      DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:lastx) ::XParflow::dataDisp($w:lasty)    \
                                     ::XParflow::dataDisp($w:lastz) 
   }

   $w.listBox insert 0 [format "(%3d, %3d, %3d) = %13e" $::XParflow::dataDisp($w:firstx)  \
      $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) [pfgetelt $dataSet            \
      $::XParflow::dataDisp($w:firstx) $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz)]]

   DataDisplay_UpdateEntries $w
   
}


# Procedure DataDisplay_PgDownButton - This procedure moves the data up
# to make the display appear to be moving down an entire page through the
# data.
# 
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#
# Return value - None

proc XParflow::DataDisplay_PgDownButton {w dataSet} {

   for {set n 1} {$n <= [$w.listBox cget -height]} {incr n} {

      if {![DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:lastx)                                          ::XParflow::dataDisp($w:lasty) ::XParflow::dataDisp($w:lastz)]} {
         DataDisplay_UpdateEntries $w
         return

      }

      $w.listBox insert end [format "(%3d, %3d, %3d) = %13e"            \
         $::XParflow::dataDisp($w:lastx) $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz)    \
         [pfgetelt $dataSet $::XParflow::dataDisp($w:lastx) $::XParflow::dataDisp($w:lasty)     \
          $::XParflow::dataDisp($w:lastz)]]

      DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:firstx) ::XParflow::dataDisp($w:firsty) \
                                     ::XParflow::dataDisp($w:firstz)

      $w.listBox delete 0 

   }

   DataDisplay_UpdateEntries $w

}


# Procedure DataDisplay_PgUpButton - This procedure moves the data down
# to make the display appear to be moving up an entire page through the
# data.
# 
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#
# Return value - None

proc XParflow::DataDisplay_PgUpButton {w dataSet} {

   # Decrement the topmost coordinate until an entire pageful
   # of data is passed 
   for {set n 1} {$n <= [$w.listBox cget -height]} {incr n} {

      # If no more is left, then update the entries and return

      if {![DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:firstx)                                          ::XParflow::dataDisp($w:firsty) ::XParflow::dataDisp($w:firstz)]} {
         DataDisplay_UpdateEntries $w
         return

      }

      $w.listBox insert 0 [format "(%3d, %3d, %3d) = %13e" $::XParflow::dataDisp($w:firstx)\
                           $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz)           \
                           [pfgetelt $dataSet $::XParflow::dataDisp($w:firstx)             \
                            $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz)]] 

      DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:lastx) ::XParflow::dataDisp($w:lasty)     \
                                     ::XParflow::dataDisp($w:lastz)

      $w.listBox delete end 

   }

   DataDisplay_UpdateEntries $w

}


# Procedure DataDisplay_ViewCoordinate - This procedure is used to 
# move the display over a particular coordinate and view the data
# below it.
#
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#
# Return value - None

proc XParflow::DataDisplay_ViewCoordinate {w dataSet} {

   # Make sure integers have been entered

   if {[scan "$::XParflow::dataDisp($w:entx) $::XParflow::dataDisp($w:enty) $::XParflow::dataDisp($w:entz)" \
             "%d %d %d %s" n n n junk] != 3} { 

      DataDisplay_UpdateEntries $w
      return

   }

   # Make sure the numbers are in range

   if {$::XParflow::dataDisp($w:entx) < $::XParflow::dataDisp($w:nx) && $::XParflow::dataDisp($w:entx) >= 0 &&
       $::XParflow::dataDisp($w:enty) < $::XParflow::dataDisp($w:ny) && $::XParflow::dataDisp($w:enty) >= 0 &&
       $::XParflow::dataDisp($w:entz) < $::XParflow::dataDisp($w:nz) && $::XParflow::dataDisp($w:entz) >= 0 } {

      DataDisplay_FillDisplay $w $dataSet $::XParflow::dataDisp($w:entx) \
                               $::XParflow::dataDisp($w:enty) $::XParflow::dataDisp($w:entz)

   } else {

      DataDisplay_UpdateEntries $w

   }

}
