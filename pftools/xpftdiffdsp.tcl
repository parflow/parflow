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
# The DiffDisplay code is used to create and manage a listbox which views
# only small pieces of parflow diff data at a time.  This allows us to
# view the data without having to load it all into the list box at once.
#------------------------------------------------------------------------------

# Procedure DiffDisplay - This procedure is used to construct a
# diff display if the frame w does not already exist.  If it does,
# then its configuration can be changed via command line arguments.
#
# Parameters - w - the path of the diff display
#
# Return value - the path of the diff display

proc XParflow::DiffDisplay {w args} {

   # If the frame that is the diff display does not
   # exist, create it
 
   if {[info commands $w] == ""} {

      DataDisplay_ConstructWidgetFrame $w
      DataDisplay_ManageWidgets        $w

   }

   DiffDisplay_ParseWidgetArgs      $w $args

   return $w

}


# Procedure DiffDisplay_ParseWidgetArgs - This procedure is
# used to parse the commands and configuration options.
#
# Parameters - w  - the path of the diff display
#            args - The configuration options and commands
#
# Return value - None

proc XParflow::DiffDisplay_ParseWidgetArgs {w args} {

   eval set args $args
   set i 0
   set n [llength $args]

   # examine each argument

   while {$i < $n} {

      switch -- [lindex $args $i] {

         -data {

            incr i
            set dataSetA [lindex $args $i]; incr i
            set dataSetB [lindex $args $i]; incr i
            set sigDigs  [lindex $args $i]; incr i
            set absZero  [lindex $args $i]; incr i

            DataDisplay_InitWidgetData    $w $dataSetA
            DiffDisplay_SetWidgetBindings $w $dataSetA $dataSetB \
                                              $sigDigs $absZero
            DiffDisplay_FillDisplay       $w $dataSetA $dataSetB 0 0 0 \
                                              $sigDigs  $absZero

         }

         default {

            incr i

         }

      }

   }

}


# Procedure DiffDisplay_SetWidgetBindings - This procedure binds
# callbacks to the buttons and entries so that the diffs can be
# displayed when the scroll buttons are pressed.  This also allows
# the display to be changed from one set of diffs to another.
#
# Parameters - w        - the path of the diff display
#              dataSetA - the data sets to be diffed 
#              dataSetB
#              sigDigs  - the maximum number of differing sig digs
#                         the user will allow
#              absZero  - absolute zero above which all diffs must
#                         be to be displayed
#
# Return value - None

proc XParflow::DiffDisplay_SetWidgetBindings {w dataSetA dataSetB sigDigs absZero} {

   $w.coordFrame.entx configure -textvariable ::XParflow::dataDisp($w:entx)
   $w.coordFrame.enty configure -textvariable ::XParflow::dataDisp($w:enty)
   $w.coordFrame.entz configure -textvariable ::XParflow::dataDisp($w:entz)

   $w.btnFrame.up   configure \
     -command "DiffDisplay_UpButton $w $dataSetA $dataSetB $sigDigs $absZero"
   $w.btnFrame.down configure \
     -command "DiffDisplay_DownButton $w $dataSetA $dataSetB $sigDigs $absZero"
   $w.btnFrame.pgUp configure \
     -command "DiffDisplay_PgUpButton $w $dataSetA $dataSetB $sigDigs $absZero"
   $w.btnFrame.pgDn configure \
     -command "DiffDisplay_PgDownButton $w $dataSetA $dataSetB $sigDigs $absZero"

   bind $w.coordFrame.entx <Return> \
      "DiffDisplay_ViewCoordinate $w $dataSetA $dataSetB $sigDigs $absZero"
   bind $w.coordFrame.enty <Return> \
      "DiffDisplay_ViewCoordinate $w $dataSetA $dataSetB $sigDigs $absZero"
   bind $w.coordFrame.entz <Return> \
      "DiffDisplay_ViewCoordinate $w $dataSetA $dataSetB $sigDigs $absZero"

}


# Procedure DiffDisplay_FillDisplay - This procedure fills the
# display with differences from the top down.  The x, y, and z specify
# what coordinate that is displayed at the top
#
# Parameters - w        - the path of the data display
#              dataSetA - the data sets being diffed
#              dataSetB
#              i j k    - the coordinates to be placed at the
#                         top of the display
#              sigDigs  - the maximum number of differing sig digs
#                         the user will allow
#              absZero  - absolute zero above which all diffs must
#                         be to be displayed
#
# Return value - None

proc XParflow::DiffDisplay_FillDisplay {w dataSetA dataSetB x y z sigDigs absZero} {

   $w.listBox delete 0 end

   set ::XParflow::dataDisp($w:firstx) $x
   set ::XParflow::dataDisp($w:firsty) $y
   set ::XParflow::dataDisp($w:firstz) $z

   DataDisplay_UpdateEntries $w

   set numDisplayed 0

   # Keep inserting diffs into the list box until
   # there is no more space available

   for {set k $z} {$k < $::XParflow::dataDisp($w:nz)} {incr k} {

      for {set j $y} {$j < $::XParflow::dataDisp($w:ny)} {incr j} {

         for {set i $x} {$i < $::XParflow::dataDisp($w:nx)} {incr i} {

            set diffElt [pfdiffelt $dataSetA $dataSetB $i $j $k \
                                   $sigDigs $absZero]

            # Don't display the difference if it does not meet the given
            # criteria

            if {$diffElt == ""} {

               $w.listBox insert end [format "(%3d, %3d, %3d) = %s" $i $j $k \
                  "N/A"]
 
            } else {

               $w.listBox insert end [format "(%3d, %3d, %3d) = %13e" $i $j $k \
                  $diffElt]

            }
               
            incr numDisplayed

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
   


# Procedure DiffDisplay_DownButton - This procedure is used to
# scroll the display down one diff value.
#
# Parameters - w        - the path of the diff display
#              dataSetA - the data sets being diffed
#              dataSetB
#              sigDigs  - the maximum number of differing sig digs
#                         the user will allow
#              absZero  - absolute zero above which all diffs must
#                         be to be displayed
#
# Return value - None

proc XParflow::DiffDisplay_DownButton {w dataSetA dataSetB sigDigs absZero} {

   if {![DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:lastx) ::XParflow::dataDisp($w:lasty) \
                                     ::XParflow::dataDisp($w:lastz)]} {
      return

   }

   if {[$w.listBox size] == [$w.listBox cget -height]} {

       $w.listBox delete 0

   }

   set diffElt [pfdiffelt $dataSetA $dataSetB $::XParflow::dataDisp($w:lastx) \
                $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz) $sigDigs $absZero] 

   if {$diffElt == ""} {

      $w.listBox insert end [format "(%3d, %3d, %3d) = %s" $::XParflow::dataDisp($w:lastx) \
        $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz) "N/A"]

   } else {

      $w.listBox insert end [format "(%3d, %3d, %3d) = %13e" $::XParflow::dataDisp($w:lastx) \
           $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz) $diffElt]

   }

   DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:firstx) ::XParflow::dataDisp($w:firsty) \
                                  ::XParflow::dataDisp($w:firstz) 
   DataDisplay_UpdateEntries $w

}


# Procedure DiffDisplay_UpButton - This procedure is used to
# scroll the display up one diff value.
#
# Parameters - w       - the path of the data display
#              dataSetA - the data sets being diffed
#              dataSetB
#              sigDigs  - the maximum number of differing sig digs
#                         the user will allow
#              absZero  - absolute zero above which all diffs must
#                         be to be displayed
#
# Return value - None


proc XParflow::DiffDisplay_UpButton {w dataSetA dataSetB sigDigs absZero} {

   if {![DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:firstx) ::XParflow::dataDisp($w:firsty)         ::XParflow::dataDisp($w:firstz)]} {
      return
   
   }

   if {[$w.listBox size] == [$w.listBox cget -height]} {
     
      $w.listBox delete end
      DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:lastx) ::XParflow::dataDisp($w:lasty)    \
                                     ::XParflow::dataDisp($w:lastz)
   }

   set diffElt [pfdiffelt $dataSetA $dataSetB $::XParflow::dataDisp($w:firstx) \
                $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) $sigDigs $absZero]

   if {$diffElt == ""} {

      $w.listBox insert 0 [format "(%3d, %3d, %3d) = %s" $::XParflow::dataDisp($w:firstx)  \
         $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) "N/A"] 

   } else {

      $w.listBox insert 0 [format "(%3d, %3d, %3d) = %13e" $::XParflow::dataDisp($w:firstx)  \
         $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) $diffElt]

   }

   DataDisplay_UpdateEntries $w
   
}


# Procedure DataDisplay_PgDownButton - This procedure moves the data up
# to make the display appear to be moving down an entire page through the
# data.
#
# Parameters - w        - the path of the data display
#              dataSetA - the data sets being diffed
#              dataSetB
#              sigDigs  - the maximum number of differing sig digs
#                         the user will allow
#              absZero  - absolute zero above which all diffs must
#                         be to be displayed
#
# Return value - None

proc XParflow::DiffDisplay_PgDownButton {w dataSetA dataSetB sigDigs absZero} {

   for {set n 1} {$n <= [$w.listBox cget -height]} {incr n} {

      if {![DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:lastx)                                          ::XParflow::dataDisp($w:lasty) ::XParflow::dataDisp($w:lastz)]} {
         DataDisplay_UpdateEntries $w
         Return

      }

      set diffElt [pfdiffelt $dataSetA $dataSetB $::XParflow::dataDisp($w:lastx) \
         $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz) $sigDigs $absZero]

      if {$diffElt == ""} {

         $w.listBox insert end [format "(%3d, %3d, %3d) = %s" \
            $::XParflow::dataDisp($w:lastx) $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz) "N/A"]

      } else {
         
         $w.listBox insert end [format "(%3d, %3d, %3d) = %13e"            \
            $::XParflow::dataDisp($w:lastx) $::XParflow::dataDisp($w:lasty) $::XParflow::dataDisp($w:lastz)    \
            $diffElt]

      }

      DataDisplay_IncrCoordinate $w ::XParflow::dataDisp($w:firstx) ::XParflow::dataDisp($w:firsty) \
                                     ::XParflow::dataDisp($w:firstz)

      $w.listBox delete 0 

   }

   DataDisplay_UpdateEntries $w

}

# Procedure DiffDisplay_PgUpButton - This procedure moves the differences
# down to make the display appear to be moving up an entire page through the
# data.
#
# Parameters - w        - the path of the data display
#              dataSetA - the data sets being diffed
#              dataSetB
#              sigDigs  - the maximum number of differing sig digs
#                         the user will allow
#              absZero  - absolute zero above which all diffs must
#                         be to be displayed
#
# Return value - None

proc XParflow::DiffDisplay_PgUpButton {w dataSetA dataSetB sigDigs absZero} {

   for {set n 1} {$n <= [$w.listBox cget -height]} {incr n} {

      if {![DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:firstx)                                          ::XParflow::dataDisp($w:firsty) ::XParflow::dataDisp($w:firstz)]} {
         DataDisplay_UpdateEntries $w
         return

      }

      set diffElt [pfdiffelt $dataSetA $dataSetB $::XParflow::dataDisp($w:firstx) \
         $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) $sigDigs $absZero]

      if {$diffElt == ""} {

         $w.listBox insert 0 [format "(%3d, %3d, %3d) = %s" \
            $::XParflow::dataDisp($w:firstx) $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) \
            "N/A"]

      } else {

         $w.listBox insert 0 [format "(%3d, %3d, %3d) = %s" \
            $::XParflow::dataDisp($w:firstx) $::XParflow::dataDisp($w:firsty) $::XParflow::dataDisp($w:firstz) \
            $diffElt]

      }


      DataDisplay_DecrCoordinate $w ::XParflow::dataDisp($w:lastx) ::XParflow::dataDisp($w:lasty) \
                                     ::XParflow::dataDisp($w:lastz) 

      $w.listBox delete end 

   }

   DataDisplay_UpdateEntries $w

}


# Procedure DiffDisplay_ViewCoordinate - This procedure is used to
# move the display over a particular coordinate and view the data
# below it.
#
# Parameters - w       - the path of the data display
#              dataSet - the data set being displayed
#
# Return value - None

proc XParflow::DiffDisplay_ViewCoordinate {w dataSetA dataSetB sigDigs absZero} {

   if {[scan "$::XParflow::dataDisp($w:entx) $::XParflow::dataDisp($w:enty) $::XParflow::dataDisp($w:entz)" \
             "%d %d %d%s" n n n junk] != 3} { 

      DataDisplay_UpdateEntries $w
      return

   }
       
   if {$::XParflow::dataDisp($w:entx) < $::XParflow::dataDisp($w:nx) && $::XParflow::dataDisp($w:entx) >= 0 &&
       $::XParflow::dataDisp($w:enty) < $::XParflow::dataDisp($w:ny) && $::XParflow::dataDisp($w:enty) >= 0 &&
       $::XParflow::dataDisp($w:entz) < $::XParflow::dataDisp($w:nz) && $::XParflow::dataDisp($w:entz) >= 0 } {

      DiffDisplay_FillDisplay $w $dataSetA $dataSetB $::XParflow::dataDisp($w:entx) \
                               $::XParflow::dataDisp($w:enty) $::XParflow::dataDisp($w:entz)     \
                               $sigDigs $absZero

   } else {

      DataDisplay_UpdateEntries $w

   }

}
