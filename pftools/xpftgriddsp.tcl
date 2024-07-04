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

# Procedure GridFrame - This procedure is called in order to create
#           a frame which contains a description of a grid.  The
#           code is modular so that more than grid can have its
#           description on the display at once.
#
# Parameters - w - The name of the frame to be created
#
# Variables - None
#
# Return value - w

proc XParflow::GridFrame {w args} {

   if {[info commands $w] == ""} {

      GridFrame_ConstructWidgetFrame $w
      GridFrame_ManageWidgets        $w

   }

   GridFrame_ParseWidgetArgs $w options $args

   return $w

}

# Procedure GridFrame_ParseWidgetArgs - This procedure
# configures options.
#
# Parameters - w  - the path of the grid frame
#            opts - the options array
#            args - configuration arguments

proc XParflow::GridFrame_ParseWidgetArgs {w opts args} {

   upvar $opts options

   eval set args $args
   set i 0
   set n [llength $args]

   while {$i < $n} {

      switch -- [lindex $args $i] {

         -grid {  

            incr i
            set options(grid) [lindex $args $i]; incr i
            GridFrame_UpdateEntries $w $options(grid)
         }

         -entrystates {

            incr i
            set options(entryStates) [lindex $args $i]; incr i
            
            if {$options(entryStates) == "disabled" ||
                $options(entryStates) == "normal"} {

               GridFrame_SetEntryStates $w $options(entryStates)

            }

         }

         default {

            incr i

         }
 
      }

   }


}
   
# Procedure GridFrame_ConstructWidgetFrame - This procedure constructs
#           all of the widgets to be used in the creation of the grid
#           description.  The number of data points, grid origin, and
#           the intervals between each point are displayed using these
#           widgets.
#
# Parameters - w - The frame the grid description will be contained in
#
# Variables - None
#
# Return value - None

proc XParflow::GridFrame_ConstructWidgetFrame {w} {

   #-----------------------------------------------
   # Create the widget frame
   #-----------------------------------------------

   frame $w -relief raised -borderwidth 1
   label $w.title -text "Description of Grid"

   #-----------------------------------------------
   # Create three smaller frames to hold the number
   # of points, origin, and intervals.
   #-----------------------------------------------

   frame $w.dimension -relief groove -borderwidth 4
   frame $w.origin -relief groove -borderwidth 4
   frame $w.intervals -relief groove -borderwidth 4

   #-----------------------------------------------
   # Create labels and entries for the number of
   # points on each axis.
   #-----------------------------------------------
   
   label $w.dimension.nx -text "Dimension :  NX :"
   label $w.dimension.ny -text " NY :"
   label $w.dimension.nz -text " NZ :"
   entry $w.entnx -width 13
   entry $w.entny -width 13
   entry $w.entnz -width 13

   #-----------------------------------------------
   # Create labels and entries for the origin.
   #-----------------------------------------------

   label $w.origin.x -text "Origin    :   X :"
   label $w.origin.y -text "  Y :"
   label $w.origin.z -text "  Z :"
   entry $w.entx -width 13
   entry $w.enty -width 13
   entry $w.entz -width 13

   #-----------------------------------------------
   # Create labels and entries for the intervals
   # between each data point.
   #-----------------------------------------------
   
   label $w.intervals.dx -text "Intervals :  DX :"
   label $w.intervals.dy -text " DY :"
   label $w.intervals.dz -text " DZ :"
   entry $w.entdx -width 13
   entry $w.entdy -width 13
   entry $w.entdz -width 13

}


# Procedure GridFrame_ManageWidgets - This procedure manages the
#           widgets that were created by CreateWidgetFrame.
#
# Parameters - w - The frame the grid description will be contained in
#
# Variables - None
#
# Return value - None


proc XParflow::GridFrame_ManageWidgets {w} {

   pack $w.title

   #-------------------------------------------------------
   # Manage the labels and entries for the number of points
   #-------------------------------------------------------

   pack $w.dimension.nx -side left
   pack $w.entnx -in $w.dimension -side left
   pack $w.dimension.ny -side left
   pack $w.entny -in $w.dimension -side left
   pack $w.dimension.nz -side left
   pack $w.entnz -in $w.dimension -side left
   pack $w.dimension -anchor center -pady 5 -ipadx 4 -ipady 4

   #-------------------------------------------------------
   # Manage the labels and entries for the grid origin
   #-------------------------------------------------------

   pack $w.origin.x -side left
   pack $w.entx -in $w.origin -side left
   pack $w.origin.y -side left
   pack $w.enty -in $w.origin -side left
   pack $w.origin.z -side left
   pack $w.entz -in $w.origin -side left
   pack $w.origin -anchor center -pady 5 -ipadx 4 -ipady 4 

   #-------------------------------------------------------
   # Manage the labels and entries for the intervals
   # between points.
   #-------------------------------------------------------

   pack $w.intervals.dx -side left
   pack $w.entdx -in $w.intervals -side left
   pack $w.intervals.dy -side left
   pack $w.entdy -in $w.intervals -side left
   pack $w.intervals.dz -side left
   pack $w.entdz  -in $w.intervals -side left
   pack $w.intervals -anchor center -pady 5 -ipadx 4 -ipady 4

}


proc XParflow::GridFrame_UpdateEntries {w grid} {

   #-------------------------------------------------
   # Clear the entries
   #-------------------------------------------------

   $w.entnx delete 0 end
   $w.entny delete 0 end
   $w.entnz delete 0 end
   $w.entx  delete 0 end
   $w.enty  delete 0 end
   $w.entz  delete 0 end
   $w.entdx delete 0 end
   $w.entdy delete 0 end
   $w.entdz delete 0 end

   #-------------------------------------------------
   # Fill the entries that give that display the
   # number of points along each axis.
   #-------------------------------------------------

   $w.entnx insert 0 [lindex [lindex $grid 0] 0]
   $w.entny insert 0 [lindex [lindex $grid 0] 1]
   $w.entnz insert 0 [lindex [lindex $grid 0] 2]

   #-------------------------------------------------
   # Fill in the entries that are used to display the
   # origin.
   #-------------------------------------------------

   $w.entx  insert 0 [format "%e" [lindex [lindex $grid 1] 0]]
   $w.enty  insert 0 [format "%e" [lindex [lindex $grid 1] 1]]
   $w.entz  insert 0 [format "%e" [lindex [lindex $grid 1] 2]]

   #-------------------------------------------------
   # Fill in the entries that are used to display the
   # interval between each point along each axis.
   #-------------------------------------------------

   $w.entdx insert 0 [format "%e" [lindex [lindex $grid 2] 0]]
   $w.entdy insert 0 [format "%e" [lindex [lindex $grid 2] 1]]
   $w.entdz insert 0 [format "%e" [lindex [lindex $grid 2] 2]]
 
}


proc XParflow::GridFrame_SetEntryStates {w state} {

   $w.entnx configure -state $state
   $w.entny configure -state $state
   $w.entnz configure -state $state
   $w.entx  configure -state $state
   $w.enty  configure -state $state
   $w.entz  configure -state $state
   $w.entdx configure -state $state
   $w.entdy configure -state $state
   $w.entdz configure -state $state

}
