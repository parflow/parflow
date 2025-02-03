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

# File xpftstatdsp.tcl - This file contains functions that create
# and maintain a statistical display.  This display shows the min,
# max, mean, sum, variance, and standard deviation.

# Procedure StatDisplay - This function is called to create
# a statistics display.  If a widget path is given that does
# not exist, then a frame by that path name will be created
# and become the stat display.  If the path already exists, then
# the path should be that of a statistics display that already
# exits so that options can be passed to it.  The stat display
# is essentially a meta widget.
# 
# Parameters - w - The path of the stat display
#              args - Configuration arguments
#

proc XParflow::StatDisplay {w args} {

   # If the display does not already exist, then
   # create it

   if {[info commands $w] == ""} {

      StatDisplay_ConstructWidgetFrame $w
      StatDisplay_ManageWidgets        $w

   }

   StatDisplay_ParseWidgetArgs $w opts $args

   return $w
    
}


# Procedure StatDisplay_ParseWidgetArgs - This procedure reads
# the arguments passed to the stat display routine and parses
# them.  Once the arguments are parsed, the options are set.
#
# Parameters - w - The path of the stat display
#              opts - an array that will hold configuration
#                     options
#              args - The option arguments themselves
# 
# Return value - None

proc XParflow::StatDisplay_ParseWidgetArgs {w opts args} {

   upvar $opts options

   set options(data) ""

   eval set args $args
   set i 0
   set n [llength $args]

   # Examine each argument and determine if it is
   # a valid command or configuration option

   while {$i < $n} {

      switch -- [lindex $args $i] {

         -data {  

            incr i
            set options(data) [lindex $args $i]; incr i
            StatDisplay_UpdateEntries $w $options(data)

         }

         default {

            incr i

         }
 
      }

   }

}


# Procedure StatDisplay_ConstructWidgetFrame - This procedure
# creates the widgets that compose the statistics display
#
# Parameters - w - the path of the stat display
#
# Return value - None

proc XParflow::StatDisplay_ConstructWidgetFrame {w} {

   frame $w -relief raised -borderwidth 1
   frame $w.stats -relief groove -borderwidth 4
   label $w.label   -text Statistics

   frame $w.row1    -relief flat 
   label $w.row1.min     -text "Minimum        :"
   entry $w.entMin  -width 13 -state disabled -textvariable XParflow::statEnt($w:min)
   set XParflow::statEnt($w:min) ""
   label $w.row1.max     -text "  Maximum        :"
   entry $w.entMax  -width 13 -state disabled -textvariable XParflow::statEnt($w:max)
   set XParflow::statEnt($w:max) ""

   frame $w.row2    -relief flat
   label $w.row2.mean    -text "Mean           :"
   entry $w.entMean -width 13 -state disabled -textvariable XParflow::statEnt($w:mean)
   set XParflow::statEnt($w:mean) ""
   label $w.row2.sum     -text "  Sum            :"
   entry $w.entSum  -width 13 -state disabled -textvariable XParflow::statEnt($w:sum)
   set XParflow::statEnt($w:sum) ""

   frame $w.row3    -relief flat
   label $w.row3.var     -text "Variance       :"
   entry $w.entVar  -width 13 -state disabled -textvariable XParflow::statEnt($w:var)
   set XParflow::statEnt($w:var) ""
   label $w.row3.err     -text "  Standard Dev.  :"
   entry $w.entDev  -width 13 -state disabled -textvariable XParflow::statEnt($w:dev)
   set XParflow::statEnt($w:dev) ""

}


# Procedure StatDisplay_ManageWidgets - This procedure manages the
# above widgets.
#
# Parameters - w - the path of the stat window
#
# Return value - None

proc XParflow::StatDisplay_ManageWidgets {w} {

   pack $w.label

   pack $w.row1.min -side left
   pack $w.entMin   -in $w.row1 -side left
   pack $w.row1.max -side left
   pack $w.entMax   -in $w.row1 -side left

   pack $w.row2.mean -side left
   pack $w.entMean   -in $w.row2 -side left
   pack $w.row2.sum  -side left
   pack $w.entSum    -in $w.row2 -side left

   pack $w.row3.var -side left
   pack $w.entVar   -in $w.row3 -side left
   pack $w.row3.err -side left
   pack $w.entDev   -in $w.row3 -side left

   pack $w.row1     -in $w.stats -pady 5
   pack $w.row2     -in $w.stats -pady 5
   pack $w.row3     -in $w.stats -pady 5
   
   pack $w.stats -ipadx 4 -padx 10 -pady 5
 
}


# Procedure StatDisplay_UpdateEntries - This procedure fills the 
# stat display entries with the statistics derived from the
# data set passed to it.
#
# Parameters - w - the path of the stat display
#           dataSet - the data set the stats will be derived from
#
# Return value - None

proc XParflow::StatDisplay_UpdateEntries {w dataSet} {

   set newStats [pfgetstats $dataSet]

   set XParflow::statEnt($w:min)  [lindex $newStats 0]
   set XParflow::statEnt($w:max)  [lindex $newStats 1]
   set XParflow::statEnt($w:mean) [lindex $newStats 2]
   set XParflow::statEnt($w:sum)  [lindex $newStats 3]
   set XParflow::statEnt($w:var)  [lindex $newStats 4]
   set XParflow::statEnt($w:dev)  [lindex $newStats 5]

}


# Procedure StatDisplay_DestroyDisplay - This procedure is used to
# unset the globals used as entry texvariables and destroys the
# stat display.
#
# Parameters - w - the path of the stat display
#
# Return value - None

proc XParflow::StatDisplay_DestroyDisplay {w} {

   # Make sure they exist

   if {[info exists XParflow::statEnt($w:min)]} {
      unset XParflow::statEnt($w:min)
      unset XParflow::statEnt($w:max)
      unset XParflow::statEnt($w:mean)
      unset XParflow::statEnt($w:sum)
      unset XParflow::statEnt($w:var)
      unset XParflow::statEnt($w:dev)
   }

   destroy $w

}
