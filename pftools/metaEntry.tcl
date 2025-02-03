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

# File metaEntry.tcl - These procedures implement a meta widget composed
# of a label and an entry.  The label and entry are both packed to the
# left within a frame.  The `metaEntry' function is called to create a
# new meta entry or to perform operations on an existing one.  A
#
# To create a meta entry: metaEntry <meta_entry_path> <options>
# To perform commands   : metaEntry <existing_meta_entry_path> <commands>
#
# Commands can only be performed on meta entries that have already been
# created or else they are ignored.
#
# Options may be : -label, -width, -state
# Commands may be: insert, delete, get
#
# Structure of the meta entry:
#
# <meta_entry_path>.label     # Label the precede the entry
# <meta_entry_path>.entry     # Entry 
#



# metaEntry - This procedure accepts the path of a frame that is
# to be created, if the path given does not already exist.  If
# the path does already exist, then the user may change the 
# configuration options of the meta entry.
#
# Parameters - w    - The frame that acts as the meta widget
#              args - a list of options and commands
#             
# Return Value - The frame w that gets created if it has not
#                previously been created with `metaEntr', or
#                 the return value of a command such as `get'


proc XParflow::metaEntry {w args} {

   
   metaEntry_InitWidgetData $w options
   set retVal [metaEntry_ParseWidgetArgs $w options $args]

   # Create the meta widget if it does not already exist

   if {[info commands $w] == ""} {

      metaEntry_ConstructWidgetFrame $w options

   }

   return $retVal

}


# metaEntry_InitWidgetData - This procedure sets default
# options.
#
# Parameters - w    - The path of the meta entry
#              opts - an array that will hold
#                     configuration info
#

proc XParflow::metaEntry_InitWidgetData {w opts} {

   upvar $opts options

   set options(label)        {}
   set options(width)        {20}
   set options(state)        {normal}

}


# metaEntry_ParseWidgetArgs - This procedure parses the
# configuration options and the commands pass to `metaEntry'.
#
# Parameters - w    - the path of the meta entry
#              opts - the configuration options array
#              args - the options and commands
#
# Variables - retVal - The return value of a command
#             i      - Counts the arguments to `metaEntry'
#             n      - The number of arguments

proc XParflow::metaEntry_ParseWidgetArgs {w opts args} {

   upvar $opts options
   set retVal $w

   eval set args $args

   set i 0
   set n [llength $args]

   # Examine each argument.  IF it is an option, then set
   # the option.  If it is a command, then execute it.

   while {$i < $n} {

      # Determine the argument

      switch -- [lindex $args $i] {

          -label {
              incr i
              set options(label) [lindex $args $i]; incr i
          }
          -width {
              incr i
              set options(width) [lindex $args $i]; incr i
          }
          -state {
              incr i
              set options(state) [lindex $args $i]; incr i
          }
          insert {

              # The meta entry should exist first

              if {[info commands $w] != ""} {
                 incr i
                 set index [lindex $args $i]; incr i
                 set item [lindex $args $i]; incr i
                 $w.entry insert $index $item
              }
          }
          delete {

              # The meta entry should exist first

              if {[info commands $w] != ""} {
                 incr i
                 set first [lindex $args $i]; incr i
                 set second [lindex $args $i]; incr i
                 $w.entry delete $first $second
              }
          }
          get {

              # The meta entry should exits first

              if {[info commands $w] != ""} {
                 incr i
                 set retVal [$w.entry get]
              }
          }
          default {
             incr i
          }

      }

   }

   return $retVal

} 


# metaEntry_ConstructWidgetFrame - This procedure is used to 
# build the actual meta widget.
#
# Parameters - w    - The frame that forms the meta widget
#              opts - an array of configuration options
#

proc XParflow::metaEntry_ConstructWidgetFrame {w opts} {

   upvar $opts options

   # This frame and its contents will be the
   # meta entry

   frame $w -class metaEntry

   label $w.label -text $options(label)
   entry $w.entry -relief ridge -bd 2 \
                  -width $options(width) \
                  -state $options(state)

   pack $w.label $w.entry -side left

}
