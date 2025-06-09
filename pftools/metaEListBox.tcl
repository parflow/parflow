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

#############################################################################
#
# An "entry listbox" widget.
#
#
# To create:
#
#   metaEListBox <metaEListBox_name> [-label <label>]
#                                    [-width <width>]
#                                    [-entrystate <state>]
#                                    [-command <script>]
#                                    [insert <index> <item>]
#                                    [delete <index>]
#                                    [get <index>]
#
#   "metaEListBox" can be called on dropdown list widgets that have already
#   been created.  This allows you to modify the behavior of the dropdown
#   widget during its existence.  If you want to make more specific
#   configurations, then you will have to modify the widgets directly using
#   the "configure" command on the objects listed below.
#  
# The structure of the "meta" widget must be known in order to use:
#  
#   <metaEListBox_name>.label             # label widget
#   <metaEListBox_name>.entry             # entry box widget
#   <metaEListBox_name>.button            # dropdown button widget
#   <metaEListBox_name>.slist             # dropdown list widget
#   <metaEListBox_name>.slist.frame       # frame for dropdown list
#   <metaEListBox_name>.slist.listbox     # dropdown list listbox widget
#   <metaEListBox_name>.slist.scrollbar   # dropdown list scrollbar widget
#
#############################################################################

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

proc XParflow::metaEListBox {w args} {

    metaEListBox_InitWidgetData $w options
    set retVal [metaEListBox_ParseWidgetArgs $w options $args]

    if {[info commands $w] == ""} {
 
       metaEListBox_ConstructWidgetFrame $w options
       metaEListBox_SetWidgetBindings    $w options
  
    }

    return $retVal
}

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

proc XParflow::metaEListBox_InitWidgetData {w opts} {

    upvar $opts options

    set options(label)      {}
    set options(width)      {20}
    set options(entryState) {normal}
    set options(command)    {""}
}

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

proc XParflow::metaEListBox_ParseWidgetArgs {w opts args} {

    upvar $opts options
    set retVal $w

    eval set args $args

    set i 0
    set n [llength $args]
    while {$i < $n} {
	switch -- [lindex $args $i] {
	    -label {
		incr i
		set options(label) [lindex $args $i]; incr i
	    }
	    -width {
		incr i
		set options(width) [lindex $args $i]; incr i
	    }
            -entrystate {
                incr i
                set options(entryState) [lindex $args $i]; incr i
            }
            -command {
                incr i
                set options(command) [lindex $args $i]; incr i
            }
            insert {
                if {[info commands $w] != ""} {
                   incr i
                   set index [lindex $args $i]; incr i
                   set item [lindex $args $i]; incr i
                   $w.slist.listbox insert $index $item
                }
            }
            delete {
                if {[info commands $w] != ""} {
                   incr i
                   set index [lindex $args $i]; incr i
                   $w.slist.listbox delete $index
                }
            }
            get {
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


#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

proc XParflow::metaEListBox_ConstructWidgetFrame {w opts} {

    global env
    upvar $opts options

    #--------------------------------------------
    # construct entry frame
    #--------------------------------------------

    frame  $w -class metaEListBox
    label  $w.label \
	    -text $options(label)
    entry  $w.entry \
	    -relief ridge -bd 2 \
	    -width $options(width) \
            -state $options(entryState)
    button $w.button \
	    -bitmap @$env(PARFLOW_DIR)/bin/elistarrow.xbm \
	    -relief raised \
	    -takefocus 0 \
	    -command "metaEListBox_ButtonCommand $w" 

    pack $w.label  -side left -expand 1
    pack $w.entry  -side left 
    pack $w.button -side left -expand 1 -padx 2

    #--------------------------------------------
    # construct scrolled listbox frame
    #--------------------------------------------

    toplevel            $w.slist
    wm overrideredirect $w.slist 1
    wm withdraw         $w.slist

    frame $w.slist.frame -bd 2 -relief raised
    listbox $w.slist.listbox \
	    -width $options(width) \
	    -yscrollcommand "$w.slist.scrollbar set"
    scrollbar $w.slist.scrollbar \
	    -command "$w.slist.listbox yview"

    pack $w.slist.scrollbar -in $w.slist.frame -side right -fill y
    pack $w.slist.listbox   -in $w.slist.frame -side left
    pack $w.slist.frame
}

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

proc XParflow::metaEListBox_SetWidgetBindings {w opts} {

    upvar $opts options

    bind $w.slist.listbox <Double-Button-1> \
	    "metaEListBox_SelectEntryEvent  $w $options(command)"
}

#----------------------------------------------------------------------------
# Commands
#----------------------------------------------------------------------------

proc XParflow::metaEListBox_ButtonCommand {w} {

    focus $w.entry

    if {[wm state $w.slist] == "withdrawn"} {
       set x  [winfo rootx $w.entry]
       set y  [winfo rooty $w.entry]
       incr x  [winfo width  $w.entry]
       incr x -[winfo reqwidth $w.entry]
       incr x -2
       incr y  [winfo height $w.entry]

       wm deiconify $w.slist
       raise        $w.slist
       wm geometry  $w.slist +$x+$y
       set XParflow::prevGrab [grab current]
       grab -global $w
    } else {
       wm withdraw $w.slist
       grab release $w
       catch "grab $XParflow::prevGrab"
    }
}

#----------------------------------------------------------------------------
# Bindings
#----------------------------------------------------------------------------

proc XParflow::metaEListBox_SelectEntryEvent {w command} {

    if {[set index [$w.slist.listbox curselection]] != ""} {
       set prevState [$w.entry cget -state]
       $w.entry configure -state normal
       $w.entry delete 0 end 
       $w.entry insert 0 [$w.slist.listbox get $index]
       $w.entry configure -state $prevState
       metaEListBox_ButtonCommand $w
	puts "Invoking command"
       eval $command
    }
}
