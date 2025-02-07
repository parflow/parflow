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

# FSBox.tcl -- This file contains the procedures necessary to create and
#              maintain a file selection box.  Scrolled directory and file
#              listboxes are used to select directories and files.  Entries
#              also exist so that input can be typed in.  Cancel, select, and
#              filter buttons are also created.  The name of the file selected
#              will be returned if the `Cancel' button is not pressed (Null
#              string if it is pressed).
#
# Use       -- Simply call FSBox and pass it a title, a pattern, and the
#              operation to be performed (`loading' or `saving').
#              The title parameter will be made the title of the FSB window.
#              The pattern will be used to filter files.  The operation
#              parameter is used to enable some error checking that needs to 
#              occur when files are being loaded.  For example, a beep will
#              be sounded if a loading operation is performed on a file that
#              does not exist.
# 


# Procedure - FSBox_ScanDir - This procedure lists the contents
#             of a directory whenever a directory name is double
#             clicked upon in the directory list box or the enter
#             key is pressed while the directory entry is active.
#
# Parameters - dir - The text in the directory entry
#              pat - The text in the filter entry
#
#
# return value - None
# 


proc XParflow::FSBox_ScanDir {dir pat} {

   # Make sure the directory exists
 
   if {![file isdirectory $dir] || ![file readable $dir]} {

      bell
      return
 
   }

   # get a list of files/directories in the directory

   set dirlist [exec ls -a $dir]

   # Drop the /.. or /. from the directory string
   # Ex: /usr/local/bin/.. is equivalent to /usr/local and
   #     /usr/local/bin/.  is equivalent to /usr/local/bin 

   if {[string match {*/..} $dir]} {
      set dir [file dirname [file dirname $dir]]
   } elseif {[string match {*/.} $dir]} {
      set dir [file dirname $dir] 
   }

   .fsb.dirFrame.listBox delete 0 [.fsb.dirFrame.listBox size]

   .fsb.fileFrame.listBox delete 0 [.fsb.fileFrame.listBox size]

   # Make sure the directory string ends with a slash

   if {[string index $dir [expr [string length $dir]-1]] != "/" } {
      
      set dir $dir/

   } 

   # Place the directory list elements in the proper listbox.  This means
   # directories should go in the directory listbox and the files should
   # go in the file listbox. Stale links are not included because they
   # are neither a directory or a file.

   foreach i $dirlist {
   
      if {[file isdirectory $dir$i]} {
         .fsb.dirFrame.listBox insert end $dir$i
      }

      if {[file isfile $dir$i] && [string match $pat $dir$i]} {
         .fsb.fileFrame.listBox insert end $i
      }

   }

}
         

# Procedure - FSBox_SelectFile - This procedure returns the filename
#             appended to the directory string with a slash in between
#
# Parameters - dir    - The string in the directory entry
#              fName  - The string in the filename/selection entry
#
# return value - Pathname of file
#
 
proc XParflow::FSBox_SelectFile {dir fName} {

   if {[string index $dir [expr [string length $dir]-1]] != "/" } {
      
      set dir $dir/

   } 

   return $dir$fName

}


# Procedure - FSBox_UpdateFileEntry - This procedure is used to
#             place a new filename in the filename/selection entry
#
# Parameters - dir    - The directory current being viewed
#              fName  - The name of the file selected from the
#                          filename list box
#
# Variables - fsbFileName - Textvariable associated with the filename entry
#
# return value - None
#
 
proc XParflow::FSBox_UpdateFileEntry {dir fName} {
   set XParflow::fsbFileName [FSBox_SelectFile $dir $fName]
}


# Procedure - FSBox_SelectDir - This procedure is used to change
#             the directory path given in the directory entry to a new
#             one that has been selected from the directory list box.
#
# Parameters dir - The name of the new directory chosen
#            pat - The pattern used to filter filenames 
#
# Variables - fsbDirectory - Textvariable associated with the directory
#                                   entry
#
# return value - None
#

proc XParflow::FSBox_SelectDir {dir pat} {

   set XParflow::fsbDirectory [.fsb.selFrame.dirEntry index end]

   if {[string match {*/..} $dir]} {
      set XParflow::fsbDirectory [file dirname [file dirname $dir]]
   } elseif {[string match {*/.} $dir]} {
      set XParflow::fsbDirectory [file dirname $dir] 
   } else {
      set XParflow::fsbDirectory $dir
   }

   FSBox_ScanDir $dir $pat

}


# Procedure FSBox_CreateWidgets - This procedure is used to create all
#           of the widgets to be used in the fsb
#
# Parameters - initDir - The first directory to be displayed when the
#                        fsb appears
#              initPat - The first pattern to be displayed when the 
#                        fsb appears
#
# Variables - fsbDirectory - Textvariable associated with the directory
#                                  entry
#             fsbPattern   - Textvariable associated with the filter entry
#             fsbFileName  - Textvariable associated with the filename
#                                   entry
#
# return value - None
#

proc XParflow::FSBox_CreateWidgets {initDir initPat} {

   # Create a frame for holding the entry and listbox widgets

   frame .fsb.selFrame -relief raised -borderwidth 1

   label .fsb.selFrame.dirLabel -text "Current Directory" -anchor w
   entry .fsb.selFrame.dirEntry -relief sunken -borderwidth 2 \
                                -textvariable XParflow::fsbDirectory

   # Initialize the text in the directory entry

   set XParflow::fsbDirectory $initDir

   # Create an entry widget for entering the file filter

   label .fsb.selFrame.filtLabel -text "Filter" -anchor w
   entry .fsb.selFrame.filtEntry -relief sunken -borderwidth 2 \
                                 -textvariable XParflow::fsbPattern

   #initialize the text in the file entry

   set XParflow::fsbPattern $initPat

   # Create a frame in which two other frames will reside

   frame .fsb.selFrame.listFrame -relief flat

   # Create a frame for the directory list box and its scroll bars
   # as well as create the list box and scroll bars

   frame .fsb.dirFrame -relief flat

   label .fsb.dirFrame.label -text "Directories" -anchor w
   listbox .fsb.dirFrame.listBox -yscrollcommand ".fsb.dirFrame.yScroll set"\
                                 -xscrollcommand ".fsb.dirFrame.xScroll set"
   scrollbar .fsb.dirFrame.yScroll -command ".fsb.dirFrame.listBox yview"
   scrollbar .fsb.dirFrame.xScroll -command ".fsb.dirFrame.listBox xview" \
                                   -orient horizontal

   # Create a frame for the filenames list box and its scroll bars
   # as well as create the list box and scroll bars

   frame .fsb.fileFrame -relief flat

   label .fsb.fileFrame.label -text "Files" -anchor w
   listbox .fsb.fileFrame.listBox -yscrollcommand ".fsb.fileFrame.yScroll set" \
                                  -xscrollcommand ".fsb.fileFrame.xScroll set"
   scrollbar .fsb.fileFrame.yScroll -command ".fsb.fileFrame.listBox yview"
   scrollbar .fsb.fileFrame.xScroll -command ".fsb.fileFrame.listBox xview" \
                                    -orient horizontal

   # Create a file selection label and entry
   # Initialize the text in the selection entry

   label .fsb.selFrame.fileLabel -text "Selection" -anchor w
   entry .fsb.selFrame.fileEntry -relief sunken -borderwidth 2 \
                                 -textvariable XParflow::fsbFileName

   set XParflow::fsbFileName $initDir

   # Create a button frame and buttons

   frame .fsb.btnFrame -relief raised -borderwidth 1

   button .fsb.btnFrame.select -text Select -command {

      if {[file isfile $XParflow::fsbFileName] || ($XParflow::fsbOperation == "saving")} {

         set XParflow::fsbDone true

      } else {

         bell

      }

   }


   button .fsb.btnFrame.filter -text Filter \
                      -command {FSBox_ScanDir $XParflow::fsbDirectory $XParflow::fsbPattern} 

   button .fsb.btnFrame.cancel -text Cancel -command {

      set XParflow::fsbCancel true
      set XParflow::fsbDone true

   }

}


# Procedure FSBox_BindWidgets - This procedure creates the necessary bindings
#           so that the user can interact with entry and list widgets
#
# Parameters - None
#
# Return Value - None
#

proc XParflow::FSBox_BindWidgets {} {

   # Bind the directory entry so that when the enter key is pressed while
   # the entry is active the directories contents can be displayed.

   bind  .fsb.selFrame.dirEntry <KeyPress-Return> \
                                {FSBox_ScanDir $XParflow::fsbDirectory $XParflow::fsbPattern}

   # Bind the entry to the return key

   bind  .fsb.selFrame.filtEntry <KeyPress-Return> \
                                 {FSBox_ScanDir $XParflow::fsbDirectory $XParflow::fsbPattern}

   # Bind the list box selection to the first mouse button

   bind .fsb.dirFrame.listBox <Double-Button-1> {
      set index [.fsb.dirFrame.listBox curselection]
      FSBox_SelectDir [.fsb.dirFrame.listBox get $index] $XParflow::fsbPattern
   }

   # Bind the file list box selection to the first mouse button

   bind .fsb.fileFrame.listBox <ButtonRelease-1> {

      if {[set index [.fsb.fileFrame.listBox curselection]] != ""} {
         FSBox_UpdateFileEntry $XParflow::fsbDirectory [.fsb.fileFrame.listBox get $index]
      }

   }

   bind .fsb.fileFrame.listBox <Double-Button-1> {

      if {[set index [.fsb.fileFrame.listBox curselection]] != ""} {
         FSBox_UpdateFileEntry $XParflow::fsbDirectory [.fsb.fileFrame.listBox get $index]
         set XParflow::fsbDone true
      }

   }

   # Bind the filename entry to the return key.  If the file exists, then
   # it will be returned

   bind  .fsb.selFrame.fileEntry <KeyPress-Return> {

      # Make sure the file exists or if it does not, make sure that
      # the filename represents a file to be saved.

      if {[file isfile $XParflow::fsbFileName] || ($XParflow::fsbOperation == "saving")} {

         set XParflow::fsbDone true

      } else {

         bell

      }

   }

}


# Procedure FSBox_ManageWidgets - This procedure is used to manage each of
#           the widgets with the packer.
#
# Parameters - None
#
# Return value - None
#

proc XParflow::FSBox_ManageWidgets {} {

   # Manage the Directory label and entry

   pack .fsb.selFrame -fill both -expand 1

   pack .fsb.selFrame.dirLabel -side top -anchor w -padx 5 
   pack .fsb.selFrame.dirEntry -side top -anchor w -padx 5 -fill x

   # Manage the filter label and entry

   pack .fsb.selFrame.filtLabel -side top -anchor w -padx 5 

   pack .fsb.selFrame.filtEntry -side top -anchor w -padx 5 -fill x

   # Manage the frames, labels, list boxes, and scroll bars

   pack .fsb.selFrame.listFrame -side top -pady 10 -fill both -expand 1

   pack .fsb.dirFrame -in .fsb.selFrame.listFrame -side left -padx 5 \
                                                  -fill both -expand 1
   pack .fsb.dirFrame.label -side top -anchor w -fill x
   pack .fsb.dirFrame.yScroll -side right -fill y
   pack .fsb.dirFrame.xScroll -side bottom -fill x
   pack .fsb.dirFrame.listBox -side top -fill both -expand 1

   pack .fsb.fileFrame -in .fsb.selFrame.listFrame -side right -padx 5 \
                                                   -fill both -expand 1
   pack .fsb.fileFrame.label -side top -anchor w -fill x
   pack .fsb.fileFrame.yScroll -side right -fill y
   pack .fsb.fileFrame.xScroll -side bottom -fill x
   pack .fsb.fileFrame.listBox -side top -fill both -expand 1

   # Manage the file selection entry

   pack .fsb.selFrame.fileLabel -side top -anchor w -padx 5 
   pack .fsb.selFrame.fileEntry -side top -anchor w -padx 5 -pady 5 -fill x

   # Manage the buttons.  Keep them centered

   pack .fsb.btnFrame -side bottom -ipady 10 -fill x
   pack .fsb.btnFrame.select .fsb.btnFrame.filter .fsb.btnFrame.cancel \
                                                  -side left -expand 1

}


# Procedure FSBox_Init - This procedure creates the fsb window if it does
#                        not already exist.  Once the fsb is created, it
#                        must be destroyed explicitly by the user.  This
#                        allows all of the filenames, directories, and the
#                        filter to be saved once `Cancel' or `Select' are
#                        chosen.  This way when the fsb is reopened, the
#                        information in the listboxes and entries is the
#                        same as when the user closed the listbox.
#
# Parameters - title   - The title to be placed at the top of the window.
#              initPat - The pattern to be placed in the pattern entry
#                        when FSBox is called
#              initDir - The directory the fsb will view when the fsb is
#                        activated.
#              op      - The operation the fsb is being used for - either
#                        `loading' or `saving'
#
# Variables  - fsbOperation  - used to store op globally
#              XParflow::fsbDirectory  - Textvariable associated with the directory
#                                     entry
#              fsbPattern    - Textvariable associated with the filter
#                                     entry
#              fsbCancel     - True when the cancel button has been
#                                     pressed
#              fsbDone       - True when a file has been selected or
#                                     the cancel button has been pressed
#
  
proc XParflow::FSBox_Init {title initPat initDir op} {

   set XParflow::fsbOperation $op

   # If no directory was given, then use
   # the current directory as default.

   if {$initDir == ""} {

      if {[info exists XParflow::fsbDirectory]} {

         set initDir $XParflow::fsbDirectory 

      } else {

         set initDir [pwd]

      }

   }


   # If the toplevel window for the fsb has already been created, then
   # do not initialize it again.  Just change the pattern and the title.
   # If it already has been created, then its state is currently iconified
   # and will just need to be made visible again.

   if {[catch "wm state .fsb"]} {

      # Create top level window for fsb

      toplevel .fsb

      FSBox_CreateWidgets $initDir $initPat
      FSBox_BindWidgets
      FSBox_ManageWidgets

      # Initialize the list boxes

      FSBox_ScanDir $initDir $initPat

      # We won't be needing these anymore, so
      # we can free up some space by destroying
      # them.

      rename FSBox_CreateWidgets {}
      rename FSBox_BindWidgets   {}
      rename FSBox_ManageWidgets {}

   # The fsb window already exists so we just need to refresh the
   # directory and file listboxes.

   } else {

      set XParflow::fsbPattern $initPat
      FSBox_ScanDir $initDir $initPat

   }

   set XParflow::fsbCancel false
   set XParflow::fsbDone false

   wm title .fsb $title

   # Center the fsb

   wm withdraw .fsb
   update idletasks
   set x [expr [winfo screenwidth .fsb]/2 - [winfo reqwidth .fsb]/2 \
           - [winfo vrootx [winfo parent .fsb]]]
   set y [expr [winfo screenheight .fsb]/2 - [winfo reqheight .fsb]/2 \
           - [winfo vrooty [winfo parent .fsb]]]
   wm geom .fsb +$x+$y
   wm deiconify .fsb 

}


# Procedure FSBox - This procedure is used to initialize the file
#           selection diaglog and place tk in an event loop until
#           a file is chosen
#
# Parameters   title   - The title to be placed at the top of the window
#              filters - The filters
#              op      - The operation (`saving' or `loading') the fsb is to
#                        be used for
#
# Variables - fsbFileName  - Texvariable associated with the filename
#                                   entry
#             fsbCancel    - True when the cancel button has been pressed
#             fsbDone      - True when the when a file has been selected or
#                                   the cancel button has been pressed
#
# Return Value - None
#

proc XParflow::FSBox {title filters op {initDir ""}} {

    global tk_version
    
    # If we are working with newer version of tk then use
    # builtin file selection boxes.

    # Builtin selection boxes are broken some problem with colormaps
    # Revert to local file selection boxes

    if { [expr $tk_version >= 4.2] } {

	if {[info exists initDir]} {
	} {
	    set initDir [pwd]
	}

 	if { $op == "loading"} {
	    set fName [tk_getOpenFile -title $title -filetypes $filters ]
 	} {
	    set fName [tk_getSaveFile -title $title -filetypes $filters ]
 	}
	
	#         set fsbFileName [file dirname $fName]

        return $fName
    } {
	FSBox_Init $title {*} $initDir $op
	
	# Set up the event loop and withdraw
	# the fsb when a file has been chosen
	
        update idletasks
        grab .fsb
        tkwait variable XParflow::fsbDone
        grab release .fsb
        catch "selection clear [selection own]"
        wm withdraw .fsb
	
        # Return a null string if no file was selected
	
        if {$XParflow::fsbCancel == "true"} {
	    
	    return ""
	    
        }
	
        set fName [string trimright $XParflow::fsbFileName]
        set XParflow::fsbFileName [file dirname $XParflow::fsbFileName]
	
        return $fName
    }
}

