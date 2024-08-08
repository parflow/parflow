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

# Procedure - NewGrid - This procedure is called when the user wishes to
#             create a new grid.  A window is created and the user enters
#             the dimensions, origin, and intervals between points.  A
#             new grid and a data set with all zero values at each data
#             point is created.
#
# Parameters - None
#
# Variables - global done - Set to true when the window is to be closed
#
# Return value - None

proc XParflow::NewGrid {} {

   #--------------------------------------------------
   # Create a toplevel window for the grid description
   # to exist in.
   #--------------------------------------------------

   toplevel .nGrid
   wm resizable .nGrid false false
   wm title .nGrid "Create New Grid"

   NewGrid_CreateWidgets
   NewGrid_ManageWidgets
   CenterWindow .nGrid

   #-------------------------------------------------------
   # Grab the focus and keep it on the window until
   # the user has selected the `Create' or `Cancel' button.
   #-------------------------------------------------------

   GrabFocus .nGrid
   destroy .nGrid

}


# Procedure NewGrid_CreateWidgets - This procedure creates the widgets
#           used to let the user enter a description of a new grid.
#
# Parameters - None
#
# Variables - i - A loop index used to insert data set names and labels
#                 into a dropdown listbox meta widget
#
# Return value - None

proc XParflow::NewGrid_CreateWidgets {} {

   #-------------------------------------------------
   # Create the frame for the grid description
   #-------------------------------------------------

   GridFrame .nGrid.topFrame

   #-------------------------------------------------
   # Insert a dropdown list widget into the grid
   # description frame.  This will allow users to
   # duplicate an existing grid.
   #-------------------------------------------------

   frame .nGrid.midFrame -relief raised -borderwidth 1
   metaEListBox .nGrid.dropDown -label "Duplicate Grid :"\
                                -command {NewGrid_FillEntries} \
                                -entrystate disabled

   .nGrid.dropDown configure -relief groove -borderwidth 4
 
 
   #---------------------------------------------------
   # For each data set that has been loaded, we need to
   # insert its name into the dropdown list widget.
   #---------------------------------------------------
   
   foreach i [pfgetlist] {

      metaEListBox .nGrid.dropDown insert end [format "%s  %s" \
                                               [lindex $i 0] [lindex $i 1]]

   }
  
   #---------------------------------------------------
   # Create an entry for the label of the new grid and
   # data set to be created.
   #---------------------------------------------------

   label .nGrid.midFrame.newLabel -text "New Label :"
   entry .nGrid.labelEntry 

   #---------------------------------------------------
   # Create `Create' and `Cancel' buttons
   #---------------------------------------------------
   
   frame .nGrid.botFrame -relief raised -borderwidth 1   
   button .nGrid.create -text Create -command {NewGrid_CreateDataset}
   button .nGrid.cancel -text Cancel -command {set XParflow::done true}
 
}


# Procedure NewGrid_ManageWidgets - This procedure manages the widgets
#           created in CreateWidgets.  The grid description is packed
#           on top of the dropdown list box and the label of the new
#           data set and grid.  these are packed on top of the button
#           frame.
#
# Parameters - None
#
# Variables - None
#
# Return value - None

proc XParflow::NewGrid_ManageWidgets {} {

   pack .nGrid.topFrame -side top -fill x -ipadx 10 -ipady 5
   pack .nGrid.dropDown -in .nGrid.topFrame -ipady 4

   pack .nGrid.midFrame -fill x -ipady 4
   pack .nGrid.midFrame.newLabel -side left -padx 10 
   pack .nGrid.labelEntry -in .nGrid.midFrame -side left -padx 10 -fill x \
                                                                  -expand 1

   pack .nGrid.create -in .nGrid.botFrame -side left -fill x -padx 10 -expand 1
   pack .nGrid.cancel -in .nGrid.botFrame -side left -fill x -padx 10 -expand 1
   pack .nGrid.botFrame -fill x -ipady 10

}


# Procedure NewGrid_FillEntries - This procedure takes the name of
#           whatever data set was selected with the dropdown list box and
#           displays its grid description in various entries.
#
# Parameters - None
#
# Variables - None
#
# Return value - None

proc XParflow::NewGrid_FillEntries {} {

   #-------------------------------------------------
   # Get the data set name, label, and grid
   #-------------------------------------------------

   set dataSet [lindex [.nGrid.dropDown.entry get] 0]
   set label [lindex [.nGrid.dropDown.entry get] 1]
   set grid [pfgetgrid $dataSet]
   
   GridFrame .nGrid.topFrame -grid $grid

   #-------------------------------------------------
   # Clear the entries
   #-------------------------------------------------

   .nGrid.labelEntry delete 0 end

   #-------------------------------------------------
   # Fill in the label entry
   #-------------------------------------------------

   .nGrid.labelEntry insert 0 $label 
   
}


# Procedure NewGrid_CreateDataset - This procedure is used to create the
#           new grid and data set once the `Create' button has been
#           pressed.
#
# Parameters - None
#
# Variables - global done - This is true once the `Create' button has
#                           been pressed.  The toplevel window is then
#                           destroyed.
#
# Return value - None

proc XParflow::NewGrid_CreateDataset {} {

   #-------------------------------------------------------------
   # The values entered into the entries pertaining to the number
   # of points along each axis must be integers.
   #-------------------------------------------------------------

   set num [scan "[.nGrid.topFrame.entnx get] [.nGrid.topFrame.entny get] 
                  [.nGrid.topFrame.entnz get]" "%d %d %d%s" nx ny nz junk]

   if {$num != 3} {

      ErrorDialog "The values of the numbers of points must be integers."
      return

   }

   #-------------------------------------------------------------
   # The values entered into the entries pertaining to the origin
   # must be doubles.
   #-------------------------------------------------------------

   set num [scan "[.nGrid.topFrame.entx get] [.nGrid.topFrame.enty get]
                  [.nGrid.topFrame.entz get]" "%lf %lf %lf%s" x y z junk]

   if {$num != 3} {

      ErrorDialog "The values of the coordinates of the origin must be floating point numbers."
      return

   }

   #-------------------------------------------------------------
   # The values entered into the entries pertaining to the
   # intervals along each axis must be doubles.
   #-------------------------------------------------------------

   set num [scan "[.nGrid.topFrame.entdx get] [.nGrid.topFrame.entdy get] 
                  [.nGrid.topFrame.entdz get]" "%lf %lf %lf" dx dy dz]
      
   if {$num != 3} {

      ErrorDialog "The values of the coordinate intervals must be floating point numbers."
      return

   }
    
   #-------------------------------------------------------------
   # An error can only occur here if there is not enough memory
   # to store the new grid and the new data set
   #-------------------------------------------------------------

   set code [catch {pfnewgrid "$nx $ny $nz" "$x $y $z" "$dx $dy $dz" \
                    [.nGrid.labelEntry get]} dataset]

   if {$code} {

      ErrorDialog "Memory could not be allocated for the new grid"
      return

   } else {

      MainWin_InsertDataSets $dataset
      set XParflow::done true
   }

}
