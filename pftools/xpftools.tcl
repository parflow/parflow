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

package provide xparflow 1.0

namespace eval XParflow {
    variable fsbDirectory
    variable fsbPattern
    variable fsbFileName
    variable fsbOperation
    variable fsbCancel
    variable fsbDone

    variable dataDisp

    variable done
    variable statEnt
    variable prevGrab

    variable help

    set XParflow::help(mainWin) {"The main window contains the main menu bar and a listbox.  The menu bar is used to load and save data, change grid types, and perform various operations on data sets.  Data sets may be examined by double-clicking on their names given in the main window list box."}
    
    set XParflow::help(dataMenu) {"The data menu allows you to load and save data from and to various file formats respectively. Also, grids may be created, data sets may be deleted, set labels may be changed."}  
    
    set XParflow::help(gridMenu) {"The grid menu allows you to change how PFTools interprets the grids that the data sets are a part of.  The default value is `cell centered', but can be changed by clicking on the `vertex centered' menu button."}
    
    set XParflow::help(funcMenu) {"The function menu allows you to perform various mathematical operations upon the data sets that have been loaded in or created thus far.  Selecting a function to perform will bring up a function window that you must use to specify input arguments."}
    
    set XParflow::help(velMag) {"This function computes the velocity magnitude given by X, Y, and Z components.  You must use the `Input Data' entries to select data sets for each component.  After this has been done, press `Compute Function' to obtain a data set that is the velocity magnitude."}
    
    set XParflow::help(cVel) {"This function computes the velocity in the cells from a data set that is the conductivity and another that is the pressure head.  After these data sets have been selected, press the `Compute Function' button to obtain the X, Y, and Z components of the velocity.  The component data may be viewed by selecting the corresponding radio buttons next to each component."}
    
    set XParflow::help(vVel) {"This function takes conductivity and pressure head data sets and computes the velocity at the grid vertices.  After these data sets have been selected, press the `Compute Function' button to obtain X, Y, and Z components of the velocity.  The component data may be viewed by selecting the corresponding radio buttons next to each component."}
    
    set XParflow::help(flux) {"This function computes the flux from conductivity and hydraulic head data sets.  After these data sets have been selected, press the `Compute Function' button to obtain the flux."}
    
    set XParflow::help(hHead) {"This function computes a hydraulic head from a pressure head.  After these data sets have been selected, press the `Compute Function' button to obtain the hydraulic head.  The hydraulic head may be viewed in the display at the bottom."}
    
    set XParflow::help(pHead) {"This function computes a pressure head from a hydraulic head.  After these data sets have been selected, press the `Compute Function' button to obtain the pressure head.  The pressure head may be viewd in the display at the bottom."}
    
    set XParflow::help(axpy) {"This function computes the function `y = alpha * x + y' where x and y are data sets and alpha is a scalar.  Once x, y, and alpha have been selected, the data set y will be overwritten and displayed in the data display at the bottom of the window."}
    
    set XParflow::help(diff) {"This function computes the differences between two data sets A and B.  Only the coordinates whose number of agreeing significant digits differs by more than the number specified will be considered.  Also, differences not greater than absolute zero will not be considered either.  After pressing `Compute Function' the differences, the minimum number of agreeing significant digits,  and the maximum absolute difference will be displayed."}
    
    set XParflow::help(mDiff) {"This function computes the differences between two data sets A and B with the purpose of finding the minimum number of agreeing significant digits.  The maximum absolute difference is also computed."}

    #
    # Exporting way too many functions but on initial pass did not
    # want to attempt to fix all references to these names
    #
    # Need to come back and fix all this up.  Probably only need to
    # export the main window routine

    namespace export MainWin_CreateCascadingMenus
    namespace export MainWin_CreateMenus
    namespace export MainWin_CreateListBox
    namespace export MainWin_CreateBindings
    namespace export MainWin_Manage
    namespace export MainWin_Init
    namespace export MainWin_UpdateMenuBar
    namespace export MainWin_InsertDataSets
    namespace export MainWin_UpdateList
    namespace export MainWin_DeleteDataDialog
    namespace export MainWin_CreateInfoWindow
    namespace export MainWin_ChangeLabelDialog
    namespace export CenterWindow
    namespace export metaEListBox
    namespace export metaEListBox_InitWidgetData
    namespace export metaEListBox_ParseWidgetArgs
    namespace export metaEListBox_ConstructWidgetFrame
    namespace export metaEListBox_SetWidgetBindings
    namespace export metaEListBox_ButtonCommand
    namespace export metaEListBox_SelectEntryEvent
    namespace export metaEntry
    namespace export metaEntry_InitWidgetData
    namespace export metaEntry_ParseWidgetArgs
    namespace export metaEntry_ConstructWidgetFrame
    namespace export FSBox_ScanDir
    namespace export FSBox_SelectFile
    namespace export FSBox_UpdateFileEntry
    namespace export FSBox_SelectDir
    namespace export FSBox_CreateWidgets
    namespace export FSBox_BindWidgets
    namespace export FSBox_ManageWidgets
    namespace export FSBox_Init
    namespace export FSBox
    namespace export DataDisplay
    namespace export DataDisplay_ParseWidgetArgs
    namespace export DataDisplay_DestroyDisplay
    namespace export DataDisplay_InitWidgetData
    namespace export DataDisplay_ConstructWidgetFrame
    namespace export DataDisplay_SetWidgetBindings
    namespace export DataDisplay_ManageWidgets
    namespace export DataDisplay_UpdateEntries
    namespace export DataDisplay_FillDisplay
    namespace export DataDisplay_IncrCoordinate
    namespace export DataDisplay_DecrCoordinate
    namespace export DataDisplay_DownButton
    namespace export DataDisplay_UpButton
    namespace export DataDisplay_PgDownButton
    namespace export DataDisplay_PgUpButton
    namespace export DataDisplay_ViewCoordinate
    namespace export DiffDisplay
    namespace export DiffDisplay_ParseWidgetArgs
    namespace export DiffDisplay_SetWidgetBindings
    namespace export DiffDisplay_FillDisplay
    namespace export DiffDisplay_DownButton
    namespace export DiffDisplay_UpButton
    namespace export DiffDisplay_PgDownButton
    namespace export DiffDisplay_PgUpButton
    namespace export DiffDisplay_ViewCoordinate
    namespace export VelMagnitude
    namespace export VelMagnitude_Compute
    namespace export CVel
    namespace export CVel_Compute
    namespace export VVel
    namespace export VVel_Compute
    namespace export Flux
    namespace export Flux_Compute
    namespace export HHead
    namespace export HHead_Compute
    namespace export PHead
    namespace export PHead_Compute
    namespace export Axpy
    namespace export Axpy_Compute
    namespace export Diff
    namespace export Diff_Compute
    namespace export MDiff
    namespace export MDiff_Compute
    namespace export CreateFunctionWindow
    namespace export CreateStatDisplay
    namespace export CreateDataDisplay
    namespace export CreateDiffDisplay
    namespace export FunctionEventLoop
    namespace export CreateIOFrame
    namespace export CreateInputFrame
    namespace export CreateOutputFrame
    namespace export FillInputSelectionBox
    namespace export CreateFunctionButtons
    namespace export GetInputParameters
    namespace export FillOutputEntries
    namespace export UpdateStatDisplay
    namespace export UpdateDataDisplay
    namespace export UpdateDiffDisplay
    namespace export GrabFocus
    namespace export ErrorDialog
    namespace export GridFrame
    namespace export GridFrame_ParseWidgetArgs
    namespace export GridFrame_ConstructWidgetFrame
    namespace export GridFrame_ManageWidgets
    namespace export GridFrame_UpdateEntries
    namespace export GridFrame_SetEntryStates
    namespace export Help
    namespace export InfoWindow
    namespace export InfoWindow_InitWidgetData
    namespace export InfoWindow_ConstructTopLevel
    namespace export InfoWindow_ManageTopLevel
    namespace export InfoWindow_DestroyWindow
    namespace export NewGrid
    namespace export NewGrid_CreateWidgets
    namespace export NewGrid_ManageWidgets
    namespace export NewGrid_FillEntries
    namespace export NewGrid_CreateDataset
    namespace export MainWin_CreateCascadingMenus
    namespace export MainWin_CreateMenus
    namespace export MainWin_CreateListBox
    namespace export MainWin_CreateBindings
    namespace export MainWin_Manage
    namespace export MainWin_Init
    namespace export MainWin_UpdateMenuBar
    namespace export MainWin_InsertDataSets
    namespace export MainWin_UpdateList
    namespace export MainWin_DeleteDataDialog
    namespace export MainWin_CreateInfoWindow
    namespace export MainWin_ChangeLabelDialog
    namespace export CenterWindow
    namespace export GetPFFileName
    namespace export LoadPF
    namespace export SavePF
    namespace export ScanDSNum
    namespace export GetSDSNum
    namespace export LoadSDS
    namespace export GetHDFFileName
    namespace export SaveSDS
    namespace export StatDisplay
    namespace export StatDisplay_ParseWidgetArgs
    namespace export StatDisplay_ConstructWidgetFrame
    namespace export StatDisplay_ManageWidgets
    namespace export StatDisplay_UpdateEntries
    namespace export StatDisplay_DestroyDisplay

    namespace export ThreeSlices
    namespace export ThreeSlicesDisplay_ConstructWidgetFrame
    namespace export ThreeSlices_Compute
}

#-----------------------------------------------------------------------------
# These are procedure that are used to build the main window that will
# contain the main menus as well as the main list box.  The main list box
# contains the names and labels of data sets that have been loaded by the
# user.
#-----------------------------------------------------------------------------

# Procedure MainWin_CreateCascadingMenus -This procedure creates the cascading
#           menus which will be attached to the `Read ParFlow', `Save
#           ParFlow', and `Save SDS' menu options.
#
# Parameters - None
#
# Variables - None
#
# Return value - None
#
 
proc XParflow::MainWin_CreateCascadingMenus {} {

    #------------------------------------------------------------
    # Create the cascading menu for loading ParFlow format files
    #------------------------------------------------------------
    
    menu .main.mbar.data.menu.loadpf
    
    .main.mbar.data.menu.loadpf add command -label {ParFlow Binary}           \
	    -command {LoadPF pfb}
    .main.mbar.data.menu.loadpf add command -label {ParFlow Scattered Binary} \
	    -command {LoadPF pfsb}
    .main.mbar.data.menu.loadpf add command -label {Simple Binary}            \
	    -command {LoadPF sb}
    .main.mbar.data.menu.loadpf add command -label {Simple ASCII}             \
	    -command {LoadPF sa}
    .main.mbar.data.menu.loadpf add command -label {Real Scattered ASCII}     \
	    -command {LoadPF rsa}
    .main.mbar.data.menu.loadpf add command -label {AVS Field}                \
	    -command {LoadPF fld}
    
    #------------------------------------------------------------
    # Create the cascading menu for saving ParFlow format files
    #------------------------------------------------------------
    
    menu .main.mbar.data.menu.savepf
    
    .main.mbar.data.menu.savepf add command -label {ParFlow Binary}           \
	    -command {SavePF pfb}
    .main.mbar.data.menu.savepf add command -label {Simple Binary}            \
	    -command {SavePF sb}
    .main.mbar.data.menu.savepf add command -label {Simple ASCII}             \
	    -command {SavePF sa}
    .main.mbar.data.menu.savepf add command -label {AVS Field}                \
	    -command {SavePF fld}
    
    #------------------------------------------------------------
    # Create the cascading menu for saving HDF format files
    #------------------------------------------------------------

    menu .main.mbar.data.menu.savesds
    
    .main.mbar.data.menu.savesds add command -label {Float 64}                \
	    -command {SaveSDS float64}
    .main.mbar.data.menu.savesds add command -label {Float 32}                \
	    -command {SaveSDS float32}
    .main.mbar.data.menu.savesds add command -label {Int 32}                  \
	    -command {SaveSDS int32}
   .main.mbar.data.menu.savesds add command -label {UInt 32}                 \
	   -command {SaveSDS uint32}
   .main.mbar.data.menu.savesds add command -label {Int 16}                  \
	   -command {SaveSDS int16}
   .main.mbar.data.menu.savesds add command -label {UInt 16}                 \
	   -command {SaveSDS uint16}
   .main.mbar.data.menu.savesds add command -label {Int 8}                   \
	   -command {SaveSDS int8}
   .main.mbar.data.menu.savesds add command -label {UInt 8}                  \
	   -command {SaveSDS uint8}
   
}
# Procedure MainWin_CreateMenus - This procedure creates the menu bar and the
#           main menus to be displayed in the main window.
# 
# Parameters - None
#
# Variables - None
#
# Return value - None
#           

proc XParflow::MainWin_CreateMenus {} { 
    
    #--------------------------------------------------
    # Create frame for main window menu bar
    #--------------------------------------------------
    
    frame .main.mbar -relief raised -borderwidth 2
    
    #--------------------------------------------------
    # Create buttons for the menu bar
    #--------------------------------------------------
    
    menubutton .main.mbar.data -text Data -underline 0 \
	    -menu .main.mbar.data.menu
    menubutton .main.mbar.grid -text Grid -underline 0 \
	    -menu .main.mbar.grid.menu
    menubutton .main.mbar.funcs -text Functions -underline 0 \
	    -menu .main.mbar.funcs.menu
    if { $XParflow::haveVTK } {
	menubutton .main.mbar.viz -text Visualization -underline 0 \
		-menu .main.mbar.viz.menu 
    }
    menubutton .main.mbar.help -text Help -underline 0 \
	    -menu .main.mbar.help.menu
    
    #--------------------------------------------------
    # Create data menu for the data menu button
    #--------------------------------------------------
    
    menu .main.mbar.data.menu
    
    .main.mbar.data.menu add cascade -label {Load ParFlow Data}               \
	    -menu .main.mbar.data.menu.loadpf
    .main.mbar.data.menu add command -label {Load SDS} -command {LoadSDS}
    .main.mbar.data.menu add command -label {Create New Grid}                 \
	    -command {NewGrid}
    .main.mbar.data.menu add command -label {ReLoadAll}                 \
	    -command {pfreloadall}
    .main.mbar.data.menu add separator
    .main.mbar.data.menu add cascade -label {Save ParFlow Data}               \
	    -menu .main.mbar.data.menu.savepf
    .main.mbar.data.menu add cascade -label {Save SDS}                        \
	    -menu .main.mbar.data.menu.savesds
    .main.mbar.data.menu add separator
    .main.mbar.data.menu add command -label {Update List}                     \
	    -command {MainWin_UpdateList}
    .main.mbar.data.menu add command -label {Change Label}                    \
	    -command {MainWin_ChangeLabelDialog}
    .main.mbar.data.menu add command -label {Delete Data Set}                 \
	    -command {MainWin_DeleteDataDialog}
    .main.mbar.data.menu add separator
    .main.mbar.data.menu add command -label {Quit} -command exit
    
    
    #--------------------------------------------------
    # Create grid menu for the grid menu button
    #--------------------------------------------------
    
    menu .main.mbar.grid.menu
    
    .main.mbar.grid.menu add radiobutton -label {Cell Centered} \
	    -variable gridType -value cell \
	    -command {pfgridtype cell}
    .main.mbar.grid.menu add radiobutton -label {Vertex Centered} \
	    -variable gridType -value vertex \
	    -command {pfgridtype vertex}
    
    #--------------------------------------------------
    # Create function menu for the function menu button
    #--------------------------------------------------
    
    menu .main.mbar.funcs.menu
    
    .main.mbar.funcs.menu add command -label {Compute Velocity Magnitude}   \
	    -command {VelMagnitude}
    .main.mbar.funcs.menu add command -label {Compute Velocity in Cells}    \
	    -command {CVel}
    .main.mbar.funcs.menu add command -label {Compute Velocity at Vertices} \
	    -command {VVel}
    .main.mbar.funcs.menu add separator
    .main.mbar.funcs.menu add command -label {Compute Differences}          \
	    -command {Diff}
   .main.mbar.funcs.menu add command -label {Compute Minimum #Sig. Digs.}  \
	   -command {MDiff}
   .main.mbar.funcs.menu add separator
   .main.mbar.funcs.menu add command -label {Compute Flux}                 \
	   -command {Flux}
   .main.mbar.funcs.menu add command -label {Compute Hydraulic Head}       \
	   -command {HHead}
   .main.mbar.funcs.menu add command -label {Compute Pressure Head}        \
	   -command {PHead}
   .main.mbar.funcs.menu add separator
   .main.mbar.funcs.menu add command -label {Compute y = alpha * x + y}    \
	   -command {Axpy}
   
   #--------------------------------------------------
   # Create function menu for the function menu button
   #--------------------------------------------------
   
   
   if { $XParflow::haveVTK } {
       menu .main.mbar.viz.menu
       .main.mbar.viz.menu add command -label {Three Slices}   \
	       -command {ThreeSlices}
   }
   
   #--------------------------------------------------
   # Create the help menu
   #--------------------------------------------------
   
   menu .main.mbar.help.menu
   
   .main.mbar.help.menu add command -label {PFTools Main Window}  \
	   -command "Help $XParflow::help(mainWin)"
   .main.mbar.help.menu add command -label {Data Menu}            \
	   -command "Help $XParflow::help(dataMenu)" 
   .main.mbar.help.menu add command -label {Grid Menu}            \
	   -command "Help $XParflow::help(gridMenu)"
   .main.mbar.help.menu add command -label {Function Menu}        \
	   -command "Help $XParflow::help(funcMenu)"
   
   MainWin_CreateCascadingMenus   
}


# Procedure - MainWin_CreateListBox - This procedure is used to create the
#             list box which holds the names and labels of data sets
#             that are loaded into memory. Scroll bars are also added
#             to view the contents of the list box.
#
# Parameters - None
#
# Variables  - None
# 
# Return value - None
# 

proc XParflow::MainWin_CreateListBox {} {
    
    #------------------------------------------------------------------
    # Create a frame for a list box
    #------------------------------------------------------------------
    
    frame .main.listframe -relief ridge -borderwidth 4
    
    label .main.listframe.label -text {Data set        Label} -anchor w
    
    #------------------------------------------------------------------
    # Create a list box that will display the names of loaded data sets
    #------------------------------------------------------------------
    
    listbox .main.listframe.listbox \
	    -yscrollcommand {.main.listframe.yscroll set} \
	    -xscrollcommand {.main.listframe.xscroll set} 
    scrollbar .main.listframe.yscroll -command {.main.listframe.listbox yview}
    scrollbar .main.listframe.xscroll -command {.main.listframe.listbox xview}\
	    -orient horizontal 
    
}


# Procedure - MainWin_CreateBindings - This procedure is used to create the
#             bindings associated with the main window.
#
# Parameters - None
#
# Variables  - global gridType - Determines how various functions operate
#                                on the grid data.
#
# Return value - None

proc XParflow::MainWin_CreateBindings {} {
    
    .main.mbar.grid.menu invoke 1
    bind .main.listframe.listbox <Double-Button-1> \
	    {InfoWindow [lindex [selection get] 0]}
}


# Procedure - MainWin_Manage - This procedure is used to manage all widgets
#             present in the main window.
#
# Parameters - None
#
# Variables  - None
#
# Return value - None
#

proc XParflow::MainWin_Manage {} {
    
    pack .main.mbar -side top -fill x
    
    if { $XParflow::haveVTK } {
	pack .main.mbar.data .main.mbar.grid .main.mbar.funcs \
		.main.mbar.viz -side left
    } {
	pack .main.mbar.data .main.mbar.grid .main.mbar.funcs \
		-side left
    }
    
    pack .main.mbar.help -side right
    tk_menuBar .main.mbar .main.mbar.data .main.mbar.grid .main.mbar.funcs \
	    .main.mbar.help
    
    pack .main.listframe -side bottom -fill both -expand 1
    pack .main.listframe.label -side top -anchor w -fill x
    pack .main.listframe.yscroll -side right -fill y
    pack .main.listframe.xscroll -side bottom -fill x
    pack .main.listframe.listbox -side top -fill both -expand 1
    
}

# Procedure MainWin_Init - This procedure calls other procedures
#           which create the user interface for the main window.
#
# Parameters - None
#
# Local Variables - None
#
# Return value - None
#

proc XParflow::MainWin_Init {} {
    
    global env
    
    # Create a flag that will allow xpftools to determine
    # if SDS files are supported.  If not, then the commands
    # `pfreadsds' and `savesds' will not exist.
    
    global PF_HAVE_HDF
    
    if {[lsearch -exact [info commands pf*] pfloadsds] >= 0} {
	set PF_HAVE_HDF 1
    } else {
	set PF_HAVE_HDF 0
    }
    
    wm withdraw .
    toplevel .main
    wm geometry .main =640x384
    wm iconbitmap .main @$env(PARFLOW_DIR)/bin/xpft.xbm
    wm title .main "PFTools"
    
    MainWin_CreateMenus
    MainWin_CreateListBox
    MainWin_CreateBindings
    MainWin_Manage
    MainWin_UpdateMenuBar

    rename MainWin_CreateMenus    {}
    rename MainWin_CreateListBox  {}
    rename MainWin_CreateBindings {}
    rename MainWin_Manage         {}
}


#-----------------------------------------------------------------------------
# These procedures are used to maintain the main window
#-----------------------------------------------------------------------------

# Procedure - MainWin_UpdateMenuBar - This procedure is used to make
#             various menu buttons disabled and enabled depending on
#             how many data sets have been loaded into memory.
#
# Parameters - None
#
# Variables  - global PF_HAVE_HDF - True if the HDF library is installed
#
# return value - None

proc XParflow::MainWin_UpdateMenuBar {} {
    
    global PF_HAVE_HDF
    
    #----------------------------------------------------------
    # I there are no data sets loaded, then all save options
    # such as saving ParFlow and SDS's should be disabled.
    # Functions that can be performed on data sets should also
    # be disabled.
    #----------------------------------------------------------
    
    if {[.main.listframe.listbox size] == 0} {
	
	if {!$PF_HAVE_HDF} {
	    
	    .main.mbar.data.menu entryconfigure 2 -state disabled
	    
	}
	
	.main.mbar.data.menu entryconfigure 6 -state disabled
	.main.mbar.data.menu entryconfigure 7 -state disabled
	.main.mbar.data.menu entryconfigure 10 -state disabled
	.main.mbar.data.menu entryconfigure 11 -state disabled
	
	.main.mbar.funcs configure -state disabled
	
	if { $XParflow::haveVTK } {
	    .main.mbar.viz configure -state disabled
	}
	
	#---------------------------------------------------------
	# Enable all buttons that were disabled when there
	# were zero data sets loaded.
	#---------------------------------------------------------
	
    } elseif {[.main.listframe.listbox size] == 1} {
       
	if {$PF_HAVE_HDF} {
	  .main.mbar.data.menu entryconfigure 7 -state normal
	}
	
	.main.mbar.data.menu entryconfigure 6 -state normal 
	.main.mbar.data.menu entryconfigure 10 -state normal 
	.main.mbar.data.menu entryconfigure 11 -state normal 
	
	.main.mbar.funcs configure -state normal 
	
	if { $XParflow::haveVTK } {
	    .main.mbar.viz configure -state normal 
	}
	
    }
    
}
# Procedure MainWin_InsertDataSets - This procedure is used to insert
#           a data set name and its label into the main list box.
#
# Parameters - dataSet - the name of the data set to be inserted
#              index   - the index the data set is to be inserted at
#                        (defaults to end)
#            
# Variables - str - used to hold the data set name and its label
#                   in one string.
#
# Return value - None

proc XParflow::MainWin_InsertDataSets {dataSets {index end}} {
    
    #---------------------------------------------------------------
    # Get the string by calling pfgetlist which returns the data set
    # name as well as its label.  The String is then formatted and
    # placed in the listbox.
    #---------------------------------------------------------------
    
    foreach i $dataSets {
	
	# Get the statistics so that this information only has
	# to be computed once.
	
	set str [lindex [pfgetlist $i] 0]
	.main.listframe.listbox insert $index [format "%-13s %-128s"  \
		[lindex $str 0] \
		[lindex $str 1]]
    }
    
    MainWin_UpdateMenuBar
    
}


# Procedure MainWin_UpdateList - This procedure is used to obtain the
#           list of data sets that are currently in memory and 
#           display them in the main list box.
#
# Parameters - None
#
# Variables - str - holds the data set name and its label
#
# Return value - None

proc XParflow::MainWin_UpdateList {} {
    
   .main.listframe.listbox delete 0 end
    
    foreach str [pfgetlist] {
	.main.listframe.listbox insert end [format "%-13s %-128s"  \
		[lindex $str 0] \
		[lindex $str 1]]
    }
}


# Procedure MainWin_DeleteDialog - This procedure is used to create a
#           dialog which verifies that the user wants to delete a data
#           set.
#
# Parameters - None
#
# Variables  - global tkPriv(button) - Holds the number of the button
#              that was pressed within the dialog box created by tk_dialog
#            - msg - message to be printed in the dialog box
#            - dataSet - name of data set to be deleted
#
# Return value - None

proc XParflow::MainWin_DeleteDataDialog {} {
    
    global tkPriv
    
    #-----------------------------------------------------------------
    # The user must first select a data set to be deleted
    #-----------------------------------------------------------------
    
    if {[set index [.main.listframe.listbox curselection]] == ""} {
	
	ErrorDialog "You must first select a data set to be deleted."
	return
	
    }
    
    #-----------------------------------------------------------------
    # If the data set to be deleted is associated with an information
    # window that is currently open, then give a warning before
    # deleting the data set.
    #-----------------------------------------------------------------
    
   set dataSet [lindex [.main.listframe.listbox get $index] 0]
    
    if {[lsearch [info commands] .info$dataSet] != -1} {
	
	tk_dialog .warning {Information Window Open} "There is an information window open for `$dataSet'.  The information window for this data set will be destroyed.  Do you wish to delete anyway?" warning -1 {Yes} {No}
	
      #-----------------------------------------------------------------
	# If the zeroth button is pressed, then `Yes' is selected and
	# the data set should be deleted.  If the first button is pressed,
      # then `No' has been selected.
	#-----------------------------------------------------------------
	
      switch $tkPriv(button) {
	  
	  0 { .main.listframe.listbox delete $index
	  pfdelete $dataSet
	  destroy .info$dataSet
	  MainWin_UpdateMenuBar }
	  
	  1 { }
	  
      }

  } else {
      
      #-----------------------------------------------------------------
      # Obtain the name of the data set at the selected index within the
      # main window's list box.
      #-----------------------------------------------------------------

      set dataSet [lindex [.main.listframe.listbox get $index] 0]
      set msg "Are you sure you want to delete `$dataSet'."
      tk_dialog .delete {Delete Data Set} $msg warning -1 {Delete} {Cancel}
      
      
      #-----------------------------------------------------------------
      # If the zeroth button is pressed, then `Delete' is selected and
      # the data set should be deleted.  If the first button is pressed,
      # then `Cancel' has been selected.
      #-----------------------------------------------------------------
      
      switch $tkPriv(button) {
	  
	  0 { .main.listframe.listbox delete $index
	  pfdelete $dataSet
	  MainWin_UpdateMenuBar }
	  
	  1 { }

      }
      
  }
  
}



# Procedure MainWin_CreateInfoWindow - This procedure is executed when
#           the user selects `Display Grid Information' from the `Data'
#           pulldown menu.  It gets the name of the data set selected
#           and calls InfoWindow to display information about the grid
#           and its data.
#
# Parameters - None
#
# Variables - str - The string selected from the main list box
#
# Return Value - None
#

proc XParflow::MainWin_CreateInfoWindow {} {
    
    
    #-----------------------------------------------------------------
    # Make sure a data set has been selected from the main list box.
    #-----------------------------------------------------------------

    if {[set index [.main.listframe.listbox curselection]] == ""} {
	
	ErrorDialog "You must first select a data set you wish to view." 
	return
	
    }
    
    #-----------------------------------------------------------------
    # Obtain the name of the data set from the string contained in the
    # particular list box entry selected.
    #-----------------------------------------------------------------
    
    set str [.main.listframe.listbox get $index]
    InfoWindow [lindex $str 0]
    
}


# Procedure MainWin_ChangeLabelDialog - This procedure creates a dialog
#           box used to let the user change the label of a data set that
#           has been loaded into memory.
#
# Parameters - None
#
# Variables  - global done - This variable is used to tell when either
#                            the `OK' or `Cancel' buttons have been pressed.
#            - index       - the index within the main window list box of the
#                            data set name and label to be changed
#            - dataSet     - the name of the data set whose label will be
#                            changed
#            - label       - the label to be changed
#
# Return value - None

proc XParflow::MainWin_ChangeLabelDialog {} {
    
    #--------------------------------------------------------------------
    # A data set in the main window list box must first be selected.
    #--------------------------------------------------------------------
    
    if {[set index [.main.listframe.listbox curselection]] == ""} {
	
	ErrorDialog "You must first select a data set whose label you wish to change."
	return
	
   }
   
   toplevel .newLabel
   wm title .newLabel "Change Data Set Label"
   wm resizable .newLabel true false
   
   set str [.main.listframe.listbox get $index]
   set dataSet [lindex $str 0]
   set label [lindex $str 1]
   
   #--------------------------------------------------------------------
   # Create a label and an entry for the data set label that is about to
   # be changed. `OK' and `Cancel' buttons must also be created.
   #--------------------------------------------------------------------
   
   frame .newLabel.entryFrame -relief raised -borderwidth 1
   label .newLabel.entryFrame.dataSet -text "$dataSet :"
   entry .newLabel.entryFrame.label -width 32
   frame .newLabel.btnFrame -relief raised -borderwidth 1
   
   #--------------------------------------------------------------------
   # The `OK' button must make the change and set done to true once it
   # has been pressed.  `Cancel' must just set done to true.
   #--------------------------------------------------------------------
   
   button .newLabel.btnFrame.ok -text OK \
      -command ".main.listframe.listbox delete $index
   pfnewlabel $dataSet [.newLabel.entryFrame.label get]
   MainWin_InsertDataSets $dataSet $index" 

   button .newLabel.btnFrame.cancel -text Cancel -command {set done true}
   
   .newLabel.entryFrame.label insert 0 $label
   
   #--------------------------------------------------------------------
   # Pack the widgets so that the label and entry are in the uppermost
   # frame and the buttons are in the bottommost frame
   #--------------------------------------------------------------------
   
   pack .newLabel.entryFrame -ipady 4 -fill x -expand 1
   pack .newLabel.entryFrame.dataSet -side left -padx 4
   pack .newLabel.entryFrame.label -side left -padx 4 -fill x -expand 1
   pack .newLabel.btnFrame -ipady 10 -fill x -expand 1
   pack .newLabel.btnFrame.ok .newLabel.btnFrame.cancel -side left -padx 10 \
	   -fill x -expand 1
   
   #--------------------------------------------------------------------
   # Center the window and grab the focus so that no other application
   # window can be used until a button is pressed to either accept or
   # cancel the change.
   #--------------------------------------------------------------------
   
   CenterWindow .newLabel
   GrabFocus .newLabel
   destroy .newLabel
   
}


#-----------------------------------------------------------------------------
# These are procedures used to maintain various windows throughout the
# xpftools application.
#-----------------------------------------------------------------------------


# Procedure CenterWindow - This procedure is used to center toplevel windows
#
# Parameters - w - The path of the window to be centered
#
# Variables - x - The new x coordinate the window is to be placed
#             y - The new y coordinate the window is to be placed
#
# Return value - None

proc XParflow::CenterWindow w {
    
    wm withdraw $w 
    update idletasks
    set x [expr [winfo screenwidth $w]/2 - [winfo reqwidth $w]/2 \
	    - [winfo vrootx [winfo parent $w]]]
    set y [expr [winfo screenheight $w]/2 - [winfo reqheight $w]/2 \
	    - [winfo vrooty [winfo parent $w]]]
    wm geom $w +$x+$y
    wm deiconify $w 
    
}
