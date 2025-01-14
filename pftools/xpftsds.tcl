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

# File xpftsds.tcl - These procedures ares used to read and save
#                    data sets that are in HDF format. 


# Procedure ScanDSNum - This procedure is used to parse the value
#           given as the DS number when reading in an SDS.
#
# Parameters - None
#
# Return value - None

proc XParflow::ScanDSNum {} {

   # Make sure an integer is entered

   if {[scan [.dsQuery.entryFrame.dsNumEntry get] "%d" num] != 1} {

      bell

   # Let the dialog box know that the correct value has been entered
 
   } else {

      global done
      set done true

   }

}


# Procedure getSDSNum - This procedure creates a dialog box used to
#           obtain the SDS number of the data set to be loaded in
#           from a file that is in HDF format.
#
# Parameters - None
#
# Variables - global dsNum - The SDS number of the SDS to be loaded
#
# Return value - dsNum

proc XParflow::GetSDSNum {} {

   #--------------------------------------
   # Create the toplevel dialog window
   #--------------------------------------

   toplevel .dsQuery
   wm title .dsQuery "HDF Data Set Number"
   wm resizable .dsQuery false false

   set query "Please enter the number\nof the SDS to be loaded\nfrom the HDF file."

   #---------------------------------------------------
   # Create the widgets to be used in the dialog box.
   # An entry is used to get obtain the SDS number which
   # is zero by default.
   #---------------------------------------------------

   frame .dsQuery.msgFrame -relief raised -borderwidth 1
   label .dsQuery.msgFrame.bmp -bitmap question
   message .dsQuery.msgFrame.msg -justify left -width 10c -text $query
   frame .dsQuery.entryFrame -relief raised -borderwidth 1
   label .dsQuery.entryFrame.dsNumLabel -text "SDS Number:"
   entry .dsQuery.entryFrame.dsNumEntry -relief sunken -borderwidth 2 -width 2
   .dsQuery.entryFrame.dsNumEntry insert 0 0

   bind .dsQuery.entryFrame.dsNumEntry <Return> {ScanDSNum}

   #---------------------------------------------------
   # Manage the widgets.
   #---------------------------------------------------

   pack .dsQuery.msgFrame -side top -fill x
   pack .dsQuery.msgFrame.bmp -side left -anchor center -padx 10
   pack .dsQuery.msgFrame.msg -anchor center -padx 5 -pady 10
   pack .dsQuery.entryFrame -side bottom -fill x -ipady 5
   pack .dsQuery.entryFrame.dsNumLabel 
   pack .dsQuery.entryFrame.dsNumEntry

   CenterWindow .dsQuery

   #---------------------------------------------------
   # Wait until the number in entered before releasing
   # the focus.
   #---------------------------------------------------
   
   GrabFocus .dsQuery
   set dsNum [.dsQuery.entryFrame.dsNumEntry get] 
   destroy .dsQuery

   return $dsNum 

}


# Procedure LoadSDS - This procedure is used to load in an SDS from an
#           HDF file.
#
# Parameters - None
#
# Variables - fileName - The name of the HDF file containing the SDS to be
#                        loaded.
#
# Return value - None

proc XParflow::LoadSDS {} {

    set pattern { {"Scientific DataSet" {.hdf}} \
	    {"All Files" *}}

   set fileName [FSBox "Load Scientific Data Set" $pattern loading]

   #---------------------------------------------
   # Return if no file was selected
   #---------------------------------------------
   
   if {$fileName == ""} {

      return

   }

   set dsNum [GetSDSNum]

   #------------------------------------------------
   # If the file is not readable then give an error
   # else load the SDS.
   #------------------------------------------------

   if {[file readable $fileName]} {

      #--------------------------------------------------
      # An error can only occur here is there is not
      # enough memory available to store the data.
      #--------------------------------------------------

      if {[catch "pfloadsds $fileName $dsNum" dataset]} {

         ErrorDialog "The SDS could not be loaded.  The SDS number given may be invalid or there is not enough memory available to store the data."
         LoadSDS

      } else {

         MainWin_InsertDataSets $dataset

      }

   } else {

      ErrorDialog "You do not have read access\nto the requested file."
      LoadSDS

   }


}

# Procedure GetHDFFileName - This procedure is used to set the title of
#           the fsb when an HDF file is to be saved.
#
# Parameters - sdsType - The type or format of the SDS the grid data is
#                        to be saved as.
#
# Variables - None
#
# Return value - Name of file the data is to be saved in
 
proc XParflow::GetHDFFileName sdsType {

    set pattern { {"Scientific DataSet" {.hdf}} \
	    {"All Files" *} }

   #--------------------------------------------------
   # Set the title appropriately depending on what type
   # the SDS is to be saved as.
   #--------------------------------------------------

   switch $sdsType {

      float64 {set fsbTitle {Save SDS in Float 64 HDF format}}

      float32 {set fsbTitle {Save SDS in Float 32 HDF format}}

      int32   {set fsbTitle {Save SDS in Int 32 HDF format}}

      uint32  {set fsbTitle {Save SDS in UInt 32 HDF format}} 

      int16   {set fsbTitle {Save SDS in Int 16 HDF format}}
      
      uint16  {set fsbTitle {Save SDS in UInt 16 HDF format}}

      int8    {set fsbTitle {Save SDS in Int 8 HDF format}}

      uint8   {set fsbTitle {Save SDS in UInt 8 HDF format}}

   }

   return [FSBox $fsbTitle $pattern saving]

}


# Procedure SaveSDS - This procedure is used to save data to a SDS.
#
# Parameters - sdsType - The type of the SDS the data is to be saved in
#
# Variables - fileName - The name of the HDF file the grid data is to be
#                        saved to
#             dataSet  - The name of the data set to be saved
#
# Return value - None

proc XParflow::SaveSDS sdsType {

   #--------------------------------------------
   # Make sure a data set has been selected
   #--------------------------------------------

   if {[catch "selection get" str]} {

      bell
      ErrorDialog "You must first select a data set\n to be saved."
      return

   }

   set dataSet [lindex $str 0]
   set fileName [GetHDFFileName $sdsType]

   #--------------------------------------------
   # Return if no filename was selected
   #--------------------------------------------

   if {$fileName == ""} {

      return

   }

   #--------------------------------------------
   # Make sure that the directory is writable
   #--------------------------------------------
   
   if {![file writable [file dirname $fileName]]} {

      ErrorDialog "You do not have write access to the requested directory."
      SaveSDS $sdsType $dataSet

   #----------------------------------------------------------
   # If the file already exists, then make sure it is writable 
   #----------------------------------------------------------
    
   } elseif {[file exists $fileName] && ![file writable $fileName]} {

      ErrorDialog "You do not have write access to the requested file."
      SaveSDS $sdsType $dataSet

   } else {

      pfsavesds $dataSet -$sdsType $fileName

   }

}
  
   
