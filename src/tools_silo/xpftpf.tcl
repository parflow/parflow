#BHEADER***********************************************************************
# (c) 1996   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

#-----------------------------------------------------------------------------
# This file contains procedures used to load in and save various ParFlow
# format files.
#-----------------------------------------------------------------------------


# Procedure GetPFFileName - This procedure is used select a pattern and
#           a title to be utilized by the file selection box.  The operation
#           (loading or saving) the file selection box will be used for
#           is also passed to the FSBox routine.
#
# Parameters - pfType    - The type of the ParFlow file to be loaded/saved
#              operation - The type of operation to be performed by the fsb
#
# Variables - pattern  - used to store the pattern to be used by the fsb
#             fsbTitle - used to store the title to be used by the fsb
# 
# Return value - The name of the ParFlow file to be loaded/saved

proc XParflow::GetPFFileName {pfType operation} {

   #----------------------------------------------
   # Determine what the operation is so that the
   # title of the fsb will properly reflect it.
   #----------------------------------------------

   if {$operation == "loading"} {
      set opWord Load
   } elseif {$operation == "saving"} {
      set opWord Save
   } 

   #----------------------------------------------
   # Determine the file type to be loaded and set
   # the pattern and title appropriatly.
   #----------------------------------------------

   switch $pfType {

      pfb {
	  set pattern { {"ParFlow Binary" {.pfb} } }
          set fsbTitle "$opWord ParFlow Binary File"
      }

      pfsb {
	  set pattern { {"ParFlow Simple Binary" {.pfsb} } }
	  set fsbTitle "$opWord ParFlow Simple Binary File"
      }

      sa {
	  set pattern { {"ParFlow Simple ASCII" {.sa}} } 
	  set fsbTitle "$opWord ParFlow Simple ASCII File"
      }

      sb {
	  set pattern { {"ParFlow Simple Binary" {.sb}} } 
	  set fsbTitle "$opWord ParFlow Simple Binary File"
      }

      rsa {
	  set pattern { {"Parflow Real Scattered ASCII" {.rsa}} }
	  set fsbTitle "$opWord ParFlow Real Scattered ASCII File"
      }
      fld {
	  set pattern { {"AVS Field" {.fld}} }
      	  set fsbTitle "$opWord AVS Field File"
      }
   }

   return [FSBox $fsbTitle $pattern $operation]
}



# Procedure LoadPF - This procedure is used to make sure the file that
#           selected using the fsb is actually readable and then 
#           executes the pfload command to load it.
#
# Parameters - pfType - The type of the file to be loaded
#
# Variables - fileName - The name of the file to be loaded
#             
# Return value - NOne

proc XParflow::LoadPF pfType {

   set fileName [GetPFFileName $pfType loading]

   #-------------------------------------------
   # Return if no file was selected
   #-------------------------------------------
  
   if {$fileName == ""} {
      
      return

   }

   #-------------------------------------------
   # Check if the file is readable and attempt
   # to load it.
   #-------------------------------------------

   if {[file readable $fileName]} {


      #------------------------------------------------------
      # The only reason an error would occur here is if there
      # is not enough memory available to load the data.
      #------------------------------------------------------

      if {[catch "pfload -$pfType $fileName" dataset]} {

         ErrorDialog {There is not enough memory to store the data.  Try deleting a few data sets to free up space.}
         return

      } else {

         MainWin_InsertDataSets $dataset

      }

   #-------------------------------------------
   # The user does not have read permission
   #-------------------------------------------

   } else {

      ErrorDialog {You do not have read access\nto the requested file.}
      LoadPF $pfType 

   }


}


# Procedure SavePF - This procedure is used to save grid data to a
#           ParFlow format file.
#
# Parameters - pfType - The type of ParFlow data file to be saved
#
# Variables - str      - The string containing the data set name and label
#             dataSet  - The name of the data set to be saved
#             fileName - The name of the file to be saved.
# 
#  
proc XParflow::SavePF pfType {

   #-----------------------------------------------------
   # Make sure that a data set in the main list box has
   # been selected.
   #-----------------------------------------------------

   if {[catch "selection get" str]} {

      bell
      ErrorDialog "You must first select a data set to be saved."
      return

   }

   set dataSet [lindex $str 0]
   set fileName [GetPFFileName $pfType saving]

   #-----------------------------------------------------
   # Return if the filename selected was null.
   #-----------------------------------------------------

   if {$fileName == ""} {

      return

   }


   #--------------------------------------------------------
   # Save the data to the file if the directory is writable
   #--------------------------------------------------------
   
   if {![file writable [file dirname $fileName]]} {

      ErrorDialog "You do not have write access to the requested directory."
      SavePF $pfType $dataSet

   #--------------------------------------------------------
   # If the file already exists, then make sure it is
   # writable.
   #--------------------------------------------------------

   } elseif {[file exists $fileName] && ![file writable $fileName]} {

      ErrorDialog "You do not have write access to the requested file."
      SavePF $pfType $dataSet

   } else {

      pfsave $dataSet -$pfType $fileName

   }

}
