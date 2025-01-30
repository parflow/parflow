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

# File xpftfunctions.tcl - These functions create the GUIs for each
# of the mathematical PFTools operations that get performed on data
# sets.  They specify what each GUI is to look like using the
# procedures from xpftfuncwin.tcl.

proc XParflow::VelMagnitude {} {

   set inputEntries {{{Component X} META_E_LIST_BOX}
                     {{Component Y} META_E_LIST_BOX}
                     {{Component Z} META_E_LIST_BOX}}

   set outputEntries {{Magnitude RADIO_BUTTON}}

   CreateFunctionWindow {Compute Velocity Magnitude}
   CreateIOFrame $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons VelMagnitude_Compute $XParflow::help(velMag)
   FunctionEventLoop

}
    

proc XParflow::VelMagnitude_Compute {} {

   GetInputParameters dataSetX dataSetY dataSetZ

   if {$dataSetX == "" || $dataSetY == "" || $dataSetZ == ""} {

      ErrorDialog "You must select three components before the velocity magnitude can be computed."
      return

   }

   if {[catch "pfvmag $dataSetX $dataSetY $dataSetZ" magnitude]} {

      ErrorDialog $magnitude
      return

   }

   MainWin_InsertDataSets $magnitude
   FillOutputEntries $magnitude
   UpdateStatDisplay
   UpdateDataDisplay

}


proc XParflow::CVel {} {

   set inputEntries {{{Conductivity}  META_E_LIST_BOX}
                     {{Pressure Head} META_E_LIST_BOX}}

   set outputEntries {{{Component X} RADIO_BUTTON}
                      {{Component Y} RADIO_BUTTON}
                      {{Component Z} RADIO_BUTTON}}

   CreateFunctionWindow "Compute Veloctiy In Cells"
   CreateIOFrame        $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons CVel_Compute $XParflow::help(cVel)
   FunctionEventLoop

}


proc XParflow::CVel_Compute {} {

   GetInputParameters cond pHead

   if {$cond == "" || $pHead == ""} {

      ErrorDialog "You must select two vector before the velocity in the cells can be computed."
      return

   }

   if {[catch "pfcvel $cond $pHead" comps]} {

      ErrorDialog $comps
      return

   }

   MainWin_InsertDataSets $comps
   FillOutputEntries [lindex $comps 0] [lindex $comps 1] [lindex $comps 2]
   UpdateStatDisplay
   UpdateDataDisplay

}


proc XParflow::VVel {} {

   set inputEntries {{{Conductivity}  META_E_LIST_BOX}
                     {{Pressure Head} META_E_LIST_BOX}}

   set outputEntries {{{Component X} RADIO_BUTTON}
                      {{Component Y} RADIO_BUTTON}
                      {{Component Z} RADIO_BUTTON}}

   CreateFunctionWindow "Compute Veloctiy At Vertices"
   CreateIOFrame $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons VVel_Compute $XParflow::help(vVel)
   FunctionEventLoop

}


proc XParflow::VVel_Compute {} {

   GetInputParameters cond phead 

   if {$cond == "" || $phead == ""} {

      ErrorDialog "You must select two components before the velocity at the vertices can be computed."
      return

   }

   if {[catch "pfvvel $cond $phead" comps]} {

      ErrorDialog $comps
      return

   }

   MainWin_InsertDataSets $comps
   FillOutputEntries [lindex $comps 0] [lindex $comps 1] [lindex $comps 2]
   UpdateStatDisplay
   UpdateDataDisplay

}


proc XParflow::Flux {} {

   set inputEntries {{{Conductivity}   META_E_LIST_BOX}
                     {{Hydraulic Head} META_E_LIST_BOX}}

   set outputEntries {{{Flux} RADIO_BUTTON}}

   CreateFunctionWindow "Compute Flux"
   CreateIOFrame $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons Flux_Compute $XParflow::help(flux)
   FunctionEventLoop

}


proc XParflow::Flux_Compute {} {

   GetInputParameters cond hhead 

   if {$cond == "" || $hhead == ""} {

      ErrorDialog "You must select two vectors before the flux can be computed."
      return

   }

   if {[catch "pfflux $cond $hhead" flux]} {

      ErrorDialog $flux
      return

   }

   MainWin_InsertDataSets $flux
   FillOutputEntries $flux
   UpdateStatDisplay
   UpdateDataDisplay

}


proc XParflow::HHead {} {

   set inputEntries {{{Pressure Head} META_E_LIST_BOX}}
   set outputEntries {{{Hydraulic Head} RADIO_BUTTON}}

   CreateFunctionWindow "Compute Hydraulic Head"
   CreateIOFrame $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons HHead_Compute $XParflow::help(hHead)
   FunctionEventLoop

}


proc XParflow::HHead_Compute {} {

   GetInputParameters dataSetX

   if {$dataSetX == ""} {

      ErrorDialog "You must select a vector before the hydraulic head can be computed."
      return

   }

   if {[catch "pfhhead $dataSetX" hhead]} {

      ErrorDialog $hhead
      return

   }

   MainWin_InsertDataSets $hhead
   FillOutputEntries $hhead
   UpdateStatDisplay
   UpdateDataDisplay

}


proc XParflow::PHead {} {

   set inputEntries {{{Hydraulic Head} META_E_LIST_BOX}}
   set outputEntries {{{Pressure Head} RADIO_BUTTON}}

   CreateFunctionWindow "Compute Pressure Head"
   CreateIOFrame $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons PHead_Compute $XParflow::help(pHead)
   FunctionEventLoop

}


proc XParflow::PHead_Compute {} {

   GetInputParameters dataSetX

   if {$dataSetX == ""} {

      ErrorDialog "You must select a vector before the pressure head can be computed."
      return

   }

   if {[catch "pfphead $dataSetX" phead]} {

      ErrorDialog $phead
      return

   }

   MainWin_InsertDataSets $phead
   FillOutputEntries $phead
   UpdateStatDisplay
   UpdateDataDisplay

}


proc XParflow::Axpy {} {

   set inputEntries {{{Vector X} META_E_LIST_BOX}
                     {{Vector Y} META_E_LIST_BOX}
                     {{Alpha}    META_ENTRY}}

   set outputEntries {{{Vector Y} radioButton}}

   CreateFunctionWindow "Compute Axpy"
   CreateIOFrame $inputEntries $outputEntries
   CreateStatDisplay
   CreateDataDisplay
   CreateFunctionButtons Axpy_Compute $XParflow::help(axpy)
   FunctionEventLoop

}


proc XParflow::Axpy_Compute {} {

   GetInputParameters dataSetX dataSetY alpha

   set num [scan $alpha "%lf" alpha]

   if {$dataSetX == "" || $dataSetY == "" || $num != 1} {

      ErrorDialog "You must select a two vectors and enter the constant for `alpha' before `y = alpha * x + y' can be computed." 
      return

   }

   if {[catch "pfaxpy $alpha $dataSetX $dataSetY" error]} {

      ErrorDialog $error

   }

   FillOutputEntries $dataSetY
   UpdateStatDisplay
   UpdateDataDisplay

}
    

proc XParflow::Diff {} {

   set inputEntries  {{{Set A}            META_E_LIST_BOX}
                      {{Set B}            META_E_LIST_BOX}
                      {{# Differing SD's} META_ENTRY     }
                      {{Absolute Zero}    META_ENTRY     }}

   set outputEntries {{{Min # Sig. Digs.} NO_RADIO_BUTTON}
                      {{At}               NO_RADIO_BUTTON}
                      {{Max Abs Diff.}    NO_RADIO_BUTTON}}

   CreateFunctionWindow {Compute Differences}
   CreateIOFrame $inputEntries $outputEntries
   CreateDiffDisplay

   CreateFunctionButtons {Diff_Compute} $XParflow::help(diff)
   FunctionEventLoop
 
}


proc XParflow::Diff_Compute {} {

   GetInputParameters dataSetA dataSetB sigDigs absZero



    if {$dataSetA == "" || $dataSetB == ""} {
	ErrorDialog "You must select two data sets before the difference can be computed."
	return
    }

    if { $sigDigs == "" } {
	set sigDigs 12
    }
    
    if { [scan "$sigDigs" "%d%s" x junk ] != 1 || $sigDigs < 0 } {
	ErrorDialog "You must enter a positive integer for the minimum number of differing significant digits the absolute zero before the difference can be computed."
	return

    }


    if { $absZero == "" } {
	set absZero 0.0
    }
    if { [scan "$absZero" "%lf%s" x junk] != 1 || $absZero < 0 } {
	ErrorDialog "You must enter a positive value for the absolute zero before the difference can be computed."
	return
    }

   # Result is of the form: {x y z Min_sig_digs} Max_abs_diff

   if {[catch "pfmdiff $dataSetA $dataSetB $sigDigs $absZero" diff]} {

      ErrorDialog $diff

   }
      
   if {$diff == ""} {

      set msg "There were no differences found given the current criteria."
      tk_dialog .func.message {No Differences} $msg warning 0 {OK}
      return

   }

   set coord [lindex $diff 0]
   set coordinate [format "( %3d, %3d, %3d)" [lindex $coord 0] \
                  [lindex $coord 1] [lindex $coord 2]]

   set minSigDigs [lindex $coord 3]
   set maxAbsDiff [lindex $diff 1]

   FillOutputEntries $minSigDigs $coordinate $maxAbsDiff
   UpdateDiffDisplay $dataSetA $dataSetB $sigDigs $absZero

}
   
   
proc XParflow::MDiff {} {

   set inputEntries {{{Set A}         META_E_LIST_BOX}
                     {{Set B}         META_E_LIST_BOX}
                     {{Absolute Zero} META_ENTRY}}

   set outputEntries {{{Min # Sig. Digs.} NO_RADIO_BUTTON}
                      {{At}               NO_RADIO_BUTTON}
                      {{Max Abs Diff.}    NO_RADIO_BUTTON}}

   CreateFunctionWindow {Compute Maximum Difference}
   CreateIOFrame $inputEntries $outputEntries

   CreateFunctionButtons {MDiff_Compute} $XParflow::help(mDiff)
   FunctionEventLoop

}


proc XParflow::MDiff_Compute {} {

    GetInputParameters dataSetA dataSetB absZero

    if {$dataSetA == "" || $dataSetB == ""} {
	ErrorDialog "You must select two data sets before the maximum difference can be computed."
	return
    }

    if { $absZero == ""} {
	set absZero 0
    }

    if { [scan "$absZero" "%lf%s" x junk] != 1 || $absZero < 0} {
	ErrorDialog "You must select a positive value for the absolute zero before the maximum difference can be computed."
	return
    }

   # Result is of the form: {x y z Min_sig_digs} Max_abs_diff

   if {[catch "pfmdiff $dataSetA $dataSetB -1 $absZero" diff]} {

      ErrorDialog $diff

   }

   if {$diff == ""} {

      set msg "There were no differences found given the current criteria."
      tk_dialog .func.message {No Differences} $msg "" 0 {OK}
      return

   }

   set coord [lindex $diff 0]
   set coordinate [format "( %3d, %3d, %3d)" [lindex $coord 0] \
                  [lindex $coord 1] [lindex $coord 2]]

   set minSigDigs [lindex $coord 3]
   set maxAbsDiff [lindex $diff 1]

   FillOutputEntries $minSigDigs $coordinate $maxAbsDiff

}
