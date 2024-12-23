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

###############################################################################
#
# Problem input:
#
###############################################################################

# FORSYTH #5
#  This problem comes from Forsyth, Wu, and Pruess (Advances in Water
#  Resources, 1995).  It is a 3-dimensional version of problem 2/3 from the
#  same source.  The saturation levels are tracked over the course 
#  of 30 days, with infiltration from a patch on the top surface
#  of the domain.  
#

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

#  Take the process topology and output name as inputs from command line
# set ProcessesX [lindex $argv 0] 
# set ProcessesY [lindex $argv 1] 
# set ProcessesZ [lindex $argv 2]
# set outname    [lindex $argv 3]
# pfset Process.Topology.P  $ProcessesX
# pfset Process.Topology.Q  $ProcessesY
# pfset Process.Topology.R  $ProcessesZ

pfset Process.Topology.P  1
pfset Process.Topology.Q  1
pfset Process.Topology.R  1

set outname   forsyth5

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

#pfset ComputationalGrid.NX                88
#pfset ComputationalGrid.NY                88
#pfset ComputationalGrid.NZ                64

pfset ComputationalGrid.NX                46
pfset ComputationalGrid.NY                46
pfset ComputationalGrid.NZ                21

set   UpperX                              800.0
set   UpperY                              800.0
set   UpperZ                              650.0

set   LowerX                              [pfget ComputationalGrid.Lower.X]
set   LowerY                              [pfget ComputationalGrid.Lower.Y]
set   LowerZ                              [pfget ComputationalGrid.Lower.Z]

set   NX                                  [pfget ComputationalGrid.NX]
set   NY                                  [pfget ComputationalGrid.NY]
set   NZ                                  [pfget ComputationalGrid.NZ]

pfset ComputationalGrid.DX	          [expr ($UpperX - $LowerX) / $NX]
pfset ComputationalGrid.DY                [expr ($UpperY - $LowerY) / $NY]
pfset ComputationalGrid.DZ	          [expr ($UpperZ - $LowerZ) / $NZ]

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
set   Zones                           "zone1 zone2 zone3above4 zone3left4 \
                                      zone3right4right5 zone3above5 
                                      zone3below5 zone3behindmain5
                                      zone3behindtail5 zone3behind4 
                                      zone3front4 zone4 zone5main zone5tail"

pfset GeomInput.Names                 "solidinput $Zones background"

pfset GeomInput.solidinput.InputType  SolidFile
pfset GeomInput.solidinput.GeomNames  domain
pfset GeomInput.solidinput.FileName   fors5_hf.pfsol


pfset GeomInput.zone1.InputType       Box
pfset GeomInput.zone1.GeomName        zone1

pfset Geom.zone1.Lower.X              0.0
pfset Geom.zone1.Lower.Y              0.0
pfset Geom.zone1.Lower.Z              610.0
pfset Geom.zone1.Upper.X              800.0
pfset Geom.zone1.Upper.Y              800.0
pfset Geom.zone1.Upper.Z              650.0


pfset GeomInput.zone2.InputType       Box
pfset GeomInput.zone2.GeomName        zone2

pfset Geom.zone2.Lower.X              0.0
pfset Geom.zone2.Lower.Y              0.0
pfset Geom.zone2.Lower.Z              560.0
pfset Geom.zone2.Upper.X              800.0
pfset Geom.zone2.Upper.Y              800.0
pfset Geom.zone2.Upper.Z              610.0


pfset GeomInput.zone3above4.InputType Box
pfset GeomInput.zone3above4.GeomName  zone3above4

pfset Geom.zone3above4.Lower.X        0.0
pfset Geom.zone3above4.Lower.Y        0.0
pfset Geom.zone3above4.Lower.Z        500.0
pfset Geom.zone3above4.Upper.X        800.0
pfset Geom.zone3above4.Upper.Y        800.0
pfset Geom.zone3above4.Upper.Z        560.0


pfset GeomInput.zone3left4.InputType  Box
pfset GeomInput.zone3left4.GeomName   zone3left4

pfset Geom.zone3left4.Lower.X         0.0
pfset Geom.zone3left4.Lower.Y         0.0
pfset Geom.zone3left4.Lower.Z         285.0
pfset Geom.zone3left4.Upper.X         100.0
pfset Geom.zone3left4.Upper.Y         800.0
pfset Geom.zone3left4.Upper.Z         500.0


pfset GeomInput.zone3right4right5.InputType  Box
pfset GeomInput.zone3right4right5.GeomName   zone3right4right5

pfset Geom.zone3right4right5.Lower.X        300.0
pfset Geom.zone3right4right5.Lower.Y        0.0
pfset Geom.zone3right4right5.Lower.Z        0.0
pfset Geom.zone3right4right5.Upper.X        800.0
pfset Geom.zone3right4right5.Upper.Y        800.0
pfset Geom.zone3right4right5.Upper.Z        500.0


pfset GeomInput.zone3above5.InputType  Box
pfset GeomInput.zone3above5.GeomName   zone3above5

pfset Geom.zone3above5.Lower.X         100.0
pfset Geom.zone3above5.Lower.Y         0.0
pfset Geom.zone3above5.Lower.Z         285.0
pfset Geom.zone3above5.Upper.X         273.0
pfset Geom.zone3above5.Upper.Y         800.0
pfset Geom.zone3above5.Upper.Z         400.0


pfset GeomInput.zone3below5.InputType  Box
pfset GeomInput.zone3below5.GeomName   zone3below5

pfset Geom.zone3below5.Lower.X         0.0
pfset Geom.zone3below5.Lower.Y         0.0
pfset Geom.zone3below5.Lower.Z         0.0
pfset Geom.zone3below5.Upper.X         300.0
pfset Geom.zone3below5.Upper.Y         800.0
pfset Geom.zone3below5.Upper.Z         200.0


pfset GeomInput.zone3behindmain5.InputType  Box
pfset GeomInput.zone3behindmain5.GeomName   zone3behindmain5

pfset Geom.zone3behindmain5.Lower.X         0.0
pfset Geom.zone3behindmain5.Lower.Y         300.0
pfset Geom.zone3behindmain5.Lower.Z         200.0
pfset Geom.zone3behindmain5.Upper.X         300.0
pfset Geom.zone3behindmain5.Upper.Y         800.0
pfset Geom.zone3behindmain5.Upper.Z         285.0


pfset GeomInput.zone3behindtail5.InputType  Box
pfset GeomInput.zone3behindtail5.GeomName   zone3behindtail5

pfset Geom.zone3behindtail5.Lower.X         273.0
pfset Geom.zone3behindtail5.Lower.Y         300.0
pfset Geom.zone3behindtail5.Lower.Z         285.0
pfset Geom.zone3behindtail5.Upper.X         300.0
pfset Geom.zone3behindtail5.Upper.Y         800.0
pfset Geom.zone3behindtail5.Upper.Z         400.0


pfset GeomInput.zone3behind4.InputType  Box
pfset GeomInput.zone3behind4.GeomName   zone3behind4

pfset Geom.zone3behind4.Lower.X         100.0
pfset Geom.zone3behind4.Lower.Y         300.0
pfset Geom.zone3behind4.Lower.Z         400.0
pfset Geom.zone3behind4.Upper.X         300.0
pfset Geom.zone3behind4.Upper.Y         800.0
pfset Geom.zone3behind4.Upper.Z         500.0


pfset GeomInput.zone3front4.InputType  Box
pfset GeomInput.zone3front4.GeomName   zone3front4

pfset Geom.zone3front4.Lower.X         100.0
pfset Geom.zone3front4.Lower.Y         0.0
pfset Geom.zone3front4.Lower.Z         400.0
pfset Geom.zone3front4.Upper.X         300.0
pfset Geom.zone3front4.Upper.Y         100.0
pfset Geom.zone3front4.Upper.Z         500.0


pfset GeomInput.zone4.InputType  Box
pfset GeomInput.zone4.GeomName   zone4

pfset Geom.zone4.Lower.X         100.0
pfset Geom.zone4.Lower.Y         100.0
pfset Geom.zone4.Lower.Z         400.0
pfset Geom.zone4.Upper.X         300.0
pfset Geom.zone4.Upper.Y         300.0
pfset Geom.zone4.Upper.Z         500.0


pfset GeomInput.zone5main.InputType  Box
pfset GeomInput.zone5main.GeomName   zone5main

pfset Geom.zone5main.Lower.X         0.0
pfset Geom.zone5main.Lower.Y         0.0
pfset Geom.zone5main.Lower.Z         200.0
pfset Geom.zone5main.Upper.X         300.0
pfset Geom.zone5main.Upper.Y         300.0
pfset Geom.zone5main.Upper.Z         285.0


pfset GeomInput.zone5tail.InputType  Box
pfset GeomInput.zone5tail.GeomName   zone5tail

pfset Geom.zone5tail.Lower.X         273.0
pfset Geom.zone5tail.Lower.Y         0.0
pfset Geom.zone5tail.Lower.Z         285.0
pfset Geom.zone5tail.Upper.X         300.0
pfset Geom.zone5tail.Upper.Y         300.0
pfset Geom.zone5tail.Upper.Z         400.0


pfset GeomInput.background.InputType  Box
pfset GeomInput.background.GeomName   background

pfset Geom.background.Lower.X         -99999999.0
pfset Geom.background.Lower.Y         -99999999.0
pfset Geom.background.Lower.Z         -99999999.0
pfset Geom.background.Upper.X         99999999.0
pfset Geom.background.Upper.Y         99999999.0
pfset Geom.background.Upper.Z         99999999.0

pfset Geom.domain.Patches             "infiltration z-upper x-lower y-lower \
                                      x-upper y-upper z-lower"


#---------------------------------------------------------
# Permeability:
#---------------------------------------------------------

pfset Geom.Perm.Names                 $Zones

# Values in cm^2

pfset Geom.zone1.Perm.Type            Constant
pfset Geom.zone1.Perm.Value           9.1496e-5

pfset Geom.zone2.Perm.Type            Constant
pfset Geom.zone2.Perm.Value           5.4427e-5

pfset Geom.zone3above4.Perm.Type            Constant
pfset Geom.zone3above4.Perm.Value           4.8033e-5

pfset Geom.zone3left4.Perm.Type            Constant
pfset Geom.zone3left4.Perm.Value           4.8033e-5

pfset Geom.zone3right4right5.Perm.Type            Constant
pfset Geom.zone3right4right5.Perm.Value           4.8033e-5

pfset Geom.zone3above5.Perm.Type            Constant
pfset Geom.zone3above5.Perm.Value           4.8033e-5

pfset Geom.zone3below5.Perm.Type            Constant
pfset Geom.zone3below5.Perm.Value           4.8033e-5

pfset Geom.zone3behindmain5.Perm.Type            Constant
pfset Geom.zone3behindmain5.Perm.Value           4.8033e-5

pfset Geom.zone3behindtail5.Perm.Type            Constant
pfset Geom.zone3behindtail5.Perm.Value           4.8033e-5

pfset Geom.zone3behind4.Perm.Type            Constant
pfset Geom.zone3behind4.Perm.Value           4.8033e-5

pfset Geom.zone3front4.Perm.Type            Constant
pfset Geom.zone3front4.Perm.Value           4.8033e-5

pfset Geom.zone4.Perm.Type            Constant
pfset Geom.zone4.Perm.Value           4.8033e-4

pfset Geom.zone5main.Perm.Type            Constant
pfset Geom.zone5main.Perm.Value           9.8067e-8

pfset Geom.zone5tail.Perm.Type            Constant
pfset Geom.zone5tail.Perm.Value           9.8067e-8

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "background"

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	        Constant
pfset Phase.water.Density.Value	        1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.124e-2

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

pfset Contaminants.Names			"tce"
pfset Contaminants.tce.Degradation.Value	 0.0

pfset PhaseConcen.water.tce.Type                 Constant
pfset PhaseConcen.water.tce.GeomNames            domain
pfset PhaseConcen.water.tce.Geom.domain.Value    0.0

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

pfset Geom.Retardation.GeomNames           background
pfset Geom.background.tce.Retardation.Type     Linear
pfset Geom.background.tce.Retardation.Rate     0.0

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#---------------------------------------------------------
# Setup timing info
#---------------------------------------------------------

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		0
pfset TimingInfo.StartTime		0.0
# If testing only solve for 6 dump intervals (3 maxstep intervals)
if { [info exists ::env(PF_TEST) ] } {
    pfset TimingInfo.StopTime               [expr 86400.0 * 3]
    pfset TimingInfo.DumpInterval	     43200.0
} {
    pfset TimingInfo.StopTime               2592000.0
    pfset TimingInfo.DumpInterval	     432000.0
}

pfset TimeStep.Type                     Growth
pfset TimeStep.InitialStep              43200.0
pfset TimeStep.GrowthFactor             1.2
pfset TimeStep.MaxStep                  86400.0
pfset TimeStep.MinStep                  43200.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames           $Zones

pfset Geom.zone1.Porosity.Type          Constant
pfset Geom.zone1.Porosity.Value         0.3680

pfset Geom.zone2.Porosity.Type          Constant
pfset Geom.zone2.Porosity.Value         0.3510

pfset Geom.zone3above4.Porosity.Type          Constant
pfset Geom.zone3above4.Porosity.Value         0.3250

pfset Geom.zone3left4.Porosity.Type          Constant
pfset Geom.zone3left4.Porosity.Value         0.3250

pfset Geom.zone3right4right5.Porosity.Type    Constant
pfset Geom.zone3right4right5.Porosity.Value   0.3250

pfset Geom.zone3above5.Porosity.Type    Constant
pfset Geom.zone3above5.Porosity.Value   0.3250

pfset Geom.zone3below5.Porosity.Type    Constant
pfset Geom.zone3below5.Porosity.Value   0.3250

pfset Geom.zone3behindmain5.Porosity.Type    Constant
pfset Geom.zone3behindmain5.Porosity.Value   0.3250

pfset Geom.zone3behindtail5.Porosity.Type    Constant
pfset Geom.zone3behindtail5.Porosity.Value   0.3250

pfset Geom.zone3behind4.Porosity.Type    Constant
pfset Geom.zone3behind4.Porosity.Value   0.3250

pfset Geom.zone3front4.Porosity.Type    Constant
pfset Geom.zone3front4.Porosity.Value   0.3250

pfset Geom.zone4.Porosity.Type    Constant
pfset Geom.zone4.Porosity.Value   0.3250

pfset Geom.zone5main.Porosity.Type    Constant
pfset Geom.zone5main.Porosity.Value   0.3250

pfset Geom.zone5tail.Porosity.Type    Constant
pfset Geom.zone5tail.Porosity.Value   0.3250

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          $Zones

pfset Geom.zone1.RelPerm.Alpha         0.0334
pfset Geom.zone1.RelPerm.N             1.982 

pfset Geom.zone2.RelPerm.Alpha         0.0363
pfset Geom.zone2.RelPerm.N             1.632 

pfset Geom.zone3above4.RelPerm.Alpha   0.0345
pfset Geom.zone3above4.RelPerm.N       1.573 

pfset Geom.zone3left4.RelPerm.Alpha    0.0345
pfset Geom.zone3left4.RelPerm.N        1.573 

pfset Geom.zone3right4right5.RelPerm.Alpha    0.0345
pfset Geom.zone3right4right5.RelPerm.N        1.573 

pfset Geom.zone3above5.RelPerm.Alpha    0.0345
pfset Geom.zone3above5.RelPerm.N        1.573 

pfset Geom.zone3below5.RelPerm.Alpha    0.0345
pfset Geom.zone3below5.RelPerm.N        1.573 

pfset Geom.zone3behindmain5.RelPerm.Alpha    0.0345
pfset Geom.zone3behindmain5.RelPerm.N        1.573 

pfset Geom.zone3behindtail5.RelPerm.Alpha    0.0345
pfset Geom.zone3behindtail5.RelPerm.N        1.573 

pfset Geom.zone3behind4.RelPerm.Alpha    0.0345
pfset Geom.zone3behind4.RelPerm.N        1.573 

pfset Geom.zone3front4.RelPerm.Alpha    0.0345
pfset Geom.zone3front4.RelPerm.N        1.573 

pfset Geom.zone4.RelPerm.Alpha    0.0345
pfset Geom.zone4.RelPerm.N        1.573 

pfset Geom.zone5main.RelPerm.Alpha    0.0345
pfset Geom.zone5main.RelPerm.N        1.573 

pfset Geom.zone5tail.RelPerm.Alpha    0.0345
pfset Geom.zone5tail.RelPerm.N        1.573 

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         $Zones

pfset Geom.zone1.Saturation.Alpha        0.0334
pfset Geom.zone1.Saturation.N            1.982
pfset Geom.zone1.Saturation.SRes         0.2771
pfset Geom.zone1.Saturation.SSat         1.0

pfset Geom.zone2.Saturation.Alpha        0.0363
pfset Geom.zone2.Saturation.N            1.632
pfset Geom.zone2.Saturation.SRes         0.2806
pfset Geom.zone2.Saturation.SSat         1.0

pfset Geom.zone3above4.Saturation.Alpha  0.0345
pfset Geom.zone3above4.Saturation.N      1.573
pfset Geom.zone3above4.Saturation.SRes   0.2643
pfset Geom.zone3above4.Saturation.SSat   1.0

pfset Geom.zone3left4.Saturation.Alpha   0.0345
pfset Geom.zone3left4.Saturation.N       1.573
pfset Geom.zone3left4.Saturation.SRes    0.2643
pfset Geom.zone3left4.Saturation.SSat    1.0

pfset Geom.zone3right4right5.Saturation.Alpha   0.0345
pfset Geom.zone3right4right5.Saturation.N       1.573
pfset Geom.zone3right4right5.Saturation.SRes    0.2643
pfset Geom.zone3right4right5.Saturation.SSat    1.0

pfset Geom.zone3above5.Saturation.Alpha   0.0345
pfset Geom.zone3above5.Saturation.N       1.573
pfset Geom.zone3above5.Saturation.SRes    0.2643
pfset Geom.zone3above5.Saturation.SSat    1.0

pfset Geom.zone3below5.Saturation.Alpha   0.0345
pfset Geom.zone3below5.Saturation.N       1.573
pfset Geom.zone3below5.Saturation.SRes    0.2643
pfset Geom.zone3below5.Saturation.SSat    1.0

pfset Geom.zone3behindmain5.Saturation.Alpha   0.0345
pfset Geom.zone3behindmain5.Saturation.N       1.573
pfset Geom.zone3behindmain5.Saturation.SRes    0.2643
pfset Geom.zone3behindmain5.Saturation.SSat    1.0

pfset Geom.zone3behindtail5.Saturation.Alpha   0.0345
pfset Geom.zone3behindtail5.Saturation.N       1.573
pfset Geom.zone3behindtail5.Saturation.SRes    0.2643
pfset Geom.zone3behindtail5.Saturation.SSat    1.0

pfset Geom.zone3behind4.Saturation.Alpha   0.0345
pfset Geom.zone3behind4.Saturation.N       1.573
pfset Geom.zone3behind4.Saturation.SRes    0.2643
pfset Geom.zone3behind4.Saturation.SSat    1.0

pfset Geom.zone3front4.Saturation.Alpha   0.0345
pfset Geom.zone3front4.Saturation.N       1.573
pfset Geom.zone3front4.Saturation.SRes    0.2643
pfset Geom.zone3front4.Saturation.SSat    1.0

pfset Geom.zone4.Saturation.Alpha   0.0345
pfset Geom.zone4.Saturation.N       1.573
pfset Geom.zone4.Saturation.SRes    0.2643
pfset Geom.zone4.Saturation.SSat    1.0

pfset Geom.zone5main.Saturation.Alpha   0.0345
pfset Geom.zone5main.Saturation.N       1.573
pfset Geom.zone5main.Saturation.SRes    0.2643
pfset Geom.zone5main.Saturation.SSat    1.0

pfset Geom.zone5tail.Saturation.Alpha   0.0345
pfset Geom.zone5tail.Saturation.N       1.573
pfset Geom.zone5tail.Saturation.SRes    0.2643
pfset Geom.zone5tail.Saturation.SSat    1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names constant
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

pfset Patch.infiltration.BCPressure.Type	      FluxConst
pfset Patch.infiltration.BCPressure.Cycle	      "constant"
pfset Patch.infiltration.BCPressure.alltime.Value     -2.3148e-5

pfset Patch.x-lower.BCPressure.Type		      FluxConst
pfset Patch.x-lower.BCPressure.Cycle		      "constant"
pfset Patch.x-lower.BCPressure.alltime.Value	      0.0

pfset Patch.y-lower.BCPressure.Type		      FluxConst
pfset Patch.y-lower.BCPressure.Cycle		      "constant"
pfset Patch.y-lower.BCPressure.alltime.Value	      0.0

pfset Patch.z-lower.BCPressure.Type		      FluxConst
pfset Patch.z-lower.BCPressure.Cycle		      "constant"
pfset Patch.z-lower.BCPressure.alltime.Value	      0.0

pfset Patch.x-upper.BCPressure.Type		      FluxConst
pfset Patch.x-upper.BCPressure.Cycle		      "constant"
pfset Patch.x-upper.BCPressure.alltime.Value	      0.0

pfset Patch.y-upper.BCPressure.Type		      FluxConst
pfset Patch.y-upper.BCPressure.Cycle		      "constant"
pfset Patch.y-upper.BCPressure.alltime.Value	      0.0

pfset Patch.z-upper.BCPressure.Type		      FluxConst
pfset Patch.z-upper.BCPressure.Cycle		      "constant"
pfset Patch.z-upper.BCPressure.alltime.Value	      0.0

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type                                   Constant
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -10000.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    background
pfset PhaseSources.water.Geom.background.Value        0.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "background"
pfset Geom.background.SpecificStorage.Value          1.0e-5

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "domain"
pfset TopoSlopesX.Geom.domain.Value 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "domain"
pfset TopoSlopesY.Geom.domain.Value 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value 2.3e-7

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     10000

pfset Solver.Nonlinear.MaxIter                           15
pfset Solver.Nonlinear.ResidualTol                       1e-8
pfset Solver.Nonlinear.StepTol                           1e-8
pfset Solver.Nonlinear.EtaChoice                         Walker2
#pfset Solver.Nonlinear.EtaValue                          1e-5
pfset Solver.Nonlinear.EtaAlpha                          2
pfset Solver.Nonlinear.EtaGamma                          0.9
pfset Solver.Nonlinear.UseJacobian                       False
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-7
pfset Solver.Nonlinear.PrintFlag                         HighVerbosity
pfset Solver.Nonlinear.Globalization                     LineSearch

pfset Solver.Linear.KrylovDimension                      15
pfset Solver.Linear.MaxRestarts                          1

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      100

pfset Solver.PrintConcentration  False
pfset Solver.PrintPressure       True 
pfset Solver.PrintSaturation     True 
pfset Solver.PrintSubsurfData    True 
pfset Solver.PrintVelocities     False
pfset Solver.PrintWells          False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfrun $outname
pfundist $outname

#-----------------------------------------------------------------------------
# If running as test; check output.
# You do not need this for normal PF input files; this is done so the examples
# are run and checked as part of our testing process.
#-----------------------------------------------------------------------------
if { [info exists ::env(PF_TEST) ] } {
    set TEST $outname
    source pftest.tcl
    set sig_digits 4

    set passed 1

    #
    # Tests 
    #
    if ![pftestFile $TEST.out.press.00003.pfb "Max difference in Pressure" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.satur.00003.pfb "Max difference in Saturation" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.perm_x.pfb "Max difference in perm_x" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $TEST.out.perm_y.pfb "Max difference in perm_y" $sig_digits] {
	set passed 0
    }
    if ![pftestFile $TEST.out.perm_z.pfb "Max difference in perm_z" $sig_digits] {
	set passed 0
    }

    if ![pftestFile $TEST.out.porosity.pfb "Max difference in porosity" $sig_digits] {
	set passed 0
    }
    
    if $passed {
	puts "$TEST : PASSED"
    } {
	puts "$TEST : FAILED"
    }
}
