# Trying to compute - \Delta p = 0 in one dimension with solution p = x.
# using pfb input for both ICPressure and BCPressure

# Import the ParFlow TCL package
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

# remove files from previous run
foreach file [glob -nocomplain *prueba.out.*] {file delete -force -- $file}
foreach file [glob -nocomplain prueba.pfidb] {file delete -force -- $file}
foreach file [glob -nocomplain icpressure.pfb*] {file delete -force -- $file}


pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                  0.0
pfset ComputationalGrid.Lower.Y                  0.0
pfset ComputationalGrid.Lower.Z                  0.0

pfset ComputationalGrid.DX	                 0.2
pfset ComputationalGrid.DY                       0.2
pfset ComputationalGrid.DZ	                 0.2

pfset ComputationalGrid.NX                       8
pfset ComputationalGrid.NY                       8
pfset ComputationalGrid.NZ                       1

#---------------------------------------------------------
# Computational SubGrid dims
#---------------------------------------------------------
pfset ComputationalSubgrid.MX                    8
pfset ComputationalSubgrid.MY                    8
pfset ComputationalSubgrid.MZ                    1

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input background_input"

#---------------------------------------------------------
# Known Solution
#---------------------------------------------------------
pfset KnownSolution                                   2D_XY

#--------------------------------------------------------
# prepare ICPressure File which has the initial values for pressure
#---------------------------------------------------------
# set icp [pfload cosh_icp_ascii.sa]
# pfsave $icp -pfb icpressure.pfb
# pfdist icpressure.pfb


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X                          0.0 
pfset Geom.domain.Lower.Y                          0.0
pfset Geom.domain.Lower.Z                          0.0

pfset Geom.domain.Upper.X                          1.0
pfset Geom.domain.Upper.Y                          1.0
pfset Geom.domain.Upper.Z                          1.0

pfset Geom.domain.Patches "left right front back bottom top"

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
pfset GeomInput.background_input.InputType         Box
pfset GeomInput.background_input.GeomName          background

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
pfset Geom.background.Lower.X -99999999.0
pfset Geom.background.Lower.Y -99999999.0
pfset Geom.background.Lower.Z -99999999.0

pfset Geom.background.Upper.X  99999999.0
pfset Geom.background.Upper.Y  99999999.0
pfset Geom.background.Upper.Z  99999999.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "background"

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    1.0

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "background"

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	Constant
pfset Phase.water.Density.Value	1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names			""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames           ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit		1.0
pfset TimingInfo.StartCount		0
pfset TimingInfo.StartTime		0.0
pfset TimingInfo.StopTime               1.0
pfset TimingInfo.DumpInterval	       -1
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          background

pfset Geom.background.Porosity.Type    Constant
pfset Geom.background.Porosity.Value   1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
pfset Phase.RelPerm.Type               Constant
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Value         1.0


#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type            Constant
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Value     1.0

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
pfset BCPressure.PatchNames "left right front back bottom top"

pfset Patch.left.BCPressure.Type			ExactSolution
pfset Patch.left.BCPressure.Cycle			"constant"
pfset Patch.left.BCPressure.alltime.PredefinedFunction	2D_XY

pfset Patch.right.BCPressure.Type			ExactSolution
pfset Patch.right.BCPressure.Cycle			"constant"
pfset Patch.right.BCPressure.alltime.PredefinedFunction 2D_XY

pfset Patch.front.BCPressure.Type			ExactSolution
pfset Patch.front.BCPressure.Cycle			"constant"
pfset Patch.front.BCPressure.alltime.PredefinedFunction  2D_XY

pfset Patch.back.BCPressure.Type			ExactSolution
pfset Patch.back.BCPressure.Cycle			"constant"
pfset Patch.back.BCPressure.alltime.PredefinedFunction  2D_XY

# pfset Patch.front.BCPressure.Type			FluxConst
# pfset Patch.front.BCPressure.Cycle			"constant"
# pfset Patch.front.BCPressure.alltime.Value              0.0
# 
# pfset Patch.back.BCPressure.Type			FluxConst
# pfset Patch.back.BCPressure.Cycle			"constant"
# pfset Patch.back.BCPressure.alltime.Value               0.0
# 
pfset Patch.bottom.BCPressure.Type			FluxConst
pfset Patch.bottom.BCPressure.Cycle			"constant"
pfset Patch.bottom.BCPressure.alltime.Value             0.0

pfset Patch.top.BCPressure.Type			        FluxConst
pfset Patch.top.BCPressure.Cycle			"constant"
pfset Patch.top.BCPressure.alltime.Value                0.0


#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type 					Constant
pfset ICPressure.GeomNames				domain
pfset Geom.domain.ICPressure.Value       		1.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                   "domain"
pfset PhaseSources.water.Geom.domain.Value             0.0


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     5000

pfset Solver.Nonlinear.MaxIter                           1500
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaValue                          1e-2
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-8

pfset Solver.Linear.KrylovDimension                      10
pfset Solver.Linear.Preconditioner                       MGSemi

pfset Solver.Linear.Preconditioner.SymmetricMat          Symmetric
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      100

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
pfset Mannings.Geom.domain.Value 0.0
 
#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain"
pfset Geom.domain.SpecificStorage.Value 0.0

#-----------------------------------------------------------------------------
# Write out data base
#-----------------------------------------------------------------------------
pfwritedb test_brick

