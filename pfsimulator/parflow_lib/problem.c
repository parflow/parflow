/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

/*****************************************************************************
*
* Routines for initializing the problem.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"
#include "problem.h"


/*--------------------------------------------------------------------------
 * NewProblem
 *--------------------------------------------------------------------------*/

Problem   *NewProblem(
                      int solver) /* Designates the solver from which this routine is
                                   * called.  Values defined in problem.h. */
{
  Problem      *problem;

  int version_number;
  int num_phases;
  int num_contaminants;

  char *phases;
  char *contaminants;

  int i;

  char key[IDB_MAX_KEY_LEN];

  problem = ctalloc(Problem, 1);

  /*-----------------------------------------------------------------------
   * Check the file version number
   *-----------------------------------------------------------------------*/

  version_number = GetInt("FileVersion");

  if (version_number != PFIN_VERSION)
  {
    amps_Printf("Error: need input file version %d\n", PFIN_VERSION);
    exit(1);
  }

  /*-----------------------------------------------------------------------
   * ProblemGeometries
   *-----------------------------------------------------------------------*/

  ProblemGeometries(problem) =
    PFModuleNewModule(Geometries, ());

  /*-----------------------------------------------------------------------
   * Setup timing information
   *-----------------------------------------------------------------------*/

  ProblemBaseTimeUnit(problem) = GetDouble("TimingInfo.BaseUnit");

  ProblemStartCount(problem) = GetInt("TimingInfo.StartCount");

  ProblemStartTime(problem) = GetDouble("TimingInfo.StartTime");
  CheckTime(problem, "TimingInfo.StartTime", ProblemStartTime(problem));

  ProblemStopTime(problem) = GetDouble("TimingInfo.StopTime");
  CheckTime(problem, "TimingInfo.StopTime", ProblemStopTime(problem));

  ProblemDumpInterval(problem) = GetDouble("TimingInfo.DumpInterval");
  CheckTime(problem, "TimingInfo.DumpInterval", ProblemDumpInterval(problem));

  ProblemDumpIntervalExecutionTimeLimit(problem) = GetIntDefault("TimingInfo.DumpIntervalExecutionTimeLimit", 0);

#ifndef HAVE_SLURM
  if (ProblemDumpIntervalExecutionTimeLimit(problem))
  {
    /*
     * SGS TODO should create a print warning/error function.   Can we use some standard logging library?
     */
    amps_Printf("Warning: TimingInfo.DumpIntervalExecutionTimeLimit is ignored if SLURM not linked with Parflow");
  }
#endif

  NameArray switch_na = NA_NewNameArray("False True");
  char *switch_name = GetStringDefault("TimingInfo.DumpAtEnd", "False");
  ProblemDumpAtEnd(problem) = NA_NameToIndexExitOnError(switch_na, switch_name, "TimingInfo.DumpAtEnd");
  NA_FreeNameArray(switch_na);

  /*-----------------------------------------------------------------------
   * Read in the cycle data
   *-----------------------------------------------------------------------*/
  ReadGlobalTimeCycleData();

  if (solver == RichardsSolve)
  {
    ProblemSelectTimeStep(problem) =
      PFModuleNewModule(SelectTimeStep, ());
  }

  /*-----------------------------------------------------------------------
   * ProblemDomain
   *-----------------------------------------------------------------------*/

  ProblemDomain(problem) =
    PFModuleNewModule(Domain, ());

  /*-----------------------------------------------------------------------
   * Setup ProblemNumPhases and ProblemNumContaminants
   *-----------------------------------------------------------------------*/

  phases = GetString("Phase.Names");
  GlobalsPhaseNames = NA_NewNameArray(phases);
  num_phases = ProblemNumPhases(problem) = NA_Sizeof(GlobalsPhaseNames);

  contaminants = GetString("Contaminants.Names");
  GlobalsContaminatNames = NA_NewNameArray(contaminants);
  num_contaminants = ProblemNumContaminants(problem) =
    NA_Sizeof(GlobalsContaminatNames);

  /*-----------------------------------------------------------------------
   * PDE coefficients
   *-----------------------------------------------------------------------*/

  ProblemGravity(problem) = GetDouble("Gravity");

  ProblemPhaseDensity(problem) =
    PFModuleNewModuleType(PhaseDensityNewPublicXtraInvoke,
                          PhaseDensity, (num_phases));

  problem->phase_viscosity = ctalloc(double, num_phases);
  for (i = 0; i < num_phases; i++)
  {
    /* SGS need to add switch on type */
    sprintf(key, "Phase.%s.Viscosity.Value",
            NA_IndexToName(GlobalsPhaseNames, i));
    problem->phase_viscosity[i] = GetDouble(key);
  }

  (problem->contaminant_degradation) = ctalloc(double, num_contaminants);


  for (i = 0; i < num_contaminants; i++)
  {
    /* SGS need to add switch on type */
    sprintf(key, "Contaminants.%s.Degradation.Value",
            NA_IndexToName(GlobalsContaminatNames, i));
    problem->contaminant_degradation[i] = GetDouble(key);
  }


  ProblemPermeability(problem) =
    PFModuleNewModule(Permeability, ());

  ProblemPorosity(problem) =
    PFModuleNewModule(Porosity, ());

  ProblemRetardation(problem) =
    PFModuleNewModuleType(RetardationNewPublicXtraInvoke,
                          Retardation, (num_contaminants));

  if (solver != RichardsSolve)
  {
    ProblemPhaseMobility(problem) =
      PFModuleNewModuleType(PhaseMobilityNewPublicXtraInvoke,
                            PhaseMobility, (num_phases));
  }
  else  /* Richards case */
  {
    ProblemPhaseRelPerm(problem) = PFModuleNewModule(PhaseRelPerm, ());
  }

  ProblemPhaseSource(problem) =
    PFModuleNewModuleType(PhaseSourceNewPublicXtraInvoke,
                          PhaseSource, (num_phases));

  ProblemSpecStorage(problem) =
    PFModuleNewModule(SpecStorage, ());   //sk

  ProblemXSlope(problem) =
    PFModuleNewModule(XSlope, ());   //sk

  ProblemYSlope(problem) =
    PFModuleNewModule(YSlope, ());   //sk

  ProblemXChannelWidth(problem) =
    PFModuleNewModule(XChannelWidth, ());

  ProblemYChannelWidth(problem) =
    PFModuleNewModule(YChannelWidth, ());

  ProblemMannings(problem) =
    PFModuleNewModule(Mannings, ());   //sk

  ProblemdzScale(problem) =
    PFModuleNewModule(dzScale, ()); //RMM

  ProblemFBx(problem) =
    PFModuleNewModule(FBx, ()); //RMM

  ProblemFBy(problem) =
    PFModuleNewModule(FBy, ()); //RMM

  ProblemFBz(problem) =
    PFModuleNewModule(FBz, ()); //RMM

  ProblemRealSpaceZ(problem) =
    PFModuleNewModule(realSpaceZ, ());


  ProblemOverlandFlowEval(problem) =
    PFModuleNewModule(OverlandFlowEval, ());   //DOK

  ProblemOverlandFlowEvalDiff(problem) =
    PFModuleNewModule(OverlandFlowEvalDiff, ()); //@RMM

  ProblemOverlandFlowEvalKin(problem) =
    PFModuleNewModule(OverlandFlowEvalKin, ());

  if (solver != RichardsSolve)
  {
    ProblemCapillaryPressure(problem) =
      PFModuleNewModuleType(CapillaryPressureNewPublicXtraInvoke,
                            CapillaryPressure, (num_phases));
  }
  else  /* Richards case */
  {
    ProblemSaturation(problem) =
      PFModuleNewModuleExtendedType(NewDefault, Saturation, ());
  }

  /*-----------------------------------------------------------------------
   * Boundary conditions
   *-----------------------------------------------------------------------*/

  if (solver != RichardsSolve)
  {
    ProblemBCInternal(problem) = PFModuleNewModule(BCInternal, ());
  }
  else
  {
    ProblemBCInternal(problem) =
      PFModuleNewModule(RichardsBCInternal, ());
  }

  ProblemBCPressure(problem) =
    PFModuleNewModuleType(BCPressureNewPublicXtraInvoke,
                          BCPressure, (num_phases));

  ProblemBCPressurePackage(problem) =
    PFModuleNewModuleType(BCPressurePackageNewPublicXtraInvoke,
                          BCPressurePackage, (num_phases));

  if (solver != RichardsSolve)
  {
    ProblemBCPhaseSaturation(problem) =
      PFModuleNewModuleType(BCPhaseSaturationNewPublicXtraInvoke,
                            BCPhaseSaturation, (num_phases));
  }

  /*-----------------------------------------------------------------------
   * Initial conditions
   *-----------------------------------------------------------------------*/

  if (solver != RichardsSolve)
  {
    ProblemICPhaseSatur(problem) =
      PFModuleNewModuleType(ICPhaseSaturNewPublicXtraInvoke,
                            ICPhaseSatur, (num_phases));
  }
  else
  {
    ProblemICPhasePressure(problem) =
      PFModuleNewModule(ICPhasePressure, ());
  }

  ProblemICPhaseConcen(problem) =
    PFModuleNewModuleType(ICPhaseConcenNewPublicXtraInvoke,
                          ICPhaseConcen, (num_phases, num_contaminants));

  /*-----------------------------------------------------------------------
   * If know exact solution for Richards' case, get data for which
   * predefined problem is being run
   *-----------------------------------------------------------------------*/

  if (solver == RichardsSolve)
  {
    ProblemL2ErrorNorm(problem) =
      PFModuleNewModule(L2ErrorNorm, ());
  }

  /*-----------------------------------------------------------------------
   * Constitutive relations
   *-----------------------------------------------------------------------*/

  if (solver != RichardsSolve)
  {
    ProblemSaturationConstitutive(problem) =
      PFModuleNewModuleType(SaturationConstitutiveNewPublicXtraInvoke,
                            SaturationConstitutive, (num_phases));
  }

  /*----------------------------------------------------------------------
   * Package setups
   *-----------------------------------------------------------------------*/

  ProblemWellPackage(problem) =
    PFModuleNewModuleType(WellPackageNewPublicXtraInvoke,
                          WellPackage, (num_phases, num_contaminants));

  ProblemReservoirPackage(problem) =
    PFModuleNewModuleType(ReservoirPackageNewPublicXtraInvoke,
                          ReservoirPackage, (num_phases, num_contaminants));

  return problem;
}


/*--------------------------------------------------------------------------
 * FreeProblem
 *--------------------------------------------------------------------------*/

void      FreeProblem(
                      Problem *problem,
                      int      solver)
{
  PFModuleFreeModule(ProblemWellPackage(problem));
  PFModuleFreeModule(ProblemReservoirPackage(problem));


  NA_FreeNameArray(GlobalsPhaseNames);
  NA_FreeNameArray(GlobalsContaminatNames);

  if (solver != RichardsSolve)
  {
    PFModuleFreeModule(ProblemSaturationConstitutive(problem));
    PFModuleFreeModule(ProblemICPhaseSatur(problem));
    PFModuleFreeModule(ProblemBCPhaseSaturation(problem));
    PFModuleFreeModule(ProblemCapillaryPressure(problem));
    PFModuleFreeModule(ProblemPhaseMobility(problem));
  }
  else
  {
    PFModuleFreeModule(ProblemSaturation(problem));
    PFModuleFreeModule(ProblemICPhasePressure(problem));
    PFModuleFreeModule(ProblemPhaseRelPerm(problem));
    PFModuleFreeModule(ProblemSelectTimeStep(problem));
    PFModuleFreeModule(ProblemL2ErrorNorm(problem));
  }
  PFModuleFreeModule(ProblemICPhaseConcen(problem));
  PFModuleFreeModule(ProblemBCPressurePackage(problem));
  PFModuleFreeModule(ProblemBCPressure(problem));
  PFModuleFreeModule(ProblemBCInternal(problem));

  PFModuleFreeModule(ProblemPhaseSource(problem));
  PFModuleFreeModule(ProblemRetardation(problem));
  PFModuleFreeModule(ProblemPorosity(problem));
  PFModuleFreeModule(ProblemPermeability(problem));
  tfree(problem->contaminant_degradation);
  PFModuleFreeModule(ProblemPhaseDensity(problem));
  tfree(problem->phase_viscosity);
  PFModuleFreeModule(ProblemSpecStorage(problem));  //sk
  PFModuleFreeModule(ProblemXSlope(problem));  //sk
  PFModuleFreeModule(ProblemYSlope(problem));
  PFModuleFreeModule(ProblemXChannelWidth(problem));
  PFModuleFreeModule(ProblemYChannelWidth(problem));
  PFModuleFreeModule(ProblemMannings(problem));
  PFModuleFreeModule(ProblemdzScale(problem));    //RMM
  PFModuleFreeModule(ProblemFBx(problem));    //RMM
  PFModuleFreeModule(ProblemFBy(problem));    //RMM
  PFModuleFreeModule(ProblemFBz(problem));    //RMM
  PFModuleFreeModule(ProblemRealSpaceZ(problem));


  PFModuleFreeModule(ProblemOverlandFlowEval(problem));  //DOK
  PFModuleFreeModule(ProblemOverlandFlowEvalDiff(problem));   //@RMM
  PFModuleFreeModule(ProblemOverlandFlowEvalKin(problem));

  PFModuleFreeModule(ProblemDomain(problem));

  PFModuleFreeModule(ProblemGeometries(problem));

  FreeGlobalTimeCycleData();

  tfree(problem);
}


/*--------------------------------------------------------------------------
 * NewProblemData
 *--------------------------------------------------------------------------*/

ProblemData   *NewProblemData(
                              Grid *grid,
                              Grid *grid2d)
{
  ProblemData  *problem_data;

  problem_data = ctalloc(ProblemData, 1);

  ProblemDataPermeabilityX(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);
  ProblemDataPermeabilityY(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);
  ProblemDataPermeabilityZ(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);

  ProblemDataSpecificStorage(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);  //sk
  ProblemDataTSlopeX(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);   //sk
  ProblemDataTSlopeY(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);   //sk
  ProblemDataChannelWidthX(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
  ProblemDataChannelWidthY(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
  ProblemDataMannings(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);  //sk

  /* @RMM added vectors for subsurface slopes for terrain-following grid */
  ProblemDataSSlopeX(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);   //RMM
  ProblemDataSSlopeY(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);   //RMM

  /* @RMM added vector dz multiplier */
  ProblemDataZmult(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);   //RMM

  /* @RMM added flow barrier in X, Y, Z  */
  /* these values are on the cell edges e.g. [ip] is really from i to i+1 or i+1/2 */
  ProblemDataFBx(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);   //RMM
  ProblemDataFBy(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);   //RMM
  ProblemDataFBz(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);   //RMM

  ProblemDataRealSpaceZ(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);

  ProblemDataIndexOfDomainTop(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);
  ProblemDataPatchIndexOfDomainTop(problem_data) = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);

  ProblemDataPorosity(problem_data) = NewVectorType(grid, 1, 1, vector_cell_centered);

  ProblemDataBCPressureData(problem_data) = NewBCPressureData();

  ProblemDataWellData(problem_data) = NewWellData();
  ProblemDataReservoirData(problem_data) = NewReservoirData();

  return problem_data;
}


/*--------------------------------------------------------------------------
 * FreeProblemData
 *--------------------------------------------------------------------------*/

void          FreeProblemData(
                              ProblemData *problem_data)
{
  int i;


  if (problem_data)
  {
#if 1
    /* SGS This is freed in problem_geometries.c where it is
     * created */
    for (i = 0; i < ProblemDataNumSolids(problem_data); i++)
      GrGeomFreeSolid(ProblemDataGrSolids(problem_data)[i]);
    tfree(ProblemDataGrSolids(problem_data));
#endif

    FreeWellData(ProblemDataWellData(problem_data));
    FreeReservoirData(ProblemDataReservoirData(problem_data));
    FreeBCPressureData(ProblemDataBCPressureData(problem_data));
    FreeVector(ProblemDataPorosity(problem_data));
    FreeVector(ProblemDataPermeabilityX(problem_data));
    FreeVector(ProblemDataPermeabilityY(problem_data));
    FreeVector(ProblemDataPermeabilityZ(problem_data));
    FreeVector(ProblemDataSpecificStorage(problem_data));   //sk
    FreeVector(ProblemDataTSlopeX(problem_data));   //sk
    FreeVector(ProblemDataTSlopeY(problem_data));   //sk
    FreeVector(ProblemDataChannelWidthX(problem_data));
    FreeVector(ProblemDataChannelWidthY(problem_data));
    FreeVector(ProblemDataMannings(problem_data));   //sk
    FreeVector(ProblemDataSSlopeX(problem_data));   //RMM
    FreeVector(ProblemDataSSlopeY(problem_data));   //RMM
    FreeVector(ProblemDataZmult(problem_data));   //RMM
    FreeVector(ProblemDataFBx(problem_data));   //RMM
    FreeVector(ProblemDataFBy(problem_data));   //RMM
    FreeVector(ProblemDataFBz(problem_data));   //RMM

    FreeVector(ProblemDataRealSpaceZ(problem_data));
    FreeVector(ProblemDataIndexOfDomainTop(problem_data));
    FreeVector(ProblemDataPatchIndexOfDomainTop(problem_data));

    tfree(problem_data);
  }
}

