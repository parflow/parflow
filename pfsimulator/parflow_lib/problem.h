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

#ifndef _PROBLEM_HEADER
#define _PROBLEM_HEADER

/*----------------------------------------------------------------
 * Problem structure
 *----------------------------------------------------------------*/

typedef struct {
  PFModule   *geometries;        /* All geometry info input here */

  PFModule   *domain;

  int num_phases;

  int num_contaminants;

  double base_time_unit;
  int start_count;
  double start_time;
  double stop_time;
  double dump_interval;

  /* Stop execution if remaining time is less than user specified
   * value (s) */
  int dump_interval_execution_time_limit;

  int dump_at_end;                      /* write out files at end of simulation */

  /* Time step info */
  PFModule   *select_time_step;         /* Selects time steps used in
                                         * SolverRichards */
  /* PDE coefficients */
  double gravity;
  double     *phase_viscosity;          /* array of size num_phases */
  double     *contaminant_degradation;  /* array of size num_contaminants */
  PFModule   *phase_density;
  PFModule   *permeability;
  PFModule   *porosity;
  PFModule   *retardation;
  PFModule   *phase_mobility;
  PFModule   *phase_rel_perm;           /* relative permeability used in
                                         * SolverRichards */
  PFModule   *phase_source;
  PFModule   *specific_storage;         //sk
  PFModule   *FBx;         //rmm flow barrier multipliers in X, Y, Z
  PFModule   *FBy;         //rmm
  PFModule   *FBz;

  PFModule   *capillary_pressure;
  PFModule   *saturation;               /* saturation function used in
                                         * SolverRichards */

  /* boundary conditions */
  PFModule   *bc_internal;
  PFModule   *bc_pressure;
  PFModule   *bc_pressure_package;
  PFModule   *bc_phase_saturation;      /* RDF assume Dirichlet from IC */

  /* initial conditions */
  PFModule   *ic_phase_concen;
  PFModule   *ic_phase_satur;
  PFModule   *ic_phase_pressure;        /* Pressure initial cond. used by
                                         * SolverRichards */

  /* error calculations */
  PFModule  *l2_error_norm;             /* Error calculation used for known
                                         * solution cases for SolverRichards */

  /* constitutive relations */
  PFModule   *constitutive;

  /*****  packages  *****/
  PFModule  *well_package;
  PFModule  *reservoir_package;

  /*sk**  overland flow*/
  PFModule  *x_slope;
  PFModule  *y_slope;
  PFModule  *wc_x;
  PFModule  *wc_y;
  PFModule  *mann;
  PFModule  *overlandflow_eval;        //DOK
  PFModule  *overlandflow_eval_diff;         //@RMM
  PFModule  *overlandflow_eval_kin;  //@MCB

  /* @RMM Variable dZ */
  PFModule  *dz_mult;           //rmm
  PFModule  *real_space_z;
} Problem;

typedef struct {
  /* geometry information */
  int num_solids;
  GeomSolid     **solids;
  GrGeomSolid   **gr_solids;

  GeomSolid      *domain;
  GrGeomSolid    *gr_domain;

  /*
   * This is a NX * NY vector of Z indices to the top
   * of the domain.
   *
   * -1 means domain is not present at that i,j index.
   */
  Vector         *index_of_domain_top;

  /*
   * This is a NX * NY vector of patch id/index of the top
   * of the domain.
   *
   * -1 means domain is not present at the i,j index.
   */
  Vector         *patch_index_of_domain_top;

  /*
   * This is a NX * NY vector of Z indices to the bottom
   * of the domain.
   *
   * -1 means domain is not present at that i,j index.
   */
  Vector         *index_of_domain_bottom;

  Vector         *permeability_x;
  Vector         *permeability_y;
  Vector         *permeability_z;

  Vector         *porosity;

  Vector         *specific_storage;   //sk

  Vector         *FBx;  //RMM
  Vector         *FBy;  //RMM
  Vector         *FBz;  //RMM


  WellData       *well_data;
  ReservoirData       *reservoir_data;
  BCPressureData *bc_pressure_data;

  /*sk  overland flow*/
  Vector *x_slope;
  Vector *y_slope;
  Vector *wc_x;
  Vector *wc_y;
  Vector *mann;

  /* @RMM terrain grid */
  Vector *x_sslope;
  Vector *y_sslope;

  /* @RMM variable dz  */
  Vector *dz_mult;
  Vector *rsz;
} ProblemData;

/* Values of solver argument to NewProblem function */

#define ImpesSolve     0
#define DiffusionSolve 1
#define RichardsSolve  2


/*--------------------------------------------------------------------------
 * Accessor macros: Problem
 *--------------------------------------------------------------------------*/

#define ProblemGeometries(problem)                ((problem)->geometries)

#define ProblemDomain(problem)                    ((problem)->domain)

#define ProblemNumPhases(problem)                 ((problem)->num_phases)

#define ProblemContaminants(problem)              ((problem)->contaminants)
#define ProblemNumContaminants(problem)           ((problem)->num_contaminants)

/* Time accessors */
#define ProblemBaseTimeUnit(problem)              ((problem)->base_time_unit)
#define ProblemStartCount(problem)                ((problem)->start_count)
#define ProblemStartTime(problem)                 ((problem)->start_time)
#define ProblemStopTime(problem)                  ((problem)->stop_time)
#define ProblemDumpInterval(problem)              ((problem)->dump_interval)
#define ProblemDumpIntervalExecutionTimeLimit(problem)              ((problem)->dump_interval_execution_time_limit)
#define ProblemDumpAtEnd(problem)                 ((problem)->dump_at_end)
#define ProblemSelectTimeStep(problem)            ((problem)->select_time_step)


/* PDE accessors */
#define ProblemGravity(problem)                   ((problem)->gravity)
#define ProblemPhaseDensity(problem)              ((problem)->phase_density)
#define ProblemPhaseViscosities(problem)          ((problem)->phase_viscosity)
#define ProblemPhaseViscosity(problem, i)         ((problem)->phase_viscosity[i])
#define ProblemContaminantDegradations(problem)   ((problem)->contaminant_degradation)
#define ProblemContaminantDegradation(problem, i) ((problem)->contaminant_degradation[i])
#define ProblemPermeability(problem)              ((problem)->permeability)
#define ProblemPorosity(problem)                  ((problem)->porosity)
#define ProblemRetardation(problem)               ((problem)->retardation)
#define ProblemPhaseMobility(problem)             ((problem)->phase_mobility)
#define ProblemPhaseRelPerm(problem)              ((problem)->phase_rel_perm)
#define ProblemPhaseSource(problem)               ((problem)->phase_source)
#define ProblemCapillaryPressure(problem)         ((problem)->capillary_pressure)
#define ProblemSaturation(problem)                ((problem)->saturation)
#define ProblemBCInternal(problem)                ((problem)->bc_internal)
#define ProblemSpecStorage(problem)               ((problem)->specific_storage)   //sk
#define ProblemXSlope(problem)                    ((problem)->x_slope)   //sk
#define ProblemYSlope(problem)                    ((problem)->y_slope)   //sk
#define ProblemXChannelWidth(problem)             ((problem)->wc_x)
#define ProblemYChannelWidth(problem)             ((problem)->wc_y)
#define ProblemFBx(problem)                      ((problem)->FBx)    //RMM
#define ProblemFBy(problem)                      ((problem)->FBy)    //RMM
#define ProblemFBz(problem)                      ((problem)->FBz)    //RMM

#define ProblemMannings(problem)                  ((problem)->mann)   //sk

#define ProblemOverlandFlowEval(problem)          ((problem)->overlandflow_eval)   //DOK
#define ProblemOverlandFlowEvalDiff(problem)          ((problem)->overlandflow_eval_diff)   //@RMM
#define ProblemOverlandFlowEvalKin(problem)  ((problem)->overlandflow_eval_kin) //@MCB

#define ProblemdzScale(problem)            ((problem)->dz_mult)    //RMM
#define ProblemRealSpaceZ(problem)            ((problem)->real_space_z)

/* boundary condition accessors */
#define ProblemBCPressure(problem)                ((problem)->bc_pressure)
#define ProblemBCPressurePackage(problem)         ((problem)->bc_pressure_package)
#define ProblemBCPhaseSaturation(problem)         ((problem)->bc_phase_saturation)

/* initial condition accessors */
#define ProblemICPhaseConcen(problem)             ((problem)->ic_phase_concen)
#define ProblemICPhaseSatur(problem)              ((problem)->ic_phase_satur)
#define ProblemICPhasePressure(problem)           ((problem)->ic_phase_pressure)

/* constitutive relations */
#define ProblemSaturationConstitutive(problem)    ((problem)->constitutive)

/* packages */
#define ProblemWellPackage(problem)               ((problem)->well_package)
#define ProblemReservoirPackage(problem)               ((problem)->reservoir_package)

/* error calculations */
#define ProblemL2ErrorNorm(problem)               ((problem)->l2_error_norm)

/*--------------------------------------------------------------------------
 * Accessor macros: ProblemData
 *--------------------------------------------------------------------------*/

#define ProblemDataNumSolids(problem_data)      ((problem_data)->num_solids)
#define ProblemDataSolids(problem_data)         ((problem_data)->solids)
#define ProblemDataGrSolids(problem_data)       ((problem_data)->gr_solids)
#define ProblemDataSolid(problem_data, i)       ((problem_data)->solids[i])
#define ProblemDataGrSolid(problem_data, i)     ((problem_data)->gr_solids[i])

#define ProblemDataDomain(problem_data)         ((problem_data)->domain)
#define ProblemDataGrDomain(problem_data)       ((problem_data)->gr_domain)

#define ProblemDataIndexOfDomainTop(problem_data)  ((problem_data)->index_of_domain_top)
#define ProblemDataPatchIndexOfDomainTop(problem_data)  ((problem_data)->patch_index_of_domain_top)
#define ProblemDataIndexOfDomainBottom(problem_data)  ((problem_data)->index_of_domain_bottom)

#define ProblemDataPermeabilityX(problem_data)  ((problem_data)->permeability_x)
#define ProblemDataPermeabilityY(problem_data)  ((problem_data)->permeability_y)
#define ProblemDataPermeabilityZ(problem_data)  ((problem_data)->permeability_z)
#define ProblemDataPorosity(problem_data)       ((problem_data)->porosity)
#define ProblemDataFBx(problem_data)            ((problem_data)->FBx)    //RMM
#define ProblemDataFBy(problem_data)            ((problem_data)->FBy)    //RMM
#define ProblemDataFBz(problem_data)            ((problem_data)->FBz)    //RMM
#define ProblemDataWellData(problem_data)       ((problem_data)->well_data)
#define ProblemDataReservoirData(problem_data)       ((problem_data)->reservoir_data)
#define ProblemDataBCPressureData(problem_data) ((problem_data)->bc_pressure_data)
#define ProblemDataSpecificStorage(problem_data)((problem_data)->specific_storage)   //sk
#define ProblemDataTSlopeX(problem_data)        ((problem_data)->x_slope)   //sk
#define ProblemDataTSlopeY(problem_data)        ((problem_data)->y_slope)   //sk
#define ProblemDataChannelWidthX(problem_data)  ((problem_data)->wc_x)
#define ProblemDataChannelWidthY(problem_data)  ((problem_data)->wc_y)
#define ProblemDataMannings(problem_data)       ((problem_data)->mann)   //sk
#define ProblemDataSSlopeX(problem_data)        ((problem_data)->x_sslope)   //RMM
#define ProblemDataSSlopeY(problem_data)        ((problem_data)->y_sslope)   //RMM
#define ProblemDataZmult(problem_data)          ((problem_data)->dz_mult)    //RMM
#define ProblemDataRealSpaceZ(problem_data)     ((problem_data)->rsz)
/*--------------------------------------------------------------------------
 * Misc macros
 *   RDF not quite right, maybe?
 *--------------------------------------------------------------------------*/

#define Permeability                   SubsrfSim
#define PermeabilityInitInstanceXtra   SubsrfSimInitInstanceXtra
#define PermeabilityFreeInstanceXtra   SubsrfSimFreeInstanceXtra
#define PermeabilityNewPublicXtra      SubsrfSimNewPublicXtra
#define PermeabilityFreePublicXtra     SubsrfSimFreePublicXtra
#define PermeabilitySizeOfTempData     SubsrfSimSizeOfTempData


#endif

