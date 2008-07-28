/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _PROBLEM_HEADER
#define _PROBLEM_HEADER

/*----------------------------------------------------------------
 * Problem structure
 *----------------------------------------------------------------*/

typedef struct
{
   PFModule   *geometries;       /* All geometry info input here */

   PFModule   *domain;

   int         num_phases;

   int         num_contaminants;

   double      base_time_unit;
   int         start_count;
   double      start_time;
   double      stop_time;
   double      dump_interval;

   /* Time step info */
   PFModule   *select_time_step;        /* Selects time steps used in
					   SolverRichards */
   /* PDE coefficients */
   double      gravity;
   double     *phase_viscosity;         /* array of size num_phases */
   double     *contaminant_degradation; /* array of size num_contaminants */
   PFModule   *phase_density; 
   PFModule   *permeability;
   PFModule   *porosity;
   PFModule   *retardation;
   PFModule   *phase_mobility;
   PFModule   *phase_rel_perm;          /* relative permeability used in 
                                           SolverRichards */
   PFModule   *phase_source;
   PFModule   *specific_storage;        //sk
   PFModule   *capillary_pressure;
   PFModule   *saturation;              /* saturation function used in 
                                           SolverRichards */

   /* boundary conditions */
   PFModule   *bc_internal;
   PFModule   *bc_pressure;
   PFModule   *bc_pressure_package;
   PFModule   *bc_phase_saturation;     /* RDF assume Dirichlet from IC */

   /* initial conditions */
   PFModule   *ic_phase_concen;
   PFModule   *ic_phase_satur;
   PFModule   *ic_phase_pressure;       /* Pressure initial cond. used by
					   SolverRichards */

   /* error calculations */             
   PFModule  *l2_error_norm;            /* Error calculation used for known
					   solution cases for SolverRichards */

   /* constitutive relations */
   PFModule   *constitutive;

   /*****  packages  *****/
   PFModule  *well_package;

   /*sk**  overland flow*/
   PFModule  *x_slope;
   PFModule  *y_slope;
   PFModule  *mann;

} Problem;

typedef struct
{
   /* geometry information */
   int             num_solids;
   GeomSolid     **solids;
   GrGeomSolid   **gr_solids;

   GeomSolid      *domain;
   GrGeomSolid    *gr_domain;

   Vector         *permeability_x;
   Vector         *permeability_y;
   Vector         *permeability_z;

   Vector         *porosity;

   Vector         *specific_storage;  //sk

   WellData       *well_data;
   BCPressureData *bc_pressure_data;
   
   /*sk  overland flow*/
   Vector *x_slope;
   Vector *y_slope;
   Vector *mann;

} ProblemData;

/* Values of solver argument to NewProblem function */

#define ImpesSolve     0
#define DiffusionSolve 1
#define RichardsSolve  2


/*--------------------------------------------------------------------------
 * Accessor macros: Problem
 *--------------------------------------------------------------------------*/

#define ProblemGeometries(problem)                ((problem) -> geometries)

#define ProblemDomain(problem)                    ((problem) -> domain)

#define ProblemNumPhases(problem)                 ((problem) -> num_phases)

#define ProblemContaminants(problem)              ((problem) -> contaminants)
#define ProblemNumContaminants(problem)           ((problem) -> num_contaminants)

/* Time accessors */
#define ProblemBaseTimeUnit(problem)              ((problem) -> base_time_unit)
#define ProblemStartCount(problem)                ((problem) -> start_count)
#define ProblemStartTime(problem)                 ((problem) -> start_time)
#define ProblemStopTime(problem)                  ((problem) -> stop_time)
#define ProblemDumpInterval(problem)              ((problem) -> dump_interval)
#define ProblemSelectTimeStep(problem)            ((problem) -> select_time_step)
				       
/* PDE accessors */
#define ProblemGravity(problem)                   ((problem) -> gravity)
#define ProblemPhaseDensity(problem)              ((problem) -> phase_density)
#define ProblemPhaseViscosities(problem)          ((problem) -> phase_viscosity)
#define ProblemPhaseViscosity(problem, i)         ((problem) -> phase_viscosity[i])
#define ProblemContaminantDegradations(problem)   ((problem) -> contaminant_degradation)
#define ProblemContaminantDegradation(problem, i) ((problem) -> contaminant_degradation[i])
#define ProblemPermeability(problem)              ((problem) -> permeability)
#define ProblemPorosity(problem)                  ((problem) -> porosity)
#define ProblemRetardation(problem)               ((problem) -> retardation)
#define ProblemPhaseMobility(problem)             ((problem) -> phase_mobility)
#define ProblemPhaseRelPerm(problem)              ((problem) -> phase_rel_perm)
#define ProblemPhaseSource(problem)               ((problem) -> phase_source)
#define ProblemCapillaryPressure(problem)         ((problem) -> capillary_pressure)
#define ProblemSaturation(problem)                ((problem) -> saturation)
#define ProblemBCInternal(problem)                ((problem) -> bc_internal)
#define ProblemSpecStorage(problem)               ((problem) -> specific_storage) //sk
#define ProblemXSlope(problem)                    ((problem) -> x_slope) //sk
#define ProblemYSlope(problem)                    ((problem) -> y_slope) //sk
#define ProblemMannings(problem)                  ((problem) -> mann) //sk

/* boundary condition accessors */
#define ProblemBCPressure(problem)                ((problem) -> bc_pressure)
#define ProblemBCPressurePackage(problem)         ((problem) -> bc_pressure_package)
#define ProblemBCPhaseSaturation(problem)         ((problem) -> bc_phase_saturation)

/* initial condition accessors */
#define ProblemICPhaseConcen(problem)             ((problem) -> ic_phase_concen)
#define ProblemICPhaseSatur(problem)              ((problem) -> ic_phase_satur)
#define ProblemICPhasePressure(problem)           ((problem) -> ic_phase_pressure)

/* constitutive relations */
#define ProblemSaturationConstitutive(problem)    ((problem) -> constitutive)

/* packages */
#define ProblemWellPackage(problem)               ((problem) -> well_package)

/* error calculations */
#define ProblemL2ErrorNorm(problem)               ((problem) -> l2_error_norm)

/*--------------------------------------------------------------------------
 * Accessor macros: ProblemData
 *--------------------------------------------------------------------------*/

#define ProblemDataNumSolids(problem_data)      ((problem_data) -> num_solids)
#define ProblemDataSolids(problem_data)         ((problem_data) -> solids)
#define ProblemDataGrSolids(problem_data)       ((problem_data) -> gr_solids)
#define ProblemDataSolid(problem_data, i)       ((problem_data) -> solids[i])
#define ProblemDataGrSolid(problem_data, i)     ((problem_data) -> gr_solids[i])

#define ProblemDataDomain(problem_data)         ((problem_data) -> domain)
#define ProblemDataGrDomain(problem_data)       ((problem_data) -> gr_domain)

#define ProblemDataPermeabilityX(problem_data)  ((problem_data) -> permeability_x)
#define ProblemDataPermeabilityY(problem_data)  ((problem_data) -> permeability_y)
#define ProblemDataPermeabilityZ(problem_data)  ((problem_data) -> permeability_z)
#define ProblemDataPorosity(problem_data)       ((problem_data) -> porosity)
#define ProblemDataWellData(problem_data)       ((problem_data) -> well_data)
#define ProblemDataBCPressureData(problem_data) ((problem_data) -> bc_pressure_data)
#define ProblemDataSpecificStorage(problem_data)((problem_data) -> specific_storage) //sk
#define ProblemDataTSlopeX(problem_data)        ((problem_data) -> x_slope) //sk
#define ProblemDataTSlopeY(problem_data)        ((problem_data) -> y_slope) //sk
#define ProblemDataMannings(problem_data)       ((problem_data) -> mann) //sk

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
