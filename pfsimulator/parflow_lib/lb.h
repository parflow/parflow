/*****************************************************************************
* Header file for lattice Boltzmann diffusion equation solver.
*
*
*
* $Revision: 1.1.1.1 $
*****************************************************************************/

#ifndef _LB_HEADER
#define _LB_HEADER

/*-----------------------------*
* Include files
*-----------------------------*/
#include "char_vector.h"

/*-----------------------------*
* Global Variables            *
*-----------------------------*/
#define nDirections 19
#define GeomMean(a, b)   (((a) + (b)) ? (sqrt((a) * (b))) : 0)
#define HarMean(a, b)    (((a) + (b)) ? (2.0 * (a) * (b)) / ((a) + (b)) : 0)

/*-----------------------------*
* Flags                       *
*-----------------------------*/
#define SOLVE_EQUILIBRIUM 0

/*-----------------------------*
* Physical constants          *
*-----------------------------*/

/*      name                    value           MKS unit        */
/*      ----                    -----           --------        */
#define PERM_COMPLIANCE         400.0e-9        /* Pa^-1        */
#define PORE_COMPRESSIBILITY    1.6e-9          /* Pa^-1        */
#define FLUID_COMPRESSIBILITY   0.4e-9          /* Pa^-1        */
#define FLUID_DENSITY           1.0e3           /* kg/m^3       */
#define VISCOSITY               1.0e-3          /* Pa-s         */
#define RHO_hydrostatic         1.0e3           /* kg/m^3       */
#define RHO_lithostatic         2.7e3           /* kg/m^3       */
#define RHO RHO_hydrostatic


/*-----------------------------*
* Structures                  *
*-----------------------------*/
typedef struct {
  /* Lattice grid structure */
  Grid         *grid;
  double       **e;
  double       *c;

  /* Constitutive properties and cell types */
  Vector       *pressure;
  Vector       *perm;
  Vector       *phi;
  double       **Ktensor;
  double       *Kscale;
  CharVector   *cellType;
  double beta_perm;
  double beta_pore;
  double beta_fracture;
  double beta_fluid;
  double viscosity;
  double density;

  /* Work space needed for updating the lattice */
  Vector       *pwork;

  /* Applied body force */
  double       *bforce;

  /* Time information */
  double t;
  double start;
  double stop;
  double step;
  double dump;
  double tscale;
  double cfl;

  /* Miscellaneous */
  int comp_compress_flag;
} Lattice;




#endif
