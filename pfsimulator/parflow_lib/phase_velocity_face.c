/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
*****************************************************************************/

#include "parflow.h"

#define Func(a, b) HarmonicMean(a, b)
#define FuncOps    4

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int time_index;
} PublicXtra;

typedef struct {
  PFModule          *phase_mobility;
  PFModule          *phase_density;
  PFModule          *capillary_pressure;

  Problem           *problem;
  Grid              *grid;
  Grid              *x_grid;
  Grid              *y_grid;
  Grid              *z_grid;
  double            *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * PhaseVelocityFace
 *--------------------------------------------------------------------------*/

void          PhaseVelocityFace(
                                Vector *     xvel,
                                Vector *     yvel,
                                Vector *     zvel,
                                ProblemData *problem_data,
                                Vector *     pressure,
                                Vector **    saturations,
                                int          phase)
{
  PFModule       *this_module = ThisPFModule;
  InstanceXtra   *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra     *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  PFModule       *phase_mobility = (instance_xtra->phase_mobility);
  PFModule       *phase_density = (instance_xtra->phase_density);
  PFModule       *capillary_pressure = (instance_xtra->capillary_pressure);

  Problem        *problem = (instance_xtra->problem);
  Grid           *grid = (instance_xtra->grid);
  Grid           *x_grid = (instance_xtra->x_grid);
  Grid           *y_grid = (instance_xtra->y_grid);
  Grid           *z_grid = (instance_xtra->z_grid);

  Vector         *temp_mobility_x = NULL;
  Vector         *temp_mobility_y = NULL;
  Vector         *temp_mobility_z = NULL;
  Vector         *temp_pressure = NULL;
  Vector         *temp_density = NULL;

  GrGeomSolid    *gr_domain = ProblemDataGrDomain(problem_data);

  SubgridArray   *subgrids;
  Subgrid        *subgrid;

  Subvector      *subvector_p, *subvector_m, *subvector_v, *subvector_d;

  int r;
  int ix, iy, iz;
  int nx, ny, nz;
  double dx, dy, dz;

  int nx_p, ny_p, nz_p;
  int nx_m, ny_m, nz_m;
  int nx_v, ny_v, nz_v;
  int nx_d, ny_d, nz_d;

  int pi, mi, vi, di;

  int            *fdir;

  int ipatch, sg, i, j, k;
  int flopest;

  double         *pl, *pu, *ml, *mu, *vel, *dl, *du;

  VectorUpdateCommHandle     *handle;

  Vector *pressure_vector, *vel_vec[3];
  Subvector *subvector_v0;
  double *vel0_l, *vel0_r, *vel_tmp;
  double dummy_density;
  int dir0 = 0;
  int dir[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };


  /*----------------------------------------------------------------------
   * Begin timing
   *----------------------------------------------------------------------*/

  BeginTiming(public_xtra->time_index);

  /*----------------------------------------------------------------------
   * Allocate temp vectors
   *----------------------------------------------------------------------*/
  temp_mobility_x = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  temp_mobility_y = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  temp_mobility_z = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  temp_pressure = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  temp_density = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /*----------------------------------------------------------------------
   * compute the mobility values for this phase
   *----------------------------------------------------------------------*/

  PFModuleInvokeType(PhaseMobilityInvoke, phase_mobility,
                     (temp_mobility_x, temp_mobility_y, temp_mobility_z,
                      ProblemDataPermeabilityX(problem_data),
                      ProblemDataPermeabilityY(problem_data),
                      ProblemDataPermeabilityZ(problem_data),
                      phase, saturations[phase],
                      ProblemPhaseViscosity(problem, phase)));

  /*----------------------------------------------------------------------
   * exchange boundary data for mobility values
   *----------------------------------------------------------------------*/

  handle = InitVectorUpdate(temp_mobility_x, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(temp_mobility_y, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
  handle = InitVectorUpdate(temp_mobility_z, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /*----------------------------------------------------------------------
   * if necessary compute the pressure values for this phase
   *----------------------------------------------------------------------*/

  if (phase == 0)
  {
    pressure_vector = pressure;
  }
  else
  {
    PFModuleInvokeType(CapillaryPressureInvoke, capillary_pressure,
                       (temp_pressure, phase, 0,
                        problem_data, saturations[0]));

    Axpy(1.0, pressure, temp_pressure);

    pressure_vector = temp_pressure;

    IncFLOPCount(2 * SizeOfVector(pressure));
  }

  /*----------------------------------------------------------------------
   * exchange boundary data for pressure values
   *----------------------------------------------------------------------*/

  handle = InitVectorUpdate(pressure_vector, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /*----------------------------------------------------------------------
   * compute the density values for this phase
   *----------------------------------------------------------------------*/

  PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                     (phase, pressure_vector, temp_density, &dummy_density,
                      &dummy_density, CALCFCN));

  /*-----------------------------------------------------------------------
   * exchange boundary data for density values
   *-----------------------------------------------------------------------*/

  handle = InitVectorUpdate(temp_density, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /*-----------------------------------------------------------------------
   * compute the x-face velocities for each subgrid
   *-----------------------------------------------------------------------*/

  subgrids = GridSubgrids(x_grid);
  ForSubgridI(sg, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, sg);

    subvector_p = VectorSubvector(pressure_vector, sg);
    subvector_m = VectorSubvector(temp_mobility_x, sg);
    subvector_v = VectorSubvector(xvel, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid) + 2;
    nz = SubgridNZ(subgrid) + 2;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_p = SubvectorNX(subvector_p);
    ny_p = SubvectorNY(subvector_p);
    nz_p = SubvectorNZ(subvector_p);

    nx_m = SubvectorNX(subvector_m);
    ny_m = SubvectorNY(subvector_m);
    nz_m = SubvectorNZ(subvector_m);

    nx_v = SubvectorNX(subvector_v);
    ny_v = SubvectorNY(subvector_v);
    nz_v = SubvectorNZ(subvector_v);

    flopest = (FuncOps + 4) * nx_v * ny_v * nz_v;

    pl = SubvectorElt(subvector_p, ix - 1, iy, iz);
    pu = SubvectorElt(subvector_p, ix, iy, iz);

    ml = SubvectorElt(subvector_m, ix - 1, iy, iz);
    mu = SubvectorElt(subvector_m, ix, iy, iz);

    vel = SubvectorElt(subvector_v, ix, iy, iz);

    pi = 0; mi = 0; vi = 0;

    BoxLoopI3(i, j, k,
              ix, iy, iz, nx, ny, nz,
              pi, nx_p, ny_p, nz_p, 1, 1, 1,
              mi, nx_m, ny_m, nz_m, 1, 1, 1,
              vi, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vel[vi] = -Func(ml[mi], mu[mi]) * (pu[pi] - pl[pi]) / dx;
    });

    IncFLOPCount(flopest);
  }

  /*----------------------------------------------------------------------
   * compute the y-face velocities for each subgrid
   *----------------------------------------------------------------------*/

  subgrids = GridSubgrids(y_grid);
  ForSubgridI(sg, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, sg);

    subvector_p = VectorSubvector(pressure_vector, sg);
    subvector_m = VectorSubvector(temp_mobility_y, sg);
    subvector_v = VectorSubvector(yvel, sg);

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid) + 2;
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid) + 2;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_p = SubvectorNX(subvector_p);
    ny_p = SubvectorNY(subvector_p);
    nz_p = SubvectorNZ(subvector_p);

    nx_m = SubvectorNX(subvector_m);
    ny_m = SubvectorNY(subvector_m);
    nz_m = SubvectorNZ(subvector_m);

    nx_v = SubvectorNX(subvector_v);
    ny_v = SubvectorNY(subvector_v);
    nz_v = SubvectorNZ(subvector_v);

    flopest = (FuncOps + 4) * nx_v * ny_v * nz_v;

    pl = SubvectorElt(subvector_p, ix, iy - 1, iz);
    pu = SubvectorElt(subvector_p, ix, iy, iz);

    ml = SubvectorElt(subvector_m, ix, iy - 1, iz);
    mu = SubvectorElt(subvector_m, ix, iy, iz);

    vel = SubvectorElt(subvector_v, ix, iy, iz);

    mi = 0; pi = 0; vi = 0;

    BoxLoopI3(i, j, k,
              ix, iy, iz, nx, ny, nz,
              pi, nx_p, ny_p, nz_p, 1, 1, 1,
              mi, nx_m, ny_m, nz_m, 1, 1, 1,
              vi, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vel[vi] = -Func(ml[mi], mu[mi]) * (pu[pi] - pl[pi]) / dy;
    });

    IncFLOPCount(flopest);
  }

  /*----------------------------------------------------------------------
   * compute the z-face velocities for each subgrid
   *----------------------------------------------------------------------*/

  subgrids = GridSubgrids(z_grid);
  ForSubgridI(sg, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, sg);

    subvector_p = VectorSubvector(pressure_vector, sg);
    subvector_m = VectorSubvector(temp_mobility_z, sg);
    subvector_d = VectorSubvector(temp_density, sg);
    subvector_v = VectorSubvector(zvel, sg);

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid) + 2;
    ny = SubgridNY(subgrid) + 2;
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_p = SubvectorNX(subvector_p);
    ny_p = SubvectorNY(subvector_p);
    nz_p = SubvectorNZ(subvector_p);

    nx_m = SubvectorNX(subvector_m);
    ny_m = SubvectorNY(subvector_m);
    nz_m = SubvectorNZ(subvector_m);

    nx_d = SubvectorNX(subvector_d);
    ny_d = SubvectorNY(subvector_d);
    nz_d = SubvectorNZ(subvector_d);

    nx_v = SubvectorNX(subvector_v);
    ny_v = SubvectorNY(subvector_v);
    nz_v = SubvectorNZ(subvector_v);

    flopest = (FuncOps + 5) * nx_v * ny_v * nz_v;

    pl = SubvectorElt(subvector_p, ix, iy, iz - 1);
    pu = SubvectorElt(subvector_p, ix, iy, iz);

    ml = SubvectorElt(subvector_m, ix, iy, iz - 1);
    mu = SubvectorElt(subvector_m, ix, iy, iz);

    dl = SubvectorElt(subvector_d, ix, iy, iz - 1);
    du = SubvectorElt(subvector_d, ix, iy, iz);

    vel = SubvectorElt(subvector_v, ix, iy, iz);

    mi = 0; pi = 0; vi = 0; di = 0;

    BoxLoopI4(i, j, k,
              ix, iy, iz, nx, ny, nz,
              pi, nx_p, ny_p, nz_p,
              mi, nx_m, ny_m, nz_m,
              di, nx_d, ny_d, nz_d,
              vi, nx_v, ny_v, nz_v,
    {
      vel[vi] = -Func(ml[mi], mu[mi]) *
                ((pu[pi] - pl[pi]) / dz +
                 0.5 * (du[di] + dl[di]) * ProblemGravity(problem));
    });

    IncFLOPCount(flopest);
  }

  /*----------------------------------------------------------------------
   * Fixup the boundary values.
   *
   * CSW:  Previously, we applied a divergence free condition to the
   * velocities and computed the boundary face velocities accordingly.
   * With the change to a single grid (f_grid and a_grid becoming one),
   * this was no longer an easy fix-up.  We now apply a zero boundary
   * velocity condition.
   * Below is the description of what was previously done...
   * --------------------------------------------------------------------
   * Use the div(V) = 0 constraint to put velocity values on the
   * boundary that couldn't be done in the previous code.
   *
   * In the following, the indices 0, 1, and 2 are used to represent
   * directions x, y, and z, respectively.  We use the integer variables
   * `dir0', `dir1', `dir2' to represent the "primary" direction
   * and two "secondary" directions.  Here "primary" essentially the
   * direction of interest or the direction which we are modifying.
   *----------------------------------------------------------------------*/

  vel_vec[0] = xvel;
  vel_vec[1] = yvel;
  vel_vec[2] = zvel;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    for (ipatch = 0; ipatch < GrGeomSolidNumPatches(gr_domain); ipatch++)
    {
      GrGeomPatchLoop(i, j, k, fdir, gr_domain, ipatch,
                      r, ix, iy, iz, nx, ny, nz,
      {
        /* primary direction x */
        if (fdir[0])
        {
          dir0 = 0;
        }
        /* primary direction y */
        else if (fdir[1])
        {
          dir0 = 1;
        }
        /* primary direction z */
        else if (fdir[2])
        {
          dir0 = 2;
        }
        subvector_v0 = VectorSubvector(vel_vec[dir0], sg);

        vel0_l = SubvectorElt(subvector_v0, i, j, k);
        vel0_r = SubvectorElt(subvector_v0,
                              i + dir[dir0][0],
                              j + dir[dir0][1],
                              k + dir[dir0][2]);

        if (fdir[dir0] == -1)
        {
          vel_tmp = vel0_r;
          vel0_r = vel0_l;
          vel0_l = vel_tmp;
        }

        /*      Apply a xero velocity condition on outer boundaries */
        vel0_r[0] = 0.0;

        /*
         * vel0_r[0] = vel0_l[0]
         + alpha*h0*( (vel1_r[0] - vel1_l[0])/h1
         + (vel2_r[0] - vel2_l[0])/h2 );
         */
      });
    }
  }

  /*----------------------------------------------------------------------
   * exchange boundary data for x-velocity values
   *----------------------------------------------------------------------*/

  handle = InitVectorUpdate(xvel, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /*----------------------------------------------------------------------
   * exchange boundary data for y-velocity values
   *----------------------------------------------------------------------*/

  handle = InitVectorUpdate(yvel, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /*----------------------------------------------------------------------
   * exchange boundary data for z-velocity values
   *----------------------------------------------------------------------*/

  handle = InitVectorUpdate(zvel, VectorUpdateVelZ);
  FinalizeVectorUpdate(handle);

  /*----------------------------------------------------------------------
   * Free temp vectors
   *----------------------------------------------------------------------*/
  FreeVector(temp_mobility_x);
  FreeVector(temp_mobility_y);
  FreeVector(temp_mobility_z);
  FreeVector(temp_pressure);
  FreeVector(temp_density);

  /*----------------------------------------------------------------------
   * End timing
   *----------------------------------------------------------------------*/

  EndTiming(public_xtra->time_index);
}


/*-------------------------------------------------------------------------
 * PhaseVelocityFaceInitInstanceXtra
 *-------------------------------------------------------------------------*/

PFModule *PhaseVelocityFaceInitInstanceXtra(
                                            Problem *problem,
                                            Grid *   grid,
                                            Grid *   x_grid,
                                            Grid *   y_grid,
                                            Grid *   z_grid,
                                            double * temp_data)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*----------------------------------------------------------------------
   * Setup the InstanceXtra structure
   *----------------------------------------------------------------------*/
  if (problem != NULL)
  {
    (instance_xtra->problem) = problem;
  }
  /*** Set the pointer to the grids ***/
  if (grid != NULL)
  {
    /* set new data */
    (instance_xtra->grid) = grid;
  }

  if (x_grid != NULL)
  {
    (instance_xtra->x_grid) = x_grid;
  }

  if (y_grid != NULL)
  {
    (instance_xtra->y_grid) = y_grid;
  }

  if (z_grid != NULL)
  {
    (instance_xtra->z_grid) = z_grid;
  }

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->phase_mobility) =
      PFModuleNewInstance(ProblemPhaseMobility(problem), ());
    (instance_xtra->phase_density) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
    (instance_xtra->capillary_pressure) =
      PFModuleNewInstance(ProblemCapillaryPressure(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->phase_mobility), ());
    PFModuleReNewInstance((instance_xtra->phase_density), ());
    PFModuleReNewInstance((instance_xtra->capillary_pressure), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseVelocityFaceFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  PhaseVelocityFaceFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->phase_mobility);
    PFModuleFreeInstance(instance_xtra->phase_density);
    PFModuleFreeInstance(instance_xtra->capillary_pressure);

    free(instance_xtra);
  }
}

/*-------------------------------------------------------------------------
 * PhaseVelocityFaceNewPublicXtra
 *-------------------------------------------------------------------------*/

PFModule  *PhaseVelocityFaceNewPublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra;

  /*---------------------------------------------------------------------
   * Setup the PublicXtra structure
   *---------------------------------------------------------------------*/
  public_xtra = ctalloc(PublicXtra, 1);

  /*-------------------------------------------------------------*/
  /*                receive user input parameters                */

  /*-------------------------------------------------------------*/
  /*                     setup parameters                        */

  (public_xtra->time_index) = RegisterTiming("Phase Velocity Face");

  /*-------------------------------------------------------------*/

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseVelocityFaceFreePublicXtra
 *-------------------------------------------------------------------------*/

void PhaseVelocityFaceFreePublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    free(public_xtra);
  }
}


/*-------------------------------------------------------------------------
 * PhaseVelocityFaceSizeOfTempData
 *-------------------------------------------------------------------------*/

int  PhaseVelocityFaceSizeOfTempData()
{
  int sz = 0;

  return sz;
}
