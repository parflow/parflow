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
  Problem           *problem;

  PFModule          *phase_mobility;
  PFModule          *phase_density;
  PFModule          *capillary_pressure;
  PFModule          *bc_pressure;
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
                                int          phase,
                                double       time)
{
  PFModule       *this_module = ThisPFModule;
  InstanceXtra   *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra     *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  PFModule       *phase_mobility = (instance_xtra->phase_mobility);
  PFModule       *phase_density = (instance_xtra->phase_density);
  PFModule       *capillary_pressure = (instance_xtra->capillary_pressure);
  PFModule       *bc_pressure = (instance_xtra->bc_pressure);

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

  int ix, iy, iz;
  int nx, ny, nz;
  double dx, dy, dz;

  int nx_p, ny_p, nz_p;
  int nx_m, ny_m, nz_m;
  int nx_v, ny_v, nz_v;
  int nx_d, ny_d, nz_d;

  int pi, mi, vi, di;

  int ipatch, ival, is, i, j, k;
  int flopest;

  double         *pl, *pu, *ml, *mu, *vel, *dl, *du;

  VectorUpdateCommHandle     *handle;

  Vector         *pressure_vector;
  double dummy_density;
  BCStruct *bc_struct;
  Subvector *p_sub, *subvector_m, *subvector_v, *d_sub;
  Subvector *vx_sub, *vy_sub, *vz_sub, *mx_sub, *my_sub, *mz_sub;
  double *vx, *vy, *vz;
  double *mx, *my, *mz;
  double *den, *pres, *bc_patch_values;

  int nx_vy, sy_v;
  int nx_vz, ny_vz, sz_v;


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
  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    p_sub = VectorSubvector(pressure_vector, is);
    subvector_m = VectorSubvector(temp_mobility_x, is);
    subvector_v = VectorSubvector(xvel, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid) + 2;
    nz = SubgridNZ(subgrid) + 2;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    nx_m = SubvectorNX(subvector_m);
    ny_m = SubvectorNY(subvector_m);
    nz_m = SubvectorNZ(subvector_m);

    nx_v = SubvectorNX(subvector_v);
    ny_v = SubvectorNY(subvector_v);
    nz_v = SubvectorNZ(subvector_v);

    flopest = (FuncOps + 4) * nx_v * ny_v * nz_v;

    pl = SubvectorElt(p_sub, ix - 1, iy, iz);
    pu = SubvectorElt(p_sub, ix, iy, iz);

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
  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    p_sub = VectorSubvector(pressure_vector, is);
    subvector_m = VectorSubvector(temp_mobility_y, is);
    subvector_v = VectorSubvector(yvel, is);

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid) + 2;
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid) + 2;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    nx_m = SubvectorNX(subvector_m);
    ny_m = SubvectorNY(subvector_m);
    nz_m = SubvectorNZ(subvector_m);

    nx_v = SubvectorNX(subvector_v);
    ny_v = SubvectorNY(subvector_v);
    nz_v = SubvectorNZ(subvector_v);

    flopest = (FuncOps + 4) * nx_v * ny_v * nz_v;

    pl = SubvectorElt(p_sub, ix, iy - 1, iz);
    pu = SubvectorElt(p_sub, ix, iy, iz);

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
  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    p_sub = VectorSubvector(pressure_vector, is);
    subvector_m = VectorSubvector(temp_mobility_z, is);
    d_sub = VectorSubvector(temp_density, is);
    subvector_v = VectorSubvector(zvel, is);

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid) + 2;
    ny = SubgridNY(subgrid) + 2;
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    nx_m = SubvectorNX(subvector_m);
    ny_m = SubvectorNY(subvector_m);
    nz_m = SubvectorNZ(subvector_m);

    nx_d = SubvectorNX(d_sub);
    ny_d = SubvectorNY(d_sub);
    nz_d = SubvectorNZ(d_sub);

    nx_v = SubvectorNX(subvector_v);
    ny_v = SubvectorNY(subvector_v);
    nz_v = SubvectorNZ(subvector_v);

    flopest = (FuncOps + 5) * nx_v * ny_v * nz_v;

    pl = SubvectorElt(p_sub, ix, iy, iz - 1);
    pu = SubvectorElt(p_sub, ix, iy, iz);

    ml = SubvectorElt(subvector_m, ix, iy, iz - 1);
    mu = SubvectorElt(subvector_m, ix, iy, iz);

    dl = SubvectorElt(d_sub, ix, iy, iz - 1);
    du = SubvectorElt(d_sub, ix, iy, iz);

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
   * --------------------------------------------------------------------
   * --------------------------------------------------------------------
   * Fixed global boundary values for saturated solver - JJB 04/21
   *
   *----------------------------------------------------------------------*/

  bc_struct = PFModuleInvokeType(BCPressureInvoke, bc_pressure,
                                 (problem_data, grid, gr_domain, time));
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    p_sub = VectorSubvector(pressure_vector, is);
    d_sub = VectorSubvector(temp_density, is);

    mx_sub = VectorSubvector(temp_mobility_x, is);
    my_sub = VectorSubvector(temp_mobility_y, is);
    mz_sub = VectorSubvector(temp_mobility_z, is);

    vx_sub = VectorSubvector(xvel, is);
    vy_sub = VectorSubvector(yvel, is);
    vz_sub = VectorSubvector(zvel, is);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);

    nx_vy = SubvectorNX(vy_sub);
    nx_vz = SubvectorNX(vz_sub);
    ny_vz = SubvectorNY(vz_sub);

    sy_v = nx_vy;
    sz_v = ny_vz * nx_vz;

    mx = SubvectorData(mx_sub);
    my = SubvectorData(my_sub);
    mz = SubvectorData(mz_sub);

    vx = SubvectorData(vx_sub);
    vy = SubvectorData(vy_sub);
    vz = SubvectorData(vz_sub);

    pres = SubvectorData(p_sub);
    den = SubvectorData(d_sub);

    ForBCStructNumPatches(ipatch, bc_struct)
    {
      bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);
      ForPatchCellsPerFace(DirichletBC,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                           Locals(int ip, vx_l, vy_l, vz_l;
                                  int alpha, vel_idx;
                                  double *mob_vec, *vel_vec;
                                  double pdiff, value, vel_h; ),
                           CellSetup({
        ip = SubvectorEltIndex(p_sub, i, j, k);

        vx_l = SubvectorEltIndex(vx_sub, i, j, k);
        vy_l = SubvectorEltIndex(vy_sub, i, j, k);
        vz_l = SubvectorEltIndex(vz_sub, i, j, k);

        alpha = 0;
        vel_idx = 0;

        mob_vec = NULL;
        vel_vec = NULL;

        vel_h = 0.0e0;
        pdiff = 0.0e0;
        value = bc_patch_values[ival];
      }),
                           FACE(LeftFace, {
        alpha = 0;
        mob_vec = mx;
        vel_vec = vx;
        vel_h = dx;
        vel_idx = vx_l;
        pdiff = value - pres[ip];
      }),
                           FACE(RightFace, {
        alpha = 0;
        mob_vec = mx;
        vel_vec = vx;
        vel_h = dx;
        vel_idx = vx_l + 1;
        pdiff = pres[ip] - value;
      }),
                           FACE(DownFace, {
        alpha = 0;
        mob_vec = my;
        vel_vec = vy;
        vel_h = dy;
        vel_idx = vy_l;
        pdiff = value - pres[ip];
      }),
                           FACE(UpFace, {
        alpha = 0;
        mob_vec = my;
        vel_vec = vy;
        vel_h = dy;
        vel_idx = vy_l + sy_v;
        pdiff = pres[ip] - value;
      }),
                           FACE(BackFace, {
        alpha = 1;
        mob_vec = mz;
        vel_vec = vz;
        vel_h = dz;
        vel_idx = vz_l;
        pdiff = value - pres[ip];
      }),
                           FACE(FrontFace, {
        alpha = -1;
        mob_vec = mz;
        vel_vec = vz;
        vel_h = dz;
        vel_idx = vz_l + sz_v;
        pdiff = pres[ip] - value;
      }),
                           CellFinalize({
        vel_vec[vel_idx] = mob_vec[ip] * (pdiff / (0.5 * vel_h) - alpha * den[ip] * ProblemGravity(problem));
      }),
                           AfterAllCells(DoNothing)
                           );

      ForPatchCellsPerFace(FluxBC,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                           Locals(int vel_idx, vx_l, vy_l, vz_l;
                                  double *vel_vec;
                                  double value; ),
                           CellSetup({
        vx_l = SubvectorEltIndex(vx_sub, i, j, k);
        vy_l = SubvectorEltIndex(vy_sub, i, j, k);
        vz_l = SubvectorEltIndex(vz_sub, i, j, k);

        vel_idx = 0;
        vel_vec = NULL;

        value = bc_patch_values[ival];
      }),
                           FACE(LeftFace, {
        vel_vec = vx;
        vel_idx = vx_l;
      }),
                           FACE(RightFace, {
        vel_vec = vx;
        vel_idx = vx_l + 1;
      }),
                           FACE(DownFace, {
        vel_vec = vy;
        vel_idx = vy_l;
      }),
                           FACE(UpFace, {
        vel_vec = vy;
        vel_idx = vy_l + sy_v;
      }),
                           FACE(BackFace, {
        vel_vec = vz;
        vel_idx = vz_l;
      }),
                           FACE(FrontFace, {
        vel_vec = vz;
        vel_idx = vz_l + sz_v;
      }),
                           CellFinalize({ vel_vec[vel_idx] = value; }),
                           AfterAllCells(DoNothing)
                           );
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
   * Free bc struct
   *----------------------------------------------------------------------*/
  FreeBCStruct(bc_struct);

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
    (instance_xtra->bc_pressure) =
      PFModuleNewInstanceType(BCPressurePackageInitInstanceXtraInvoke,
                              ProblemBCPressure(problem), (problem));
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->phase_mobility), ());
    PFModuleReNewInstance((instance_xtra->phase_density), ());
    PFModuleReNewInstance((instance_xtra->capillary_pressure), ());
    PFModuleReNewInstanceType(BCPressurePackageInitInstanceXtraInvoke,
                              (instance_xtra->bc_pressure), (problem));
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
    PFModuleFreeInstance(instance_xtra->bc_pressure);

    tfree(instance_xtra);
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
    tfree(public_xtra);
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
