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
* Transport helper routines for the react_trans reactive-transport coupling.
* These are TRANSPORT code, not chemistry: the solvers call them on the
* chemistry-off path as well, so this file compiles unconditionally and must
* not depend on Alquimia (see CLAUDE.md "Chem sources vs transport helpers").
* Relocated verbatim from chem_utilities.c and problem_bc_concen.c.
*
*****************************************************************************/

#include "parflow.h"
#include "transport_utilities.h"



double InterpolateTimeCycle(double total_cycle_length, double subcycle_dt)
{
  return subcycle_dt / total_cycle_length;
}

/*--------------------------------------------------------------------------
 * TransportSaturation
 * Calculates saturation delta for transient simulations
 * Places old saturation values into vector with 2 ghost layers
 *--------------------------------------------------------------------------*/

void TransportSaturation(Vector *sat_transport_start, Vector *delta_sat, Vector *old_sat, Vector *new_sat)
{
  Grid       *grid = VectorGrid(sat_transport_start);
  Subgrid    *subgrid;

  VectorUpdateCommHandle  *handle;

  Subvector  *os_sub;
  Subvector  *st_sub;

  double     *os, *st;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_os, ny_os, nz_os;
  int nx_st, ny_st, nz_st;
  int sg, i, j, k, os_i, st_i;

  PFVDiff(new_sat, old_sat, delta_sat);

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    os_sub = VectorSubvector(old_sat, sg);
    st_sub = VectorSubvector(sat_transport_start, sg);

    nx_os = SubvectorNX(os_sub);
    ny_os = SubvectorNY(os_sub);
    nz_os = SubvectorNZ(os_sub);

    nx_st = SubvectorNX(st_sub);
    ny_st = SubvectorNY(st_sub);
    nz_st = SubvectorNZ(st_sub);

    os = SubvectorElt(os_sub, ix, iy, iz);
    st = SubvectorElt(st_sub, ix, iy, iz);

    os_i = 0;
    st_i = 0;

    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              os_i, nx_os, ny_os, nz_os, 1, 1, 1,
              st_i, nx_st, ny_st, nz_st, 1, 1, 1,
    {
      st[st_i] = os[os_i];
    });
  }

  // scatter ghosts
  handle = InitVectorUpdate(sat_transport_start, VectorUpdateAll2);
  FinalizeVectorUpdate(handle);
}


void SelectReactTransTimeStep(double max_velocity, double CFL,
                              double PF_dt, double *advect_react_dt,
                              int *num_rt_iterations)
{
  double cfl_dt;

  cfl_dt = CFL / max_velocity;

  if (PF_dt <= cfl_dt)
  {
    *advect_react_dt = PF_dt;
    *num_rt_iterations = 1;
  }
  else
  {
    *num_rt_iterations = (int)ceil(PF_dt / cfl_dt);
    *advect_react_dt = PF_dt / *num_rt_iterations;
  }
}

/*--------------------------------------------------------------------------
 * BCConcenCopyPatch
 *
 *   Copies concentration values from interior cell on boundary
 *   into adjacent boundary cells
 *   3 layers deep
 *--------------------------------------------------------------------------*/
void BCConcenCopyPatch(Problem *problem, Grid *grid,
                       Vector **concentrations,
                       int ipatch)
{
  Subvector      *concen_sub;
  double         *concen_dat;

  int nx_v, ny_v, nz_v;
  int ci;
  int is, i, j, k;
  int iv, iv1, iv2, iv3;
  int num_concen;
  int ix, iy, iz, nx, ny, nz;
  int dir[6][3] = { { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };

  Subgrid *subgrid;
  SubgridArray *subgrids = GridSubgrids(grid);

  num_concen = ProblemNumContaminants(problem);

  /*-----------------------------------------------------------------------
   * Implement BC's
   *-----------------------------------------------------------------------*/
  for (int concen = 0; concen < num_concen; concen++)
  {
    ForSubgridI(is, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, is);

      concen_sub = VectorSubvector(concentrations[concen], is);
      concen_dat = SubvectorData(concen_sub);

      nx_v = SubvectorNX(concen_sub);
      ny_v = SubvectorNY(concen_sub);
      nz_v = SubvectorNZ(concen_sub);

      BCConcenPatchExtent(subgrid, &ix, &iy, &iz, &nx, &ny, &nz, ipatch);

      ci = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                ci, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        if (BoundaryCell(ipatch, i, j, k))
        {
          iv = SubvectorEltIndex(concen_sub, i, j, k);
          iv1 = SubvectorEltIndex(concen_sub, i + dir[ipatch][0], j + dir[ipatch][1], k + dir[ipatch][2]);
          iv2 = SubvectorEltIndex(concen_sub, i + 2 * dir[ipatch][0], j + 2 * dir[ipatch][1], k + 2 * dir[ipatch][2]);
          iv3 = SubvectorEltIndex(concen_sub, i + 3 * dir[ipatch][0], j + 3 * dir[ipatch][1], k + 3 * dir[ipatch][2]);

          concen_dat[iv1] = concen_dat[iv];
          concen_dat[iv2] = concen_dat[iv];
          concen_dat[iv3] = concen_dat[iv];
        }
      });
      /* ci is the BoxLoopI1 stride index; the body addresses cells via
       * SubvectorEltIndex instead, so reference it to keep clang's
       * -Wunused-but-set-variable (CI -Werror) quiet. */
      (void)ci;
    }
  }
}

/*-----------------------------------------------------------------------
 * BCConcenPatchExtent
 * Determine extent of subgrid patch
 * - one cell thick in the direction normal to the face
 * - three boundary cell thick in the other two directions
 * - like other ParFlow methods, relies on "left right front back bottom top" ordering of patches in TCL script
 *-----------------------------------------------------------------------*/
void BCConcenPatchExtent(Subgrid *subgrid, int *ix, int *iy, int *iz, int *nx, int *ny, int *nz, int ipatch)
{
  *ix = (ipatch > 1) ? SubgridIX(subgrid) - 3 : (ipatch == 0) ? SubgridIX(subgrid) : SubgridIX(subgrid) + SubgridNX(subgrid) - 1;
  *iy = (ipatch < 2 || ipatch > 3) ? SubgridIY(subgrid) - 3 : (ipatch == 2) ? SubgridIY(subgrid) : SubgridIY(subgrid) + SubgridNY(subgrid) - 1;
  *iz = (ipatch < 4) ? SubgridIZ(subgrid) - 3 : (ipatch == 4) ? SubgridIZ(subgrid) : SubgridIZ(subgrid) + SubgridNZ(subgrid) - 1;

  *nx = (ipatch < 2) ? 1 : SubgridNX(subgrid) + 6;
  *ny = (ipatch > 1 && ipatch < 4) ? 1 : SubgridNY(subgrid) + 6;
  *nz = (ipatch > 3) ? 1 : SubgridNZ(subgrid) + 6;
}

/*--------------------------------------------------------------------------
 * BCConcenCopyAdjacent
 * copy concen of adjacent interior cell into 3 ghost boundary layers
 *
 * Alternative access to BCConcen routines when not built with Alquimia
 *--------------------------------------------------------------------------*/
void BCConcenCopyAdjacent(Problem *problem, Grid *grid,
                          Vector **concentrations)
{
  int num_domain_patches;
  int domain_index;
  char *switch_name;

  switch_name = GetString("Domain.GeomName");
  domain_index = NA_NameToIndex(GlobalsGeomNames, switch_name);

  if (domain_index < 0)
    InputError("Error: invalid geometry name <%s> for key <%s>\n",
               switch_name, "Domain.GeomName");

  num_domain_patches = NA_Sizeof(GlobalsGeometries[domain_index]->patches);

  for (int ipatch = 0; ipatch < num_domain_patches; ipatch++)
  {
    BCConcenCopyPatch(problem, grid, concentrations,
                      ipatch);
  }
}

/*--------------------------------------------------------------------------
 * BoundaryCell
 * return 1 if subgrid touches current patch, 0 otherwise
 *--------------------------------------------------------------------------*/
int BoundaryCell(int ipatch, int i, int j, int k)
{
  if ((ipatch < 2 && (i == BackgroundX(GlobalsBackground) || i ==
                      BackgroundX(GlobalsBackground) + BackgroundNX(GlobalsBackground) - 1))
      ||
      ((ipatch == 2 || ipatch == 3) && (j == BackgroundY(GlobalsBackground) || j ==
                                        BackgroundY(GlobalsBackground) + BackgroundNY(GlobalsBackground) - 1))
      ||
      ((ipatch == 4 || ipatch == 5) && (k == BackgroundZ(GlobalsBackground) || k ==
                                        BackgroundZ(GlobalsBackground) + BackgroundNZ(GlobalsBackground) - 1)))
  {
    return 1;
  }
  else
  {
    return 0;
  }
}
