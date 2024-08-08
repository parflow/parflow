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
* Operator induced prolongation for MGSemi module
*
*****************************************************************************/
#include "parflow.h"

/*--------------------------------------------------------------------------
 * MGSemiProlong
 *--------------------------------------------------------------------------*/

void             MGSemiProlong(
                               Matrix *        A_f,
                               Vector *        e_f,
                               Vector *        e_c,
                               Matrix *        P,
                               SubregionArray *f_sr_array,
                               SubregionArray *c_sr_array,
                               ComputePkg *    compute_pkg,
                               CommPkg *       e_f_comm_pkg)
{
  SubregionArray *subregion_array;

  Subregion      *subregion;

  Region         *compute_reg = NULL;

  Subvector      *e_f_sub;
  Subvector      *e_c_sub;

  Submatrix      *P_sub;

  double         *e_fp, *e_cp;

  double         *p1, *p2;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_f, ny_f, nz_f;
  int nx_c, ny_c, nz_c;

  int i_f, i_c;

  int sx, sy, sz;

  int compute_i, i, j, ii, jj, kk;
  int stride;

  CommHandle     *handle = NULL;

  (void)A_f;
  (void)f_sr_array;

  /*--------------------------------------------------------------------
   * Compute e_f in c_sr_array
   *--------------------------------------------------------------------*/

  ForSubregionI(i, c_sr_array)
  {
    subregion = SubregionArraySubregion(c_sr_array, i);

    ix = SubregionIX(subregion);
    iy = SubregionIY(subregion);
    iz = SubregionIZ(subregion);

    nx = SubregionNX(subregion);
    ny = SubregionNY(subregion);
    nz = SubregionNZ(subregion);

    if (nx && ny && nz)
    {
      sx = SubregionSX(subregion);
      sy = SubregionSY(subregion);
      sz = SubregionSZ(subregion);

      e_c_sub = VectorSubvector(e_c, i);
      e_f_sub = VectorSubvector(e_f, i);

      nx_c = SubvectorNX(e_c_sub);
      ny_c = SubvectorNY(e_c_sub);
      nz_c = SubvectorNZ(e_c_sub);

      nx_f = SubvectorNX(e_f_sub);
      ny_f = SubvectorNY(e_f_sub);
      nz_f = SubvectorNZ(e_f_sub);

      e_cp = SubvectorElt(e_c_sub, ix / sx, iy / sy, iz / sz);
      e_fp = SubvectorElt(e_f_sub, ix, iy, iz);

      i_c = 0;
      i_f = 0;
      BoxLoopI2(ii, jj, kk, ix, iy, iz, nx, ny, nz,
                i_c, nx_c, ny_c, nz_c, 1, 1, 1,
                i_f, nx_f, ny_f, nz_f, sx, sy, sz,
      {
        e_fp[i_f] = e_cp[i_c];
      });
    }
  }

  /*--------------------------------------------------------------------
   * Compute e_f in f_sr_array
   *--------------------------------------------------------------------*/

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:
        handle = InitCommunication(e_f_comm_pkg);
        compute_reg = ComputePkgIndRegion(compute_pkg);
        break;

      case 1:
        FinalizeCommunication(handle);
        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    ForSubregionArrayI(i, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, i);

      if (SubregionArraySize(subregion_array))
      {
        e_f_sub = VectorSubvector(e_f, i);

        P_sub = MatrixSubmatrix(P, i);

        nx_f = SubvectorNX(e_f_sub);
        ny_f = SubvectorNY(e_f_sub);
        nz_f = SubvectorNZ(e_f_sub);

        nx_c = SubmatrixNX(P_sub);
        ny_c = SubmatrixNY(P_sub);
        nz_c = SubmatrixNZ(P_sub);
      }

      ForSubregionI(j, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, j);

        ix = SubregionIX(subregion);
        iy = SubregionIY(subregion);
        iz = SubregionIZ(subregion);

        nx = SubregionNX(subregion);
        ny = SubregionNY(subregion);
        nz = SubregionNZ(subregion);

        sx = SubregionSX(subregion);
        sy = SubregionSY(subregion);
        sz = SubregionSZ(subregion);

        stride = (sx - 1) + ((sy - 1) + (sz - 1) * ny_f) * nx_f;

        e_fp = SubvectorElt(e_f_sub, ix, iy, iz);

        p1 = SubmatrixElt(P_sub, 0, ix, iy, iz);
        p2 = SubmatrixElt(P_sub, 1, ix, iy, iz);

        i_c = 0;
        i_f = 0;
        BoxLoopI2(ii, jj, kk, ix, iy, iz, nx, ny, nz,
                  i_c, nx_c, ny_c, nz_c, 1, 1, 1,
                  i_f, nx_f, ny_f, nz_f, sx, sy, sz,
        {
          e_fp[i_f] = (p1[i_c] * e_fp[i_f - stride] +
                       p2[i_c] * e_fp[i_f + stride]);
        });
      }
    }
  }

  /*-----------------------------------------------------------------------
   * Increment the flop counter
   *-----------------------------------------------------------------------*/

  IncFLOPCount(3 * VectorSize(e_c));
}


/*--------------------------------------------------------------------------
 * NewMGSemiProlongComputePkg
 *--------------------------------------------------------------------------*/

ComputePkg   *NewMGSemiProlongComputePkg(
                                         Grid *   grid,
                                         Stencil *stencil,
                                         int      sx,
                                         int      sy,
                                         int      sz,
                                         int      c_index,
                                         int      f_index)
{
  ComputePkg  *compute_pkg;

  Region      *send_reg;
  Region      *recv_reg;
  Region      *dep_reg;
  Region      *ind_reg;


  CommRegFromStencil(&send_reg, &recv_reg, grid, stencil);

  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        GridSubgrids(grid), send_reg, recv_reg, stencil);

  /* RDF temporary until rewrite of CommRegFromStencil */
  ProjectRegion(send_reg, sx, sy, sz, c_index, c_index, c_index);
  ProjectRegion(recv_reg, sx, sy, sz, c_index, c_index, c_index);
  ProjectRegion(dep_reg, sx, sy, sz, f_index, f_index, f_index);
  ProjectRegion(ind_reg, sx, sy, sz, f_index, f_index, f_index);

  compute_pkg = NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  return compute_pkg;
}
