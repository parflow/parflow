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
*  This module computes the contributions for the spatial discretization of the
*  kinematic equation for the overland flow boundary condition:KE,KW,KN,KS.
*
*  It also computes the derivatives of these terms for inclusion in the Jacobian.
*
* Could add a switch statement to handle the diffusion wave also.
* -DOK
*****************************************************************************/

#include "parflow.h"

#if !defined(PARFLOW_HAVE_CUDA) && !defined(PARFLOW_HAVE_KOKKOS)
#include "llnlmath.h"
//#include "llnltyps.h"
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef void InstanceXtra;

/*-------------------------------------------------------------------------
 * OverlandFlowEval
 *-------------------------------------------------------------------------*/

void    OverlandFlowEval(
                         Grid *       grid, /* data struct for computational grid */
                         int          sg, /* current subgrid */
                         BCStruct *   bc_struct, /* data struct of boundary patch values */
                         int          ipatch, /* current boundary patch */
                         ProblemData *problem_data, /* Geometry data for problem */
                         Vector *     pressure, /* Vector of phase pressures at each block */
                         Vector *     old_pressure, /* Vector of phase pressures at previous time */
                         double *     ke_v, /* return array corresponding to the east face KE  */
                         double *     kw_v, /* return array corresponding to the west face KW */
                         double *     kn_v, /* return array corresponding to the north face KN */
                         double *     ks_v, /* return array corresponding to the south face KS */
                         double *     qx_v, /* return array corresponding to the flux in x-dir */
                         double *     qy_v, /* return array corresponding to the flux in y-dir */
                         int          fcn) /* Flag determining what to calculate
                                            * fcn = CALCFCN => calculate the function value
                                            * fcn = CALCDER => calculate the function
                                            *                  derivative */
{
  Vector      *slope_x = ProblemDataTSlopeX(problem_data);
  Vector      *slope_y = ProblemDataTSlopeY(problem_data);
  Vector      *mannings = ProblemDataMannings(problem_data);
  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);

  Subvector     *sx_sub, *sy_sub, *mann_sub, *top_sub, *p_sub;

  double        *sx_dat, *sy_dat, *mann_dat, *top_dat, *pp;

  int i, j, k, ival=0, sy_v;
  PF_UNUSED(ival);

  p_sub = VectorSubvector(pressure, sg);

  sx_sub = VectorSubvector(slope_x, sg);
  sy_sub = VectorSubvector(slope_y, sg);
  mann_sub = VectorSubvector(mannings, sg);
  top_sub = VectorSubvector(top, sg);

  pp = SubvectorData(p_sub);

  sx_dat = SubvectorData(sx_sub);
  sy_dat = SubvectorData(sy_sub);
  mann_dat = SubvectorData(mann_sub);
  top_dat = SubvectorData(top_sub);

  sy_v = SubvectorNX(top_sub);

  if (fcn == CALCFCN)
  {
    if (qx_v == NULL || qy_v == NULL)  /* do not return velocity fluxes */
    {
      ForPatchCellsPerFace(BC_ALL,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                           Locals(int io, itop, ip, k1, ii, step;
                                  double q_v[3], xdir, ydir;),
                           CellSetup(DoNothing),
                           FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                           FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                           FACE(BackFace, DoNothing),
                           FACE(FrontFace,
                           {
                             io = SubvectorEltIndex(sx_sub, i, j, 0);
                             itop = SubvectorEltIndex(top_sub, i, j, 0);

                             /* compute east and west faces */
                             /* First initialize velocities, q_v, for inactive region */
                             q_v[0] = 0.0;
                             q_v[1] = 0.0;
                             q_v[2] = 0.0;
                             xdir = 0.0;
                             ydir = 0.0;
                             k1 = 0;

                             for (ii = -1; ii < 2; ii++)
                             {
                               k1 = (int)top_dat[itop + ii];
                               if (k1 >= 0)
                               {
                                 ip = SubvectorEltIndex(p_sub, (i + ii), j, k1);

                                 if (sx_dat[io + ii] > 0.0)
                                   xdir = -1.0;
                                 else if (sx_dat[io + ii] < 0.0)
                                   xdir = 1.0;
                                 else
                                   xdir = 0.0;

                                 q_v[ii + 1] = xdir * (RPowerR(fabs(sx_dat[io + ii]), 0.5) / mann_dat[io + ii])
                                               * RPowerR(pfmax((pp[ip]), 0.0), (5.0 / 3.0));
                               }
                             }

                             /* compute kw and ke - NOTE: io is for current cell */
                             kw_v[io] = pfmax(q_v[0], 0.0) - pfmax(-q_v[1], 0.0);
                             ke_v[io] = pfmax(q_v[1], 0.0) - pfmax(-q_v[2], 0.0);

                             /* compute north and south faces */
                             /* First initialize velocities, q_v, for inactive region */
                             q_v[0] = 0.0;
                             q_v[1] = 0.0;
                             q_v[2] = 0.0;

                             for (ii = -1; ii < 2; ii++)
                             {
                               step = ii * sy_v;
                               k1 = (int)top_dat[itop + step];
                               if (k1 >= 0)
                               {
                                 ip = SubvectorEltIndex(p_sub, i, (j + ii), k1);

                                 if (sy_dat[io + step] > 0.0)
                                   ydir = -1.0;
                                 else if (sy_dat[io + step] < 0.0)
                                   ydir = 1.0;
                                 else
                                   ydir = 0.0;

                                 q_v[ii + 1] = ydir * (RPowerR(fabs(sy_dat[io + step]), 0.5) / mann_dat[io + step])
                                               * RPowerR(pfmax((pp[ip]), 0.0), (5.0 / 3.0));
                               }
                             }

                             /* compute ks and kn - NOTE: io is for current cell */
                             ks_v[io] = pfmax(q_v[0], 0.0) - pfmax(-q_v[1], 0.0);
                             kn_v[io] = pfmax(q_v[1], 0.0) - pfmax(-q_v[2], 0.0);
                           }),
                           CellFinalize(DoNothing),
                           AfterAllCells(DoNothing)
        );
    }
    else   /* return velocity fluxes */
    {
      ForPatchCellsPerFace(BC_ALL,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                           Locals(int io, itop, ip, k1, step, ii;
                                  double q_v[3], xdir, ydir;),
                           CellSetup(DoNothing),
                           FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                           FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                           FACE(BackFace, DoNothing),
                           FACE(FrontFace,
                           {
                             io = SubvectorEltIndex(sx_sub, i, j, 0);
                             itop = SubvectorEltIndex(top_sub, i, j, 0);

                             /* compute east and west faces */
                             /* First initialize velocities, q_v, for inactive region */
                             q_v[0] = 0.0;
                             q_v[1] = 0.0;
                             q_v[2] = 0.0;

                             for (ii = -1; ii < 2; ii++)
                             {
                               k1 = (int)top_dat[itop + ii];
                               if (k1 >= 0)
                               {
                                 ip = SubvectorEltIndex(p_sub, (i + ii), j, k1);

                                 if (sx_dat[io + ii] > 0.0)
                                   xdir = -1.0;
                                 else if (sx_dat[io + ii] < 0.0)
                                   xdir = 1.0;
                                 else
                                   xdir = 0.0;

                                 q_v[ii + 1] = xdir * (RPowerR(fabs(sx_dat[io + ii]), 0.5) / mann_dat[io + ii]) * RPowerR(pfmax((pp[ip]), 0.0), (5.0 / 3.0));
                               }
                             }
                             qx_v[io] = q_v[1];
                             /* compute kw and ke - NOTE: io is for current cell */
                             kw_v[io] = pfmax(q_v[0], 0.0) - pfmax(-q_v[1], 0.0);
                             ke_v[io] = pfmax(q_v[1], 0.0) - pfmax(-q_v[2], 0.0);

                             /* compute north and south faces */
                             /* First initialize velocities, q_v, for inactive region */
                             q_v[0] = 0.0;
                             q_v[1] = 0.0;
                             q_v[2] = 0.0;

                             for (ii = -1; ii < 2; ii++)
                             {
                               step = ii * sy_v;
                               k1 = (int)top_dat[itop + step];
                               if (k1 >= 0)
                               {
                                 ip = SubvectorEltIndex(p_sub, i, (j + ii), k1);

                                 if (sy_dat[io + step] > 0.0)
                                   ydir = -1.0;
                                 else if (sy_dat[io + step] < 0.0)
                                   ydir = 1.0;
                                 else
                                   ydir = 0.0;

                                 q_v[ii + 1] = ydir * (RPowerR(fabs(sy_dat[io + step]), 0.5) / mann_dat[io + step]) * RPowerR(pfmax((pp[ip]), 0.0), (5.0 / 3.0));
                               }
                             }
                             qy_v[io] = q_v[1];
                             /* compute ks and kn - NOTE: io is for current cell */
                             ks_v[io] = pfmax(q_v[0], 0.0) - pfmax(-q_v[1], 0.0);
                             kn_v[io] = pfmax(q_v[1], 0.0) - pfmax(-q_v[2], 0.0);
                           }),
                           CellFinalize(DoNothing),
                           AfterAllCells(DoNothing)
        );
    }
  }
  else  /* fcn == CALCDER: derivs of KE,KW,KN,KS w.r.t. current cell (i,j,k) */
  {
    if (qx_v == NULL || qy_v == NULL)  /* Do not return derivs of velocity fluxes */
    {
      ForPatchCellsPerFace(BC_ALL,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                           Locals(int io, ip;
                                  double xdir, ydir, q_mid;),
                           CellSetup(DoNothing),
                           FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                           FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                           FACE(BackFace, DoNothing),
                           FACE(FrontFace,
                           {
                             /* compute derivs for east and west faces */

                             /* current cell */
                             io = SubvectorEltIndex(sx_sub, i, j, 0);
                             ip = SubvectorEltIndex(p_sub, i, j, k);

                             if (sx_dat[io] > 0.0)
                               xdir = -1.0;
                             else if (sx_dat[io] < 0.0)
                               xdir = 1.0;
                             else
                               xdir = 0.0;

                             q_mid = xdir * (5.0 / 3.0) * (RPowerR(fabs(sx_dat[io]), 0.5) / mann_dat[io]) * RPowerR(pfmax((pp[ip]), 0.0), (2.0 / 3.0));
                             /* compute derivs of kw and ke - NOTE: io is for current cell */
                             kw_v[io] = -pfmax(-q_mid, 0.0);
                             ke_v[io] = pfmax(q_mid, 0.0);


                             /* compute north and south faces */
                             if (sy_dat[io] > 0.0)
                               ydir = -1.0;
                             else if (sy_dat[io] < 0.0)
                               ydir = 1.0;
                             else
                               ydir = 0.0;

                             q_mid = ydir * (5.0 / 3.0) * (RPowerR(fabs(sy_dat[io]), 0.5) / mann_dat[io]) * RPowerR(pfmax((pp[ip]), 0.0), (2.0 / 3.0));
                             /* compute derivs of ks and kn - NOTE: io is for current cell */
                             ks_v[io] = -pfmax(-q_mid, 0.0);
                             kn_v[io] = pfmax(q_mid, 0.0);
                           }),
                           CellFinalize(DoNothing),
                           AfterAllCells(DoNothing)
        );
    }
    else   /* return derivs of velocity fluxes */
    {
      ForPatchCellsPerFace(BC_ALL,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                           Locals(int io, ip;
                                  double xdir, ydir, q_mid;),
                           CellSetup(DoNothing),
                           FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                           FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                           FACE(BackFace, DoNothing),
                           FACE(FrontFace,
                           {
                             /* compute derivs for east and west faces */

                             /* current cell */
                             io = SubvectorEltIndex(sx_sub, i, j, 0);
                             ip = SubvectorEltIndex(p_sub, i, j, k);

                             if (sx_dat[io] > 0.0)
                               xdir = -1.0;
                             else if (sx_dat[io] < 0.0)
                               xdir = 1.0;
                             else
                               xdir = 0.0;

                             q_mid = xdir * (5.0 / 3.0) * (RPowerR(fabs(sx_dat[io]), 0.5) / mann_dat[io]) * RPowerR(pfmax((pp[ip]), 0.0), (2.0 / 3.0));
                             qx_v[io] = q_mid;
                             /* compute derivs of kw and ke - NOTE: io is for current cell */
                             kw_v[io] = -pfmax(-q_mid, 0.0);
                             ke_v[io] = pfmax(q_mid, 0.0);


                             /* compute north and south faces */
                             if (sy_dat[io] > 0.0)
                               ydir = -1.0;
                             else if (sy_dat[io] < 0.0)
                               ydir = 1.0;
                             else
                               ydir = 0.0;

                             q_mid = ydir * (5.0 / 3.0) * (RPowerR(fabs(sy_dat[io]), 0.5) / mann_dat[io]) * RPowerR(pfmax((pp[ip]), 0.0), (2.0 / 3.0));
                             qy_v[io] = q_mid;
                             /* compute derivs of ks and kn - NOTE: io is for current cell */
                             ks_v[io] = -pfmax(-q_mid, 0.0);
                             kn_v[io] = pfmax(q_mid, 0.0);
                           }),
                           CellFinalize(DoNothing),
                           AfterAllCells(DoNothing)
        );
    }
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * OverlandFlowEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  OverlandFlowEvalFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * OverlandFlowEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void  OverlandFlowEvalFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  OverlandFlowEvalSizeOfTempData()
{
  return 0;
}
