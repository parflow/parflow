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
*  This module computes the contributions for the spatial discretization of the
*  diffusive wave approximation for the overland flow boundary condition:KE,KW,KN,KS.
*
*  It also computes the derivatives of these terms for inclusion in the Jacobian.
* @LEC, @RMM
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

/*---------------------------------------------------------------------
 * Define macros for function evaluation
 *---------------------------------------------------------------------*/
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)

/*-------------------------------------------------------------------------
 * OverlandFlowEval
 *-------------------------------------------------------------------------*/

void    OverlandFlowEvalDiff(
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
                             double *     ke_vns, /* return array corresponding to the nonsymetric east face KE derivative  */
                             double *     kw_vns, /* return array corresponding to the nonsymetricwest face KW derivative */
                             double *     kn_vns, /* return array corresponding to the nonsymetricnorth face KN derivative */
                             double *     ks_vns, /* return array corresponding to the nonsymetricsouth face KS derivative*/
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

  // printf("overland_eval_diffusive called\n");
  Subvector     *sx_sub, *sy_sub, *mann_sub, *top_sub, *p_sub, *op_sub;

  Subgrid      *subgrid;

  double        *sx_dat, *sy_dat, *mann_dat, *top_dat, *pp, *opp;

  double dx, dy, ov_epsilon;

  int i, j, k, ival, sy_v;
  PF_UNUSED(ival);

  p_sub = VectorSubvector(pressure, sg);
  op_sub = VectorSubvector(old_pressure, sg);
  sx_sub = VectorSubvector(slope_x, sg);
  sy_sub = VectorSubvector(slope_y, sg);
  mann_sub = VectorSubvector(mannings, sg);
  top_sub = VectorSubvector(top, sg);

  pp = SubvectorData(p_sub);
  opp = SubvectorData(op_sub);

  sx_dat = SubvectorData(sx_sub);
  sy_dat = SubvectorData(sy_sub);
  mann_dat = SubvectorData(mann_sub);
  top_dat = SubvectorData(top_sub);

  subgrid = GridSubgrid(grid, sg);
  dx = SubgridDX(subgrid);
  dy = SubgridDY(subgrid);

  sy_v = SubvectorNX(top_sub);

  ov_epsilon = GetDoubleDefault("Solver.OverlandDiffusive.Epsilon", 1.0e-5);

  if (fcn == CALCFCN)
  {
    ForPatchCellsPerFaceWithGhost(BC_ALL,
                                  BeforeAllCells(DoNothing),
                                  LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                                  Locals(int io, itop, ip=0, ipp1, ippsy;
                                         int k1, k0x, k0y, k1x, k1y;
                                         double Press_x=NAN, Press_y;
                                         double Sf_x=NAN, Sf_y, Sf_xo=NAN, Sf_yo=NAN, Sf_mag;
                                         double Pupx=NAN, Pupy, Pupox, Pupoy, Pdown, Pdowno;),
                                  CellSetup(DoNothing),
                                  FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                                  FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                                  FACE(BackFace, DoNothing),
                                  FACE(FrontFace,
                                  {
                                    io = SubvectorEltIndex(sx_sub, i, j, 0);
                                    itop = SubvectorEltIndex(top_sub, i, j, 0);

                                    k1 = (int)top_dat[itop];
                                    k0x = (int)top_dat[itop - 1];
                                    k0y = (int)top_dat[itop - sy_v];
                                    k1x = (int)top_dat[itop + 1];
                                    k1y = (int)top_dat[itop + sy_v];

                                    if (k1 >= 0)
                                    {
                                      ip = SubvectorEltIndex(p_sub, i, j, k1);
                                      ipp1 = (int)SubvectorEltIndex(p_sub, i+1, j, k1x);
                                      ippsy = (int)SubvectorEltIndex(p_sub, i, j+1, k1y);
                                      Pupx = pfmax(pp[ipp1], 0.0);
                                      Pupy = pfmax(pp[ippsy], 0.0);
                                      Pupox = pfmax(opp[ipp1], 0.0);
                                      Pupoy = pfmax(opp[ippsy], 0.0);
                                      Pdown = pfmax(pp[ip], 0.0);
                                      Pdowno = pfmax(opp[ip], 0.0);

                                      Sf_x = sx_dat[io] + (Pupx - Pdown) / dx;
                                      Sf_y = sy_dat[io] + (Pupy - Pdown) / dy;

                                      Sf_xo = sx_dat[io] + (Pupox - Pdowno) / dx;
                                      Sf_yo = sy_dat[io] + (Pupoy - Pdowno) / dy;

                                      Sf_mag = RPowerR(Sf_xo * Sf_xo + Sf_yo * Sf_yo, 0.5);
                                      if (Sf_mag < ov_epsilon)
                                        Sf_mag = ov_epsilon;

                                      Press_x = RPMean(-Sf_x, 0.0, pfmax((pp[ip]), 0.0), pfmax((pp[ipp1]), 0.0));
                                      Press_y = RPMean(-Sf_y, 0.0, pfmax((pp[ip]), 0.0), pfmax((pp[ippsy]), 0.0));

                                      qx_v[io] = -(Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
                                      qy_v[io] = -(Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));
                                    }

                                    //fix for lower x boundary
                                    if (k0x < 0.0)
                                    {
                                      Press_x = pfmax((pp[ip]), 0.0);
                                      Sf_x = sx_dat[io] + (Press_x - 0.0) / dx;

                                      Pupox = pfmax(opp[ip], 0.0);
                                      Sf_xo = sx_dat[io] + (Pupox - 0.0) / dx;

                                      double Sf_mag = RPowerR(Sf_xo * Sf_xo + Sf_yo * Sf_yo, 0.5); //+ov_epsilon;
                                      if (Sf_mag < ov_epsilon)
                                        Sf_mag = ov_epsilon;
                                      if (Sf_x > 0.0)
                                      {
                                        qx_v[io - 1] = -(Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
                                      }
                                    }

                                    //fix for lower y boundary
                                    if (k0y < 0.0)
                                    {
                                      Press_y = pfmax((pp[ip]), 0.0);
                                      Sf_y = sy_dat[io] + (Press_y - 0.0) / dx;

                                      Pupoy = pfmax(opp[ip], 0.0);
                                      Sf_yo = sy_dat[io] + (Pupoy - 0.0) / dx;

                                      double Sf_mag = RPowerR(Sf_xo * Sf_xo + Sf_yo * Sf_yo, 0.5); //Note that the sf_xo was already corrected above
                                      if (Sf_mag < ov_epsilon)
                                        Sf_mag = ov_epsilon;

                                      if (Sf_y > 0.0)
                                      {
                                        qy_v[io - sy_v] = -(Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));
                                      }

                                      // Recalculating the x flow in the case with both the lower and left boundaries
                                      // This is exactly the same as the q_x in the left boundary conditional above but
                                      // recalculating qx_v here again becuase the sf_mag will be adjusted with the new sf_yo above
                                      if (k0x < 0.0)
                                      {
                                        if (Sf_x > 0.0)
                                        {
                                          qx_v[io - 1] = -(Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
                                        }
                                      }
                                    }
                                  }),
                                  CellFinalize(DoNothing),
                                  AfterAllCells(DoNothing)
      );

    ForPatchCellsPerFace(BC_ALL,
                         BeforeAllCells(DoNothing),
                         LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                         Locals(int io;),
                         CellSetup(DoNothing),
                         FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                         FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                         FACE(BackFace, DoNothing),
                         FACE(FrontFace,
                         {
                           io = SubvectorEltIndex(sx_sub, i, j, 0);
                           ke_v[io] = qx_v[io];
                           kw_v[io] = qx_v[io - 1];
                           kn_v[io] = qy_v[io];
                           ks_v[io] = qy_v[io - sy_v];
                           //printf("i=%d j=%d k=%d ke_v=%d kw_v=%d kn_v=%d ks_v=%f\n",i,j,k,ke_v[io],kw_v[io],kn_v[io],ks_v[io]);
                         }),
                         CellFinalize(DoNothing),
                         AfterAllCells(DoNothing)
      );
  }
  else          //fcn = CALCDER calculates the derivs of KE KW KN KS wrt to current cell (i,j,k)
  {
    ForPatchCellsPerFace(BC_ALL,
                         BeforeAllCells(DoNothing),
                         LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                         Locals(int io, itop, ip=0, ipp1, ippsy;
                                int k1, k0x, k0y, k1x, k1y;
                                double Pupx=NAN, Pupy, Pupox, Pupoy, Pdown, Pdowno;
                                double Sf_x=NAN, Sf_y, Sf_xo=NAN, Sf_yo=NAN, Sf_mag;),
                         CellSetup(DoNothing),
                         FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                         FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                         FACE(BackFace, DoNothing),
                         FACE(FrontFace,
                         {
                           io = SubvectorEltIndex(sx_sub, i, j, 0);
                           itop = SubvectorEltIndex(top_sub, i, j, 0);

                           k1 = (int)top_dat[itop];
                           k0x = (int)top_dat[itop - 1];
                           k0y = (int)top_dat[itop - sy_v];
                           k1x = (int)top_dat[itop + 1];
                           k1y = (int)top_dat[itop + sy_v];

                           if (k1 >= 0)
                           {
                             ip = SubvectorEltIndex(p_sub, i, j, k1);
                             ipp1 = (int)SubvectorEltIndex(p_sub, i+1, j, k1x);
                             ippsy = (int)SubvectorEltIndex(p_sub, i, j+1, k1y);
                             Pupx = pfmax(pp[ipp1], 0.0);
                             Pupy = pfmax(pp[ippsy], 0.0);
                             Pupox = pfmax(opp[ipp1], 0.0);
                             Pupoy = pfmax(opp[ippsy], 0.0);
                             Pdown = pfmax(pp[ip], 0.0);
                             Pdowno = pfmax(opp[ip], 0.0);

                             Sf_x = sx_dat[io] + (Pupx - Pdown) / dx;
                             Sf_y = sy_dat[io] + (Pupy - Pdown) / dy;

                             Sf_xo = sx_dat[io] + (Pupox - Pdowno) / dx;
                             Sf_yo = sy_dat[io] + (Pupoy - Pdowno) / dy;

                             Sf_mag = RPowerR(Sf_xo * Sf_xo + Sf_yo * Sf_yo, 0.5); //+ov_epsilon;
                             if (Sf_mag < ov_epsilon)
                               Sf_mag = ov_epsilon;

                             if (Sf_x < 0)
                             {
                               ke_v[io] = (5.0 / 3.0) * (-sx_dat[io] - (Pupx / dx)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pdown, (2.0 / 3.0)) +
                                          (8.0 / 3.0) * RPowerR(Pdown, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);

                               kw_v[io + 1] = -RPowerR(Pdown, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);

                               ke_vns[io] = kw_v[io + 1];
                               kw_vns[io + 1] = ke_v[io];
                             }

                             if (Sf_x >= 0)
                             {
                               ke_v[io] = RPowerR(Pupx, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);

                               kw_v[io + 1] = (5.0 / 3.0) * (-sx_dat[io] + (Pdown / dx)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pupx, (2.0 / 3.0)) -
                                              (8.0 / 3.0) * RPowerR(Pupx, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);

                               ke_vns[io] = kw_v[io + 1];
                               kw_vns[io + 1] = ke_v[io];
                             }

                             if (Sf_y < 0)
                             {
                               kn_v[io] = (5.0 / 3.0) * (-sy_dat[io] - (Pupy / dy)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pdown, (2.0 / 3.0)) +
                                          (8.0 / 3.0) * RPowerR(Pdown, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dy);

                               ks_v[io + sy_v] = -RPowerR(Pdown, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dy);

                               kn_vns[io] = ks_v[io + sy_v];
                               ks_vns[io + sy_v] = kn_v[io];
                             }

                             if (Sf_y >= 0)
                             {
                               kn_v[io] = RPowerR(Pupy, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dy);

                               ks_v[io + sy_v] = (5.0 / 3.0) * (-sy_dat[io] + (Pdown / dy)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pupy, (2.0 / 3.0)) -
                                                 (8.0 / 3.0) * RPowerR(Pupy, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dy);

                               kn_vns[io] = ks_v[io + sy_v];
                               ks_vns[io + sy_v] = kn_v[io];
                             }
                           }

                           //fix for lower x boundary
                           if (k0x < 0.0)
                           {
                             Pupx = pfmax((pp[ip]), 0.0);
                             Sf_x = sx_dat[io] + (Pupx - 0.0) / dx;

                             Pupox = pfmax(opp[ip], 0.0);
                             Sf_xo = sx_dat[io] + (Pupox - 0.0) / dx;

                             double Sf_mag = RPowerR(Sf_xo * Sf_xo + Sf_yo * Sf_yo, 0.5);
                             if (Sf_mag < ov_epsilon)
                               Sf_mag = ov_epsilon;

                             if (Sf_x < 0)
                             {
                               ke_v[io - 1] = 0.0;
                               kw_v[io] = 0.0;
                               kw_vns[io] = 0.0;
                               ke_vns[io - 1] = 0.0;
                             }

                             if (Sf_x >= 0)
                             {
                               ke_v[io - 1] = RPowerR(Pupx, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);
                               kw_v[io] = (5.0 / 3.0) * (-sx_dat[io] + 0.0) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pupx, (2.0 / 3.0)) -
                                          (8.0 / 3.0) * RPowerR(Pupx, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);
                               ke_vns[io - 1] = kw_v[io];
                               kw_vns[io] = ke_v[io - 1];
                             }
                           }

                           //fix for lower y boundary
                           if (k0y < 0.0)
                           {
                             Pupy = pfmax((pp[ip]), 0.0);
                             Sf_y = sy_dat[io] + (Pupy - 0.0) / dy;

                             Pupoy = pfmax(opp[ip], 0.0);
                             Sf_yo = sy_dat[io] + (Pupoy - 0.0) / dy;

                             double Sf_mag = RPowerR(Sf_xo * Sf_xo + Sf_yo * Sf_yo, 0.5); //Note that the sf_xo was already corrected above
                             if (Sf_mag < ov_epsilon)
                               Sf_mag = ov_epsilon;

                             if (Sf_y < 0)
                             {
                               kn_v[io - sy_v] = 0.0;
                               ks_v[io] = 0.0;
                               ks_vns[io] = 0.0;
                               kn_vns[io - sy_v] = 0.0;
                             }

                             if (Sf_y >= 0)
                             {
                               kn_vns[io - sy_v] = RPowerR(Pupy, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dy);
                               ks_v[io] = (5.0 / 3.0) * (-sy_dat[io] + 0.0) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pupy, (2.0 / 3.0)) -
                                          (8.0 / 3.0) * RPowerR(Pupy, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dy);
                               kn_vns[io - sy_v] = ks_v[io];
                               ks_vns[io] = kn_v[io - sy_v];
                             }

                             // Recalculating the x flow in the case with both the lower and left boundaries
                             // This is exactly the same as the q_x in the left boundary conditional above but
                             // recalculating qx_v here again becuase the sf_mag will be adjusted with the new sf_yo above
                             if (k0x < 0.0)
                             {
                               if (Sf_x < 0)
                               {
                                 kn_v[io - sy_v] = 0.0;
                                 ks_v[io] = 0.0;
                                 ks_vns[io] = 0.0;
                                 kn_vns[io - sy_v] = 0.0;
                               }

                               if (Sf_x >= 0)
                               {
                                 ke_v[io - 1] = RPowerR(Pupx, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);
                                 kw_v[io] = (5.0 / 3.0) * (-sx_dat[io] + 0.0) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]) * RPowerR(Pupx, (2.0 / 3.0)) -
                                            (8.0 / 3.0) * RPowerR(Pupx, (5.0 / 3.0)) / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io] * dx);
                                 ke_vns[io - 1] = kw_v[io];
                                 kw_vns[io] = ke_v[io - 1];
                               }
                             }
                           }
                         }),
                         CellFinalize(DoNothing),
                         AfterAllCells(DoNothing)
      );
  }
}

//*/
/*--------------------------------------------------------------------------
 * OverlandFlowEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalDiffInitInstanceXtra()
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

void  OverlandFlowEvalDiffFreeInstanceXtra()
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

PFModule  *OverlandFlowEvalDiffNewPublicXtra()
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

void  OverlandFlowEvalDiffFreePublicXtra()
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

int  OverlandFlowEvalDiffSizeOfTempData()
{
  return 0;
}
