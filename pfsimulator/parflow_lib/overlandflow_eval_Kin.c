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
*  kinematic wave approximation for the overland flow boundary condition:KE,KW,KN,KS.
*
*  It also computes the derivatives of these terms for inclusion in the Jacobian.
*
* @LEC, @RMM
*****************************************************************************/

#include "parflow.h"

#if !defined(PARFLOW_HAVE_CUDA) && !defined(PARFLOW_HAVE_KOKKOS)
#include "llnlmath.h"
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

void    OverlandFlowEvalKin(
                            Grid *       grid,  /* data struct for computational grid */
                            int          sg,  /* current subgrid */
                            BCStruct *   bc_struct,  /* data struct of boundary patch values */
                            int          ipatch,  /* current boundary patch */
                            ProblemData *problem_data,  /* Geometry data for problem */
                            Vector *     pressure,  /* Vector of phase pressures at each block */
                            double *     ke_v,  /* return array corresponding to the east face KE  */
                            double *     kw_v,  /* return array corresponding to the west face KW */
                            double *     kn_v,  /* return array corresponding to the north face KN */
                            double *     ks_v,  /* return array corresponding to the south face KS */
                            double *     ke_vns,  /* return array corresponding to the nonsymetric east face KE derivative  */
                            double *     kw_vns,  /* return array corresponding to the nonsymetricwest face KW derivative */
                            double *     kn_vns,  /* return array corresponding to the nonsymetricnorth face KN derivative */
                            double *     ks_vns,  /* return array corresponding to the nonsymetricsouth face KS derivative*/
                            double *     qx_v,  /* return array corresponding to the flux in x-dir */
                            double *     qy_v,  /* return array corresponding to the flux in y-dir */
                            int          fcn)  /* Flag determining what to calculate
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

  double ov_epsilon;

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

  //ov_epsilon= 1.0e-5;
  ov_epsilon = GetDoubleDefault("Solver.OverlandKinematic.Epsilon", 1.0e-5);


  if (fcn == CALCFCN)
  {
    ForPatchCellsPerFaceWithGhost(BC_ALL,
                                  BeforeAllCells(DoNothing),
                                  LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                                  Locals(int io, itop, ip, ipp1, ippsy;
                                         int k1, k0x, k0y, k1x, k1y;
                                         double Sf_x, Sf_y, Sf_mag;
                                         double Press_x, Press_y;),
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
                                    k1x = pfmax((int)top_dat[itop + 1],0);
                                    k1y = pfmax((int)top_dat[itop + sy_v],0);

                                    if (k1 >= 0)
                                    {
                                      ip = SubvectorEltIndex(p_sub, i, j, k1);
                                      Sf_x = sx_dat[io];
                                      Sf_y = sy_dat[io];
                                      ipp1 = (int)SubvectorEltIndex(p_sub, i+1, j, k1x);
                                      ippsy = (int)SubvectorEltIndex(p_sub, i, j+1, k1y);

                                      Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
                                      if (Sf_mag < ov_epsilon)
                                        Sf_mag = ov_epsilon;

                                      Press_x = RPMean(-Sf_x, 0.0,
                                                       pfmax((pp[ip]), 0.0),
                                                       pfmax((pp[ipp1]), 0.0));
                                      Press_y = RPMean(-Sf_y, 0.0,
                                                       pfmax((pp[ip]), 0.0),
                                                       pfmax((pp[ippsy]), 0.0));

                                      qx_v[io] = -(Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]))
                                                 * RPowerR(Press_x, (5.0 / 3.0));
                                      qy_v[io] = -(Sf_y / (RPowerR(fabs(Sf_mag), 0.5)
                                                           * mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));
                                    }

                                    //fix for lower x boundary
                                    if (k0x < 0.0)
                                    {
                                      if (k1 >= 0.0)
                                      {
                                        Sf_x = sx_dat[io];
                                        Sf_y = sy_dat[io];

                                        double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
                                        if (Sf_mag < ov_epsilon)
                                          Sf_mag = ov_epsilon;

                                        if (Sf_x > 0.0)
                                        {
                                          ip = SubvectorEltIndex(p_sub, i, j, k1);
                                          Press_x = pfmax((pp[ip]), 0.0);
                                          qx_v[io - 1] = -(Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
                                        }
                                      }
                                    }

                                    //fix for lower y boundary
                                    if (k0y < 0.0)
                                    {
                                      if (k1 >= 0.0)
                                      {
                                        Sf_x = sx_dat[io];
                                        Sf_y = sy_dat[io];

                                        double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
                                        if (Sf_mag < ov_epsilon)
                                          Sf_mag = ov_epsilon;

                                        if (Sf_y > 0.0)
                                        {
                                          ip = SubvectorEltIndex(p_sub, i, j, k1);
                                          Press_y = pfmax((pp[ip]), 0.0);
                                          qy_v[io - sy_v] = -(Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));
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
                         }),
                         CellFinalize(DoNothing),
                         AfterAllCells(DoNothing)
      );
  }
  else          //fcn = CALCDER calculates the derivs
  {
    ForPatchCellsPerFaceWithGhost(BC_ALL,
                                  BeforeAllCells(DoNothing),
                                  LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                                  Locals(int io, itop, ip, ipp1, ippsy;
                                         int k1, k0x, k0y, k1x, k1y;
                                         double Sf_x, Sf_y, Sf_mag;
                                         double Press_x, Press_y, qx_temp, qy_temp;),
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

                                      Sf_x = sx_dat[io];
                                      Sf_y = sy_dat[io];

                                      Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
                                      if (Sf_mag < ov_epsilon)
                                        Sf_mag = ov_epsilon;

                                      Press_x = RPMean(-Sf_x, 0.0,
                                                       pfmax((pp[ip]), 0.0),
                                                       pfmax((pp[ipp1]), 0.0));
                                      Press_y = RPMean(-Sf_y, 0.0,
                                                       pfmax((pp[ip]), 0.0),
                                                       pfmax((pp[ippsy]), 0.0));

                                      qx_temp = -(5.0 / 3.0) * (Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (2.0 / 3.0));
                                      qy_temp = -(5.0 / 3.0) * (Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (2.0 / 3.0));

                                      ke_v[io] = pfmax(qx_temp, 0);
                                      kw_v[io + 1] = -pfmax(-qx_temp, 0);
                                      kn_v[io] = pfmax(qy_temp, 0);
                                      ks_v[io + sy_v] = -pfmax(-qy_temp, 0);
                                    }

                                    //fix for lower x boundary
                                    if (k0x < 0.0)
                                    {
                                      if (k1 >= 0.0)
                                      {
                                        Sf_x = sx_dat[io];
                                        Sf_y = sy_dat[io];

                                        double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
                                        if (Sf_mag < ov_epsilon)
                                          Sf_mag = ov_epsilon;

                                        if (Sf_x > 0.0)
                                        {
                                          ip = SubvectorEltIndex(p_sub, i, j, k1);
                                          Press_x = pfmax((pp[ip]), 0.0);
                                          qx_temp = -(5.0 / 3.0) * (Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (2.0 / 3.0));

                                          kw_v[io] = qx_temp;
                                          ke_v[io - 1] = qx_temp;
                                        }
                                      }
                                    }

                                    //fix for lower y boundary
                                    if (k0y < 0.0)
                                    {
                                      if (k1 >= 0.0)
                                      {
                                        Sf_x = sx_dat[io];
                                        Sf_y = sy_dat[io];

                                        double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);  //+ov_epsilon;
                                        if (Sf_mag < ov_epsilon)
                                          Sf_mag = ov_epsilon;

                                        if (Sf_y > 0.0)
                                        {
                                          ip = SubvectorEltIndex(p_sub, i, j, k1);
                                          Press_y = pfmax((pp[ip]), 0.0);
                                          qy_temp = -(5.0 / 3.0) * (Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (2.0 / 3.0));

                                          ks_v[io] = qy_temp;
                                          kn_v[io - sy_v] = qy_temp;
                                        }
                                      }
                                    }
                                  }),
                                  CellFinalize(DoNothing),
                                  AfterAllCells(DoNothing)
      );
  }   // else calcder
}     // function


//*/
/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalKinInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  OverlandFlowEvalKinFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalKinNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * OverlandFlowEvalKinFreePublicXtra
 *-------------------------------------------------------------------------*/

void  OverlandFlowEvalKinFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinSizeOfTempData
 *--------------------------------------------------------------------------*/

int  OverlandFlowEvalKinSizeOfTempData()
{
  return 0;
}
