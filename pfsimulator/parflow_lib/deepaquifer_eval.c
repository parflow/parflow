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

#include "parflow.h"

#define DelHarmonicMean(a, b, da, db, hm)     \
        (!((a) + (b)) ? 0 :                   \
         (2 * ((da) * (b) + (a) * (db)) -     \
          (hm) * ((da) + (db))) / ((a) + (b)) \
        )


typedef void PublicXtra;

typedef struct {
  Vector *SpecificYield;
  Vector *AquiferDepth;
} InstanceXtra;


void InitDeepAquiferParameter(Vector *par_v, ParameterUnion par)
{
  switch (ParameterUnionID(par))
  {
    case 0: // is double value
    {
      InitVectorAll(par_v, ParameterUnionDataDouble(par));
      break;
    }

    case 1: // is filename string
    {
      ReadPFBinary(ParameterUnionDataString(par), par_v);
      break;
    }

    default:
    {
      InitVectorAll(par_v, 0.0);
      break;
    }
  }

  VectorUpdateCommHandle *handle = InitVectorUpdate(par_v, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  return;
}


/*--------------------------------------------------------------------------
 * DeepAquiferEval
 *--------------------------------------------------------------------------*/
/**
 * @name DeepAquifer BC Module
 * @brief Evaluation of the DeepAquifer boundary
 *
 * This module can evaluate both the non-linear function
 * and its jacobian. When invoked, the `fcn` flag tracks
 * which output to compute. The evaluation is added to
 * `groundwater_out`.
 *
 * @param groundwater_out groundwaterflow evaluation
 * @param fcn flag = {CALCFCN , CALCDER}
 * @param bc_struct boundary condition structure
 * @param subgrid current subgrid
 * @param p_sub new pressure subvector
 * @param old_pressure old pressure data
 * @param dt timestep
 * @param Kr relative permeability
 * @param Ks_x permeability_x subvector data
 * @param Ks_y permeability_y subvector data
 * @param ipatch current patch
 * @param isubgrid current subgrid
 * @param problem_data geometry data for problem
 */
void DeepAquiferEval(void *       groundwater_out,
                     int          fcn,
                     BCStruct *   bc_struct,
                     Subgrid *    subgrid,
                     Subvector *  p_sub,
                     double *     old_pressure,
                     double       dt,
                     double *     Kr,
                     double *     Ks_x,
                     double *     Ks_y,
                     int          ipatch,
                     int          isubgrid,
                     ProblemData *problem_data)
{
  if (fcn == CALCFCN)
  {
    double *fp = (double*)groundwater_out;

    DeepAquiferEvalNLFunc(fp, bc_struct, subgrid,
                          p_sub, old_pressure, dt, Kr, Ks_x, Ks_y,
                          ipatch, isubgrid, problem_data);
  }
  else     /* fcn == CALCDER */

  {
    Submatrix *J_sub = (Submatrix*)groundwater_out;

    DeepAquiferEvalJacob(J_sub, bc_struct, subgrid,
                         p_sub, old_pressure, dt, Kr, Ks_x, Ks_y,
                         ipatch, isubgrid, problem_data);
  }

  return;
}

/**
 * @name DeepAquifer Eval NLFunc
 * @brief Evaluation of the DeepAquifer boundary
 *
 * This module can evaluates the non-linear function.
 * The function evaluation is added to `fp` arg.
 *
 *
 * @param fp groundwaterflow function evaluation
 * @param bc_struct boundary condition structure
 * @param subgrid current subgrid
 * @param p_sub new pressure subvector
 * @param old_pressure old pressure data
 * @param dt timestep
 * @param Kr relative permeability
 * @param Ks_x permeability_x subvector data
 * @param Ks_y permeability_y subvector data
 * @param ipatch current patch
 * @param isubgrid current subgrid
 * @param problem_data geometry data for problem
 */
void DeepAquiferEvalNLFunc(double *     fp,
                           BCStruct *   bc_struct,
                           Subgrid *    subgrid,
                           Subvector *  p_sub,
                           double *     old_pressure,
                           double       dt,
                           double *     Kr,
                           double *     Ks_x,
                           double *     Ks_y,
                           int          ipatch,
                           int          isubgrid,
                           ProblemData *problem_data)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  double *new_pressure = SubvectorData(p_sub);

  Vector    *bottom = ProblemDataIndexOfDomainBottom(problem_data);
  Subvector *bottom_sub = VectorSubvector(bottom, isubgrid);
  double    *bottom_dat = SubvectorData(bottom_sub);

  Vector    *Sy = instance_xtra->SpecificYield;
  Vector    *Ad = instance_xtra->AquiferDepth;
  Subvector *Sy_sub = VectorSubvector(Sy, isubgrid);
  double    *Sy_dat = SubvectorData(Sy_sub);
  Subvector *Ad_sub = VectorSubvector(Ad, isubgrid);
  double    *Ad_dat = SubvectorData(Ad_sub);

  int i = 0, j = 0, k = 0, ival = 0;

  ForPatchCellsPerFace(DeepAquiferBC,
                       BeforeAllCells(DoNothing),
                       LoopVars(i, j, k, ival, bc_struct, ipatch, isubgrid),
                       Locals(
                              double Sy = 0.0;
                              double Ad_mid = 0.0;
                              double Ad_lft = 0.0;
                              double Ad_rgt = 0.0;
                              double Ad_lwr = 0.0;
                              double Ad_upr = 0.0;

                              double dx = SubgridDX(subgrid);
                              double dy = SubgridDY(subgrid);
                              double dxdy = dx * dy;
                              double dtdx_over_dy = dt * dx / dy;
                              double dtdy_over_dx = dt * dy / dx;

                              int ibot_mid = 0;
                              int ibot_lft = 0;
                              int ibot_rgt = 0;
                              int ibot_lwr = 0;
                              int ibot_upr = 0;

                              int k_lft = 0;
                              int k_rgt = 0;
                              int k_lwr = 0;
                              int k_upr = 0;

                              int ip_mid = 0;
                              int ip_lft = 0;
                              int ip_rgt = 0;
                              int ip_lwr = 0;
                              int ip_upr = 0;

                              int is_lft_edge = 0;
                              int is_rgt_edge = 0;
                              int is_lwr_edge = 0;
                              int is_upr_edge = 0;

                              double q_storage = 0.0, q_divergence = 0.0;

                              double Tx_mid = 0.0, Ty_mid = 0.0;
                              double Tx_lft = 0.0, Tx_rgt = 0.0;
                              double Ty_lwr = 0.0, Ty_upr = 0.0;

                              double old_head_mid = 0.0, new_head_mid = 0.0;
                              double new_head_lft = 0.0, new_head_rgt = 0.0;
                              double new_head_lwr = 0.0, new_head_upr = 0.0;

                              double dh_lft = 0.0, dh_rgt = 0.0;
                              double dh_lwr = 0.0, dh_upr = 0.0;
                              double dh_dt = 0.0;
                              ),
                       CellSetup({
    ip_mid = SubvectorEltIndex(p_sub, i, j, k);

    q_storage = 0.0;
    q_divergence = 0.0;

    PF_UNUSED(ival);
  }),
                       FACE(LeftFace, DoNothing),
                       FACE(RightFace, DoNothing),
                       FACE(DownFace, DoNothing),
                       FACE(UpFace, DoNothing),
                       FACE(BackFace,
  {
    ibot_mid = SubvectorEltIndex(bottom_sub, i, j, 0);
    ibot_lft = SubvectorEltIndex(bottom_sub, i - 1, j, 0);
    ibot_rgt = SubvectorEltIndex(bottom_sub, i + 1, j, 0);
    ibot_lwr = SubvectorEltIndex(bottom_sub, i, j - 1, 0);
    ibot_upr = SubvectorEltIndex(bottom_sub, i, j + 1, 0);

    k_lft = rint(bottom_dat[ibot_lft]);
    k_rgt = rint(bottom_dat[ibot_rgt]);
    k_lwr = rint(bottom_dat[ibot_lwr]);
    k_upr = rint(bottom_dat[ibot_upr]);

    // find if we are at an edge cell:
    is_lft_edge = (k_lft < 0);
    is_rgt_edge = (k_rgt < 0);
    is_lwr_edge = (k_lwr < 0);
    is_upr_edge = (k_upr < 0);

    ip_lft = is_lft_edge ? ip_mid : SubvectorEltIndex(p_sub, i - 1, j, k_lft);
    ip_rgt = is_rgt_edge ? ip_mid : SubvectorEltIndex(p_sub, i + 1, j, k_rgt);
    ip_lwr = is_lwr_edge ? ip_mid : SubvectorEltIndex(p_sub, i, j - 1, k_lwr);
    ip_upr = is_upr_edge ? ip_mid : SubvectorEltIndex(p_sub, i, j + 1, k_upr);

    Sy = Sy_dat[ibot_mid];
    Ad_mid = Ad_dat[ibot_mid];
    Ad_lft = is_lft_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_lft];
    Ad_rgt = is_rgt_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_rgt];
    Ad_lwr = is_lwr_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_lwr];
    Ad_upr = is_upr_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_upr];

    // compute pressure head in adjacent cells
    old_head_mid = old_pressure[ip_mid] + 0.5 * Ad_mid;
    new_head_mid = new_pressure[ip_mid] + 0.5 * Ad_mid;
    new_head_lft = new_pressure[ip_lft] + 0.5 * Ad_lft;
    new_head_rgt = new_pressure[ip_rgt] + 0.5 * Ad_rgt;
    new_head_lwr = new_pressure[ip_lwr] + 0.5 * Ad_lwr;
    new_head_upr = new_pressure[ip_upr] + 0.5 * Ad_upr;

    // compute transmissivity at the cell faces
    // the aquifer is assumed to be fully saturated: Kr = 1
    Tx_mid = new_head_mid * Ks_x[ip_mid];
    Ty_mid = new_head_mid * Ks_y[ip_mid];

    Tx_lft = HarmonicMean(new_head_lft * Ks_x[ip_lft], Tx_mid);
    Tx_rgt = HarmonicMean(new_head_rgt * Ks_x[ip_rgt], Tx_mid);
    Ty_lwr = HarmonicMean(new_head_lwr * Ks_y[ip_lwr], Ty_mid);
    Ty_upr = HarmonicMean(new_head_upr * Ks_y[ip_upr], Ty_mid);

    // compute difference in pressure head
    dh_dt = new_head_mid - old_head_mid;

    if (is_lft_edge)
    {
      dh_rgt = new_head_rgt - new_head_mid;
      dh_lft = dh_rgt;
    }
    else if (is_rgt_edge)
    {
      dh_lft = new_head_mid - new_head_lft;
      dh_rgt = dh_lft;
    }
    else
    {
      dh_rgt = new_head_rgt - new_head_mid;
      dh_lft = new_head_mid - new_head_lft;
    }

    if (is_lwr_edge)
    {
      dh_upr = new_head_upr - new_head_mid;
      dh_lwr = dh_upr;
    }
    else if (is_upr_edge)
    {
      dh_lwr = new_head_mid - new_head_lwr;
      dh_upr = dh_lwr;
    }
    else
    {
      dh_lwr = new_head_mid - new_head_lwr;
      dh_upr = new_head_upr - new_head_mid;
    }

    // compute flux terms
    q_storage = dxdy * Sy * dh_dt;
    q_divergence = dtdy_over_dx * (Tx_rgt * dh_rgt - Tx_lft * dh_lft)
                   + dtdx_over_dy * (Ty_upr * dh_upr - Ty_lwr * dh_lwr);
  }),
                       FACE(FrontFace, DoNothing),
                       CellFinalize({
    PlusEquals(fp[ip_mid], q_storage - q_divergence);
  }),
                       AfterAllCells(DoNothing)
                       ); /* End DeepAquifer case */
  return;
}

/**
 * @name DeepAquifer Eval Jacobian
 * @brief Evaluation of the DeepAquifer jacobian
 *
 * This module can evaluates the non-linear function.
 * The jacobian evaluation is added to `J_sub` arg.
 *
 *
 * @param J_sub groundwaterflow jacobian evaluation
 * @param bc_struct boundary condition structure
 * @param subgrid current subgrid
 * @param p_sub new pressure subvector
 * @param old_pressure old pressure data
 * @param dt timestep
 * @param Kr relative permeability
 * @param Ks_x permeability_x subvector data
 * @param Ks_y permeability_y subvector data
 * @param ipatch current patch
 * @param isubgrid current subgrid
 * @param problem_data geometry data for problem
 */
void DeepAquiferEvalJacob(Submatrix *  J_sub,
                          BCStruct *   bc_struct,
                          Subgrid *    subgrid,
                          Subvector *  p_sub,
                          double *     old_pressure,
                          double       dt,
                          double *     Kr,
                          double *     Ks_x,
                          double *     Ks_y,
                          int          ipatch,
                          int          isubgrid,
                          ProblemData *problem_data)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  double *new_pressure = SubvectorData(p_sub);

  Vector    *bottom = ProblemDataIndexOfDomainBottom(problem_data);
  Subvector *bottom_sub = VectorSubvector(bottom, isubgrid);
  double    *bottom_dat = SubvectorData(bottom_sub);

  Vector    *Sy = instance_xtra->SpecificYield;
  Vector    *Ad = instance_xtra->AquiferDepth;
  Subvector *Sy_sub = VectorSubvector(Sy, isubgrid);
  double    *Sy_dat = SubvectorData(Sy_sub);
  Subvector *Ad_sub = VectorSubvector(Ad, isubgrid);
  double    *Ad_dat = SubvectorData(Ad_sub);

  double *cp = SubmatrixStencilData(J_sub, 0);
  double *wp = SubmatrixStencilData(J_sub, 1);
  double *ep = SubmatrixStencilData(J_sub, 2);
  double *sop = SubmatrixStencilData(J_sub, 3);
  double *np = SubmatrixStencilData(J_sub, 4);

  int i = 0, j = 0, k = 0, ival = 0;

  ForPatchCellsPerFace(DeepAquiferBC,
                       BeforeAllCells(DoNothing),
                       LoopVars(i, j, k, ival, bc_struct, ipatch, isubgrid),
                       Locals(
                              double Sy = 0.0;
                              double Ad_mid = 0.0;
                              double Ad_lft = 0.0;
                              double Ad_rgt = 0.0;
                              double Ad_lwr = 0.0;
                              double Ad_upr = 0.0;

                              double dx = SubgridDX(subgrid);
                              double dy = SubgridDY(subgrid);
                              double dxdy = dx * dy;
                              double dtdx_over_dy = dt * dx / dy;
                              double dtdy_over_dx = dt * dy / dx;

                              int im = 0;

                              int ibot_mid = 0;
                              int ibot_lft = 0;
                              int ibot_rgt = 0;
                              int ibot_lwr = 0;
                              int ibot_upr = 0;

                              int k_lft = 0;
                              int k_rgt = 0;
                              int k_lwr = 0;
                              int k_upr = 0;

                              int ip_mid = 0;
                              int ip_lft = 0;
                              int ip_rgt = 0;
                              int ip_lwr = 0;
                              int ip_upr = 0;

                              int is_lft_edge = 0;
                              int is_rgt_edge = 0;
                              int is_lwr_edge = 0;
                              int is_upr_edge = 0;

                              double Tx_mid = 0.0, Ty_mid = 0.0;
                              double Tx_lft = 0.0, Tx_rgt = 0.0;
                              double Ty_lwr = 0.0, Ty_upr = 0.0;

                              double new_head_mid = 0.0;
                              double new_head_lft = 0.0, new_head_rgt = 0.0;
                              double new_head_lwr = 0.0, new_head_upr = 0.0;

                              double dh_lft = 0.0, dh_rgt = 0.0;
                              double dh_lwr = 0.0, dh_upr = 0.0;

                              double del_mid_dh_lft = 0.0;
                              double del_lft_dh_lft = 0.0;
                              double del_rgt_dh_lft = 0.0;
                              double del_mid_dh_rgt = 0.0;
                              double del_rgt_dh_rgt = 0.0;
                              double del_lft_dh_rgt = 0.0;

                              double del_mid_dh_lwr = 0.0;
                              double del_lwr_dh_lwr = 0.0;
                              double del_upr_dh_lwr = 0.0;
                              double del_mid_dh_upr = 0.0;
                              double del_lwr_dh_upr = 0.0;
                              double del_upr_dh_upr = 0.0;

                              double del_mid_Tx_mid = 0.0;
                              double del_mid_Ty_mid = 0.0;

                              double del_mid_Tx_lft = 0.0;
                              double del_mid_Tx_rgt = 0.0;
                              double del_mid_Ty_lwr = 0.0;
                              double del_mid_Ty_upr = 0.0;

                              double del_lft_Tx_lft = 0.0;
                              double del_rgt_Tx_rgt = 0.0;
                              double del_lwr_Ty_lwr = 0.0;
                              double del_upr_Ty_upr = 0.0;

                              double del_mid_q_storage = 0.0;
                              double del_mid_q_divergence = 0.0;
                              double del_lft_q_divergence = 0.0;
                              double del_rgt_q_divergence = 0.0;
                              double del_lwr_q_divergence = 0.0;
                              double del_upr_q_divergence = 0.0;
                              ),
                       CellSetup({
    ip_mid = SubvectorEltIndex(p_sub, i, j, k);
    im = SubmatrixEltIndex(J_sub, i, j, k);

    del_mid_q_storage = 0.0;
    del_mid_q_divergence = 0.0;
    del_lft_q_divergence = 0.0;
    del_rgt_q_divergence = 0.0;
    del_lwr_q_divergence = 0.0;
    del_upr_q_divergence = 0.0;

    PF_UNUSED(ival);
  }),
                       FACE(LeftFace, DoNothing),
                       FACE(RightFace, DoNothing),
                       FACE(DownFace, DoNothing),
                       FACE(UpFace, DoNothing),
                       FACE(BackFace,
  {
    ibot_mid = SubvectorEltIndex(bottom_sub, i, j, 0);
    ibot_lft = SubvectorEltIndex(bottom_sub, i - 1, j, 0);
    ibot_rgt = SubvectorEltIndex(bottom_sub, i + 1, j, 0);
    ibot_lwr = SubvectorEltIndex(bottom_sub, i, j - 1, 0);
    ibot_upr = SubvectorEltIndex(bottom_sub, i, j + 1, 0);

    k_lft = rint(bottom_dat[ibot_lft]);
    k_rgt = rint(bottom_dat[ibot_rgt]);
    k_lwr = rint(bottom_dat[ibot_lwr]);
    k_upr = rint(bottom_dat[ibot_upr]);

    // find if we are at an edge cell:
    is_lft_edge = (k_lft < 0);
    is_rgt_edge = (k_rgt < 0);
    is_lwr_edge = (k_lwr < 0);
    is_upr_edge = (k_upr < 0);

    ip_lft = is_lft_edge ? ip_mid : SubvectorEltIndex(p_sub, i - 1, j, k_lft);
    ip_rgt = is_rgt_edge ? ip_mid : SubvectorEltIndex(p_sub, i + 1, j, k_rgt);
    ip_lwr = is_lwr_edge ? ip_mid : SubvectorEltIndex(p_sub, i, j - 1, k_lwr);
    ip_upr = is_upr_edge ? ip_mid : SubvectorEltIndex(p_sub, i, j + 1, k_upr);

    Sy = Sy_dat[ibot_mid];
    Ad_mid = Ad_dat[ibot_mid];
    Ad_lft = is_lft_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_lft];
    Ad_rgt = is_rgt_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_rgt];
    Ad_lwr = is_lwr_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_lwr];
    Ad_upr = is_upr_edge ? Ad_dat[ibot_mid] : Ad_dat[ibot_upr];

    new_head_mid = new_pressure[ip_mid] + 0.5 * Ad_mid;
    new_head_lft = new_pressure[ip_lft] + 0.5 * Ad_lft;
    new_head_rgt = new_pressure[ip_rgt] + 0.5 * Ad_rgt;
    new_head_lwr = new_pressure[ip_lwr] + 0.5 * Ad_lwr;
    new_head_upr = new_pressure[ip_upr] + 0.5 * Ad_upr;

    // compute transmissivity at the cell faces
    Tx_mid = new_head_mid * Ks_x[ip_mid];
    Ty_mid = new_head_mid * Ks_y[ip_mid];

    Tx_lft = HarmonicMean(new_head_lft * Ks_x[ip_lft], Tx_mid);
    Tx_rgt = HarmonicMean(new_head_rgt * Ks_x[ip_rgt], Tx_mid);
    Ty_lwr = HarmonicMean(new_head_lwr * Ks_y[ip_lwr], Ty_mid);
    Ty_upr = HarmonicMean(new_head_upr * Ks_y[ip_upr], Ty_mid);


    // dTx[i,j,k] / dp[i,j,k]
    del_mid_Tx_mid = Ks_x[ip_mid];
    // dTy[i,j,k] / dp[i,j,k]
    del_mid_Ty_mid = Ks_y[ip_mid];

    // dTx[i-1/2,j,k] / dp[i,j,k]
    del_mid_Tx_lft = is_lft_edge ? del_mid_Tx_mid :
                     DelHarmonicMean(new_head_lft * Ks_x[ip_lft], Tx_mid,
                                     0, del_mid_Tx_mid, Tx_lft);

    // dTx[i+1/2,j,k] / dp[i,j,k]
    del_mid_Tx_rgt = is_rgt_edge ? del_mid_Tx_mid :
                     DelHarmonicMean(new_head_rgt * Ks_x[ip_rgt], Tx_mid,
                                     0, del_mid_Tx_mid, Tx_rgt);

    // dTy[i,j-1/2,k] / dp[i,j,k]
    del_mid_Ty_lwr = is_lwr_edge ? del_mid_Ty_mid :
                     DelHarmonicMean(new_head_lwr * Ks_y[ip_lwr], Ty_mid,
                                     0, del_mid_Ty_mid, Ty_lwr);

    // dTy[i,j+1/2,k] / dp[i,j,k]
    del_mid_Ty_upr = is_upr_edge ? del_mid_Ty_mid :
                     DelHarmonicMean(new_head_upr * Ks_y[ip_upr], Ty_mid,
                                     0, del_mid_Ty_mid, Ty_upr);

    // Non-diagonal terms

    // dTx[i-1/2,j,k] / dp[i-1,j,k]
    del_lft_Tx_lft = is_lft_edge ? 0.0 :
                     DelHarmonicMean(new_head_lft * Ks_x[ip_lft], Tx_mid,
                                     Ks_x[ip_lft], 0.0, Tx_lft);

    // dTx[i+1/2,j,k] / dp[i+1,j,k]
    del_rgt_Tx_rgt = is_rgt_edge ? 0.0 :
                     DelHarmonicMean(new_head_rgt * Ks_x[ip_rgt], Tx_mid,
                                     Ks_x[ip_rgt], 0.0, Tx_rgt);

    // dTy[i,j-1/2,k] / dp[i,j-1,k]
    del_lwr_Ty_lwr = is_lwr_edge ? 0.0 :
                     DelHarmonicMean(new_head_lwr * Ks_y[ip_lwr], Ty_mid,
                                     Ks_y[ip_lwr], 0.0, Ty_lwr);

    // dTy[i,j+1/2,k] / dp[i,j+1,k]
    del_upr_Ty_upr = is_upr_edge ? 0.0 :
                     DelHarmonicMean(new_head_upr * Ks_y[ip_upr], Ty_mid,
                                     Ks_y[ip_upr], 0.0, Ty_upr);


    if (is_lft_edge)
    {
      dh_rgt = new_head_rgt - new_head_mid;
      dh_lft = dh_rgt;
    }
    else if (is_rgt_edge)
    {
      dh_lft = new_head_mid - new_head_lft;
      dh_rgt = dh_lft;
    }
    else
    {
      dh_rgt = new_head_rgt - new_head_mid;
      dh_lft = new_head_mid - new_head_lft;
    }

    del_mid_dh_lft = is_lft_edge ? -1.0 :  1.0;
    del_lft_dh_lft = is_lft_edge ?  0.0 : -1.0;
    del_rgt_dh_lft = is_lft_edge ?  1.0 :  0.0;
    del_mid_dh_rgt = is_rgt_edge ?  1.0 : -1.0;
    del_lft_dh_rgt = is_rgt_edge ? -1.0 :  0.0;
    del_rgt_dh_rgt = is_rgt_edge ?  0.0 :  1.0;

    if (is_lwr_edge)
    {
      dh_upr = new_head_upr - new_head_mid;
      dh_lwr = dh_upr;
    }
    else if (is_upr_edge)
    {
      dh_lwr = new_head_mid - new_head_lwr;
      dh_upr = dh_lwr;
    }
    else
    {
      dh_lwr = new_head_mid - new_head_lwr;
      dh_upr = new_head_upr - new_head_mid;
    }

    del_mid_dh_lwr = is_lwr_edge ? -1.0 :  1.0;
    del_lwr_dh_lwr = is_lwr_edge ?  0.0 : -1.0;
    del_upr_dh_lwr = is_lwr_edge ?  1.0 :  0.0;
    del_mid_dh_upr = is_upr_edge ?  1.0 : -1.0;
    del_lwr_dh_upr = is_upr_edge ? -1.0 :  0.0;
    del_upr_dh_upr = is_upr_edge ?  0.0 :  1.0;

    // dq_storage[i,j,k] / dp[i,j,k]
    del_mid_q_storage = dxdy * Sy;

    // dq_divergence[i,j,k] / dp[i,j,k]
    del_mid_q_divergence = dtdy_over_dx * (
                                           (del_mid_Tx_rgt * dh_rgt + Tx_rgt * del_mid_dh_rgt)
                                           - (del_mid_Tx_lft * dh_lft + Tx_lft * del_mid_dh_lft)
                                           ) + dtdx_over_dy * (
                                                               (del_mid_Ty_upr * dh_upr + Ty_upr * del_mid_dh_upr)
                                                               - (del_mid_Ty_lwr * dh_lwr + Ty_lwr * del_mid_dh_lwr)
                                                               );

    // dq_divergence[i,j,k] / dp[i-1,j,k]
    del_lft_q_divergence = dtdy_over_dx * (Tx_rgt * del_lft_dh_rgt
                                           - Tx_lft * del_lft_dh_lft - del_lft_Tx_lft * dh_lft);
    // dq_divergence[i,j,k] / dp[i+1,j,k]
    del_rgt_q_divergence = dtdy_over_dx * (Tx_rgt * del_rgt_dh_rgt
                                           + del_rgt_Tx_rgt * dh_rgt - Tx_lft * del_rgt_dh_lft);
    // dq_divergence[i,j,k] / dp[i,j-1,k]
    del_lwr_q_divergence = dtdx_over_dy * (Ty_upr * del_lwr_dh_upr
                                           - Ty_lwr * del_lwr_dh_lwr - del_lwr_Ty_lwr * dh_lwr);
    // dq_divergence[i,j,k] / dp[i,j+1,k]
    del_upr_q_divergence = dtdx_over_dy * (Ty_upr * del_upr_dh_upr
                                           + del_upr_Ty_upr * dh_upr - Ty_lwr * del_upr_dh_lwr);
  }),
                       FACE(FrontFace, DoNothing),
                       CellFinalize({
    /*
     * IMPORTANT:
     * cp[im]  = dF[i,j,k] / dp[i,j,k]
     * wp[im]  = dF[i,j,k] / dp[i-1,j,k]
     * ep[im]  = dF[i,j,k] / dp[i+1,j,k]
     * np[im]  = dF[i,j,k] / dp[i,j-1,k]
     * sop[im] = dF[i,j,k] / dp[i,j+1,k]
     * lp[im]  = dF[i,j,k] / dp[i,j,k-1]
     * up[im]  = dF[i,j,k] / dp[i,j,k+1]
     */
    PlusEquals(cp[im], del_mid_q_storage - del_mid_q_divergence);
    PlusEquals(wp[im], -del_lft_q_divergence);
    PlusEquals(ep[im], -del_rgt_q_divergence);
    PlusEquals(sop[im], -del_lwr_q_divergence);
    PlusEquals(np[im], -del_upr_q_divergence);
  }),
                       AfterAllCells(DoNothing)
                       ); /* End DeepAquifer case */
  return;
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule* DeepAquiferEvalInitInstanceXtra(ProblemData *problem_data)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = ctalloc(InstanceXtra, 1);

  NameArray na_types = NA_NewNameArray("Constant PFBFile");

  ParameterUnion Sy;

  GetParameterUnion(Sy, na_types,
                    "Patch.BCPressure.DeepAquifer.SpecificYield.%s",
                    ParameterUnionDouble(0, "Value")
                    ParameterUnionString(1, "Filename")
                    );

  ParameterUnion Ad;
  GetParameterUnion(Ad, na_types,
                    "Patch.BCPressure.DeepAquifer.AquiferDepth.%s",
                    ParameterUnionDouble(0, "Value")
                    ParameterUnionString(1, "Filename")
                    );

  InitDeepAquiferParameter(ProblemDataSpecificYield(problem_data), Sy);
  instance_xtra->SpecificYield = ProblemDataSpecificYield(problem_data);
  InitDeepAquiferParameter(ProblemDataAquiferDepth(problem_data), Ad);
  instance_xtra->AquiferDepth = ProblemDataAquiferDepth(problem_data);

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/
void DeepAquiferEvalFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra =
    (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule* DeepAquiferEvalNewPublicXtra()
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * DeepAquiferEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void DeepAquiferEvalFreePublicXtra()
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int DeepAquiferEvalSizeOfTempData()
{
  return 0;
}
