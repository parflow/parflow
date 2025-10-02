/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2025, Lawrence Livermore National Security,
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
#include "parflow_netcdf.h"

typedef void PublicXtra;

typedef void InstanceXtra;


/*--------------------------------------------------------------------------
 * DeepAquiferEval
 *--------------------------------------------------------------------------*/
/**
 * @name DeepAquifer BC Module
 * @brief Evaluation of the DeepAquifer boundary
 *
 * This module can evaluate both the non-linear function
 * and its jacobian. When invoked, the `fcn` flag tracks
 * which output to compute.
 *
 * @param problem_data problem data structure - holds deepaquifer parameters
 * @param pressure current step's pressure values
 * @param bc_struct structure necessary for BC looping
 * @param ipatch current patch index
 * @param isubgrid current subgrid index
 * @param ke_ return array for east face
 * @param kw_ return array for west face
 * @param kn_ return array for north face
 * @param ks_ return array for south face
 * @param fcn flag = {CALCFCN , CALCDER} - computes function or derivative
 */
void DeepAquiferEval(ProblemData *problem_data,
                     Vector *     pressure,
                     BCStruct *   bc_struct,
                     int          ipatch,
                     int          isubgrid,
                     double *     ke_,
                     double *     kw_,
                     double *     kn_,
                     double *     ks_,
                     int          fcn)
{
  Vector *permeability = ProblemDataDeepAquiferPermeability(problem_data);
  Vector *elevation = ProblemDataDeepAquiferElevation(problem_data);
  Vector *bottom = ProblemDataIndexOfDomainBottom(problem_data);

  Subvector *p_sub = VectorSubvector(pressure, isubgrid);
  Subvector *Ks_sub = VectorSubvector(permeability, isubgrid);
  Subvector *El_sub = VectorSubvector(elevation, isubgrid);
  Subvector *bottom_sub = VectorSubvector(bottom, isubgrid);

  double *pp = SubvectorData(p_sub);
  double *Ks = SubvectorData(Ks_sub);
  double *El = SubvectorData(El_sub);
  double *bottom_dat = SubvectorData(bottom_sub);

  double Ad = ProblemDataDeepAquiferAquiferDepth(problem_data);

  int i = 0, j = 0, k = 0, ival = 0;

  if (fcn == CALCFCN)
  {
    ForPatchCellsPerFace(BC_ALL,
                         BeforeAllCells(DoNothing),
                         LoopVars(i, j, k, ival, bc_struct, ipatch, isubgrid),
                         Locals(int ibot_mid = 0;
                                int ibot_lft = 0, ibot_rgt = 0;
                                int ibot_lwr = 0, ibot_upr = 0;

                                int k_lft = 0, k_rgt = 0;
                                int k_lwr = 0, k_upr = 0;

                                int ip_mid = 0;
                                int ip_lft = 0, ip_rgt = 0;
                                int ip_lwr = 0, ip_upr = 0;

                                int is_lft_edge = FALSE;
                                int is_rgt_edge = FALSE;
                                int is_lwr_edge = FALSE;
                                int is_upr_edge = FALSE;

                                double T_mid = 0.0;
                                double T_lft = 0.0, T_rgt = 0.0;
                                double T_lwr = 0.0, T_upr = 0.0;

                                double head_mid = 0.0;
                                double head_lft = 0.0, head_rgt = 0.0;
                                double head_lwr = 0.0, head_upr = 0.0;

                                double dh_lft = 0.0, dh_rgt = 0.0;
                                double dh_lwr = 0.0, dh_upr = 0.0;
                                ),
                         CellSetup({
      ip_mid = SubvectorEltIndex(p_sub, i, j, k);
      PF_UNUSED(ival);
    }),
                         FACE(LeftFace, DoNothing),
                         FACE(RightFace, DoNothing),
                         FACE(DownFace, DoNothing),
                         FACE(UpFace, DoNothing),
                         FACE(BackFace,
    {
      /*   i-1  i  i+1
       *      +---+
       *      |upr|     j+1
       *  +---+---+---+
       *  |lft|mid|rgt|  j
       *  +---+---+---+
       *      |lwr|     j-1
       *      +---+
       */

      // get indices of bottom cell in current and adjacent cells
      ibot_mid = SubvectorEltIndex(bottom_sub, i, j, 0);
      ibot_lft = SubvectorEltIndex(bottom_sub, i - 1, j, 0);
      ibot_rgt = SubvectorEltIndex(bottom_sub, i + 1, j, 0);
      ibot_lwr = SubvectorEltIndex(bottom_sub, i, j - 1, 0);
      ibot_upr = SubvectorEltIndex(bottom_sub, i, j + 1, 0);

      // while bottom index is not available, assume flat bottom
      k_lft = rint(bottom_dat[ibot_lft]);
      k_rgt = rint(bottom_dat[ibot_rgt]);
      k_lwr = rint(bottom_dat[ibot_lwr]);
      k_upr = rint(bottom_dat[ibot_upr]);

      // find if we are at an edge cell:
      is_lft_edge = (k_lft < 0);
      is_rgt_edge = (k_rgt < 0);
      is_lwr_edge = (k_lwr < 0);
      is_upr_edge = (k_upr < 0);

      // if at edge, use current cell index
      ibot_lft = is_lft_edge ? ibot_mid : ibot_lft;
      ibot_rgt = is_rgt_edge ? ibot_mid : ibot_rgt;
      ibot_lwr = is_lwr_edge ? ibot_mid : ibot_lwr;
      ibot_upr = is_upr_edge ? ibot_mid : ibot_upr;

      // get indices of adjacent cells in 3d grid
      ip_lft = is_lft_edge ? ip_mid : SubvectorEltIndex(p_sub, i - 1, j, k_lft);
      ip_rgt = is_rgt_edge ? ip_mid : SubvectorEltIndex(p_sub, i + 1, j, k_rgt);
      ip_lwr = is_lwr_edge ? ip_mid : SubvectorEltIndex(p_sub, i, j - 1, k_lwr);
      ip_upr = is_upr_edge ? ip_mid : SubvectorEltIndex(p_sub, i, j + 1, k_upr);

      // compute pressure head in adjacent cells:
      // because Ad is constant, it cancels out in head differences
      // since head is not used anywhere else, we skip the 0.5 * Ad term.
      // also, we assume density and gravity equal to 1.
      // also, its missing adding the z coordinate of the bottom cell
      // but this is only relevant if the bottom is not flat.
      head_mid = pp[ip_mid] + El[ibot_mid]; // + 0.5 * Ad
      head_lft = pp[ip_lft] + El[ibot_lft]; // + 0.5 * Ad
      head_rgt = pp[ip_rgt] + El[ibot_rgt]; // + 0.5 * Ad
      head_lwr = pp[ip_lwr] + El[ibot_lwr]; // + 0.5 * Ad
      head_upr = pp[ip_upr] + El[ibot_upr]; // + 0.5 * Ad

      // compute transmissivity at the cell faces
      // the aquifer is assumed to be fully saturated: Kr = 1
      T_mid = Ks[ibot_mid] * Ad;

      // transmissivity at interface is given by harmonic mean
      T_lft = HarmonicMean(Ks[ibot_lft] * Ad, T_mid);
      T_rgt = HarmonicMean(Ks[ibot_rgt] * Ad, T_mid);
      T_lwr = HarmonicMean(Ks[ibot_lwr] * Ad, T_mid);
      T_upr = HarmonicMean(Ks[ibot_upr] * Ad, T_mid);

      // differences in head between adjacent cells
      // there is no flow across the domain edges
      dh_rgt = is_rgt_edge ? 0.0 : head_rgt - head_mid;
      dh_lft = is_lft_edge ? 0.0 : head_mid - head_lft;
      dh_upr = is_upr_edge ? 0.0 : head_upr - head_mid;
      dh_lwr = is_lwr_edge ? 0.0 : head_mid - head_lwr;

      // compute flux terms
      // this is unoptimised. fluxes from one cell
      // are not reused in adjacent cells.
      ke_[ibot_mid] = T_rgt * dh_rgt;
      kw_[ibot_mid] = T_lft * dh_lft;
      kn_[ibot_mid] = T_upr * dh_upr;
      ks_[ibot_mid] = T_lwr * dh_lwr;
    }),
                         FACE(FrontFace, DoNothing),
                         CellFinalize(DoNothing),
                         AfterAllCells(DoNothing)
                         );
  }
  else /* fcn == CALCDER *//*----------------------------------------------*/
  {
    /* derivs of KE,KW,KN,KS w.r.t. current cell (i,j,k) */

    ForPatchCellsPerFace(BC_ALL,
                         BeforeAllCells(DoNothing),
                         LoopVars(i, j, k, ival, bc_struct, ipatch, isubgrid),
                         Locals(int ibot_mid = 0;
                                int ibot_lft = 0, ibot_rgt = 0;
                                int ibot_lwr = 0, ibot_upr = 0;

                                int k_lft = 0, k_rgt = 0;
                                int k_lwr = 0, k_upr = 0;

                                int is_lft_edge = FALSE;
                                int is_rgt_edge = FALSE;
                                int is_lwr_edge = FALSE;
                                int is_upr_edge = FALSE;

                                double T_mid = 0.0;
                                double T_lft = 0.0, T_rgt = 0.0;
                                double T_lwr = 0.0, T_upr = 0.0;

                                double dh_lft_der = 0.0, dh_rgt_der = 0.0;
                                double dh_lwr_der = 0.0, dh_upr_der = 0.0;
                                ),
                         CellSetup({ PF_UNUSED(ival); }),
                         FACE(LeftFace, DoNothing),
                         FACE(RightFace, DoNothing),
                         FACE(DownFace, DoNothing),
                         FACE(UpFace, DoNothing),
                         FACE(BackFace,
    {
      // get indices of bottom cell in current and adjacent cells
      ibot_mid = SubvectorEltIndex(bottom_sub, i, j, 0);
      ibot_lft = SubvectorEltIndex(bottom_sub, i - 1, j, 0);
      ibot_rgt = SubvectorEltIndex(bottom_sub, i + 1, j, 0);
      ibot_lwr = SubvectorEltIndex(bottom_sub, i, j - 1, 0);
      ibot_upr = SubvectorEltIndex(bottom_sub, i, j + 1, 0);

      // while bottom index is not available, assume flat bottom
      k_lft = rint(bottom_dat[ibot_lft]);
      k_rgt = rint(bottom_dat[ibot_rgt]);
      k_lwr = rint(bottom_dat[ibot_lwr]);
      k_upr = rint(bottom_dat[ibot_upr]);

      // find if we are at an edge cell:
      is_lft_edge = (k_lft < 0);
      is_rgt_edge = (k_rgt < 0);
      is_lwr_edge = (k_lwr < 0);
      is_upr_edge = (k_upr < 0);

      // if at edge, use current cell index
      ibot_lft = is_lft_edge ? ibot_mid : ibot_lft;
      ibot_rgt = is_rgt_edge ? ibot_mid : ibot_rgt;
      ibot_lwr = is_lwr_edge ? ibot_mid : ibot_lwr;
      ibot_upr = is_upr_edge ? ibot_mid : ibot_upr;

      // compute transmissivity at the cell faces
      // the aquifer is assumed to be fully saturated: Kr = 1
      T_mid = Ks[ibot_mid] * Ad;

      // transmissivity at interface is given by harmonic mean
      T_lft = HarmonicMean(Ks[ibot_lft] * Ad, T_mid);
      T_rgt = HarmonicMean(Ks[ibot_rgt] * Ad, T_mid);
      T_lwr = HarmonicMean(Ks[ibot_lwr] * Ad, T_mid);
      T_upr = HarmonicMean(Ks[ibot_upr] * Ad, T_mid);

      // derivative of differences in head between adjacent cells
      // with respect to middle cell head
      dh_rgt_der = is_rgt_edge ? 0.0 : -1.0;
      dh_lft_der = is_lft_edge ? 0.0 :  1.0;
      dh_upr_der = is_upr_edge ? 0.0 : -1.0;
      dh_lwr_der = is_lwr_edge ? 0.0 :  1.0;

      // compute derivative of flux terms with respect to middle cell head.
      // the derivative of this BC is already symmetric, so no need for
      // non-symmetric storage.
      ke_[ibot_mid] = T_rgt * dh_rgt_der;
      kw_[ibot_mid] = T_lft * dh_lft_der;
      kn_[ibot_mid] = T_upr * dh_upr_der;
      ks_[ibot_mid] = T_lwr * dh_lwr_der;
    }),
                         FACE(FrontFace, DoNothing),
                         CellFinalize(DoNothing),
                         AfterAllCells(DoNothing)
                         );
  }

  return;
}

/*--------------------------------------------------------------------------
 * DeepAquiferEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule* DeepAquiferEvalInitInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = ctalloc(InstanceXtra, 1);

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

/*--------------------------------------------------------------------------
 * SetDeepAquiferPermeability
 *--------------------------------------------------------------------------*/

void SetDeepAquiferPermeability(ProblemData *problem_data)
{
  Vector *permeability = ProblemDataDeepAquiferPermeability(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant PFBFile NCFile SameAsBottomLayer");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.Value");
      double value = GetDouble(key);
      InitVectorAll(permeability, value);
      break;
    }

    case 1: // PFBFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.FileName");
      char *filename = GetString(key);
      ReadPFBinary(filename, permeability);

      VectorUpdateCommHandle *handle = InitVectorUpdate(permeability, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    case 2: // NCFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Permeability.FileName");
      char *filename = GetString(key);
      int time_step = 0;
      int dimensionality = 2;
      ReadPFNC(filename, permeability, "permeability", time_step, dimensionality);

      VectorUpdateCommHandle *handle = InitVectorUpdate(permeability, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    case 3: // SameAsBottomLayer
    {
      InitVectorAll(permeability, 0.0);

      GrGeomSolid *gr_solid = ProblemDataGrDomain(problem_data);
      // Since SetDeepAquiferPermeability is being called from
      // BCPressurePackageInvoke, which is the last module to be called from
      // SetProblemData, the permeabilities have already been set up.
      // Therefore, we can just copy the values from the bottom layer
      // without needing to worry if the permeability_z vector has been
      // initialized.
      Vector *permeability_z = ProblemDataPermeabilityZ(problem_data);
      Vector *bottom = ProblemDataIndexOfDomainBottom(problem_data);

      Grid         *grid3d = VectorGrid(permeability_z);
      SubgridArray *grid3d_subgrids = GridSubgrids(grid3d);

      Grid          *grid2d = VectorGrid(permeability);
      SubgridArray  *grid2d_subgrids = GridSubgrids(grid2d);

      int is = 0;
      ForSubgridI(is, grid3d_subgrids)
      {
        Subgrid *grid3d_subgrid = SubgridArraySubgrid(grid3d_subgrids, is);
        Subgrid *grid2d_subgrid = SubgridArraySubgrid(grid2d_subgrids, is);

        Subvector *permz_sub = VectorSubvector(permeability_z, is);
        Subvector *aquifer_perm_sub = VectorSubvector(permeability, is);
        Subvector *bottom_sub = VectorSubvector(bottom, is);

        int grid3d_ix = SubgridIX(grid3d_subgrid);
        int grid3d_iy = SubgridIY(grid3d_subgrid);
        int grid3d_iz = SubgridIZ(grid3d_subgrid);

        int grid3d_nx = SubgridNX(grid3d_subgrid);
        int grid3d_ny = SubgridNY(grid3d_subgrid);
        int grid3d_nz = SubgridNZ(grid3d_subgrid);

        int grid3d_r = SubgridRX(grid3d_subgrid);

        int grid2d_iz = SubgridIZ(grid2d_subgrid);

        double *aquifer_perm_dat = SubvectorData(aquifer_perm_sub);
        double *permzp = SubvectorData(permz_sub);
        double *bottom_dat = SubvectorData(bottom_sub);

        int i = 0, j = 0, k = 0;

        GrGeomInLoop(i, j, k,
                     gr_solid, grid3d_r,
                     grid3d_ix, grid3d_iy, grid3d_iz,
                     grid3d_nx, grid3d_ny, grid3d_nz,
        {
          int index2d = SubvectorEltIndex(aquifer_perm_sub, i, j, grid2d_iz);
          int k_bottom = rint(bottom_dat[index2d]);
          k_bottom = (k_bottom < 0) ? 0 : k_bottom;
          int index3d = SubvectorEltIndex(permz_sub, i, j, k_bottom);

          aquifer_perm_dat[index2d] = permzp[index3d];
        });
      } /* End of subgrid loop */

      VectorUpdateCommHandle *handle = InitVectorUpdate(permeability, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferSpecificYield
 *--------------------------------------------------------------------------*/

void SetDeepAquiferSpecificYield(ProblemData *problem_data)
{
  Vector *specific_yield = ProblemDataDeepAquiferSpecificYield(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant PFBFile NCFile");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.Value");
      double value = GetDouble(key);
      InitVectorAll(specific_yield, value);
      break;
    }

    case 1: // PFBFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.FileName");
      char *filename = GetString(key);
      ReadPFBinary(filename, specific_yield);

      VectorUpdateCommHandle *handle = InitVectorUpdate(specific_yield, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    case 2: // NCFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.SpecificYield.FileName");
      char *filename = GetString(key);
      int time_step = 0;
      int dimensionality = 2;
      ReadPFNC(filename, specific_yield, "specific_yield", time_step, dimensionality);

      VectorUpdateCommHandle *handle = InitVectorUpdate(specific_yield, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferAquiferDepth
 *--------------------------------------------------------------------------*/

void SetDeepAquiferAquiferDepth(ProblemData *problem_data)
{
  NameArray switch_na = NA_NewNameArray("Constant");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.AquiferDepth.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.AquiferDepth.Value");
      double value = GetDouble(key);
      ProblemDataDeepAquiferAquiferDepth(problem_data) = value;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * SetDeepAquiferElevation
 *--------------------------------------------------------------------------*/

void SetDeepAquiferElevation(ProblemData *problem_data)
{
  Vector *elevation = ProblemDataDeepAquiferElevation(problem_data);
  NameArray switch_na = NA_NewNameArray("Constant PFBFile NCFile");
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Patch.BCPressure.DeepAquifer.Elevations.Type");

  char *switch_name = GetString(key);
  int switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
  switch (switch_value)
  {
    case 0: // Constant
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Elevations.Value");
      double value = GetDouble(key);
      InitVectorAll(elevation, value);
      break;
    }

    case 1: // PFBFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Elevations.FileName");
      char *filename = GetString(key);
      ReadPFBinary(filename, elevation);

      VectorUpdateCommHandle *handle = InitVectorUpdate(elevation, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    case 2: // NCFile
    {
      sprintf(key, "Patch.BCPressure.DeepAquifer.Elevations.FileName");
      char *filename = GetString(key);
      int time_step = 0;
      int dimensionality = 2;
      ReadPFNC(filename, elevation, "elevations", time_step, dimensionality);

      VectorUpdateCommHandle *handle = InitVectorUpdate(elevation, VectorUpdateAll);
      FinalizeVectorUpdate(handle);

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }
  NA_FreeNameArray(switch_na);

  return;
}

/*--------------------------------------------------------------------------
 * DeepAquiferCheckPermeabilityTensorValues
 *--------------------------------------------------------------------------*/

void DeepAquiferCheckPermeabilityTensorValues()
{
  /* This method throws out an error if the user hasn't set
   * the X and Y components of the tensor to zero. */

  NameArray type_na = NA_NewNameArray("TensorByGeom TensorByFile");
  char *switch_name = GetString("Perm.TensorType");
  int tensor_type = NA_NameToIndexExitOnError(type_na, switch_name, "TensorByGeom TensorByFile");

  // Only TensorByGeom is supported for this boundary condition
  if (tensor_type != 0)
  {
    InputError("Error: DeepAquifer boundary condition only supports "
               "setting the key <Perm.TensorType> to \"TensorByGeom\".%s%s\n", "", "");
  }

  char *geometry_names = GetString("Geom.Perm.TensorByGeom.Names");
  NameArray geometry_na = NA_NewNameArray(geometry_names);
  int Ngeometries = NA_Sizeof(geometry_na);

  char key[IDB_MAX_KEY_LEN];
  for (int i = 0; i < Ngeometries; ++i)
  {
    char *geom_name = NA_IndexToName(geometry_na, i);

    sprintf(key, "Geom.%s.Perm.TensorValX", geom_name);
    double tensor_x = GetDouble(key);

    sprintf(key, "Geom.%s.Perm.TensorValY", geom_name);
    double tensor_y = GetDouble(key);

    if (fabs(tensor_x) + fabs(tensor_y) > 1e-14)
    {
      InputError("Error: To use DeepAquifer boundary condition, the keys "
                 "<Geom.%s.Perm.TensorValX> and <Geom.%s.Perm.TensorValY> must "
                 "be set to 0.\n", geom_name, geom_name);
    }
  }

  return;
}