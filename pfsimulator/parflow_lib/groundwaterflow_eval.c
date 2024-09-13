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

#define Constant 0
#define PFBFile  1

#define DelHarmonicMean(a, b, da, db, hm) ( !((a)+(b)) ? 0 :   \
    (2*((da)*(b) + (a)*(db)) - (hm)*((da) + (db))) / ((a)+(b)) \
  )

typedef struct {
  void* data;
  int type;
} GroundwaterFlowParameter;

typedef void PublicXtra;

typedef struct {
  void *SpecificYield;
  void *AquiferDepth;
} InstanceXtra;

double ParameterAt(void *parameter, int ival) {
  GroundwaterFlowParameter *par = (GroundwaterFlowParameter*)parameter;
  switch(par->type) {
    case Constant:
    {
      return *((double*)par->data);
    }
    case PFBFile: // [todo] implement parameters read from file
    {
      return 0.0;
    }
    default:
    {
      return 0.0;
    }
  }
}

void* NewGroundwaterFlowParameter(char *patch_name, char *parameter_name)
{
  GroundwaterFlowParameter *par = ctalloc(GroundwaterFlowParameter, 1);
  // Valid types of parameter value assignment
  NameArray type_na = NA_NewNameArray("Constant PFBFile");

  // entry key from which to read info
  char key[IDB_MAX_KEY_LEN];
  sprintf(key, "Patch.%s.BCPressure.GroundwaterFlow.%s.Type", 
      patch_name, parameter_name);

  char* type_name = GetString(key);
  int type = NA_NameToIndexExitOnError(type_na, type_name, key);
  par->type = type;

  switch(type) 
  {
    case Constant:
    {
      sprintf(key, "Patch.%s.BCPressure.GroundwaterFlow.%s.Value", 
          patch_name, parameter_name);

      double *data = ctalloc(double, 1);
      (*data) = GetDouble(key);
      par->data = (void*)data;
      break;
    }
    case PFBFile:
    {
      sprintf(key, "Patch.%s.BCPressure.GroundwaterFlow.%s.FileName", 
          patch_name, parameter_name);
      char *data = GetString(key);
      par->data = (void*)data;
      break;
    }
    default:
    {
      par->data = NULL;
      break;
    }
  }

  NA_FreeNameArray(type_na);

  return (void*)par;
}


void FreeGroundwaterFlowParameter(void *parameter) 
{
  GroundwaterFlowParameter *par = (GroundwaterFlowParameter*)parameter;
  if(!par) return;

  switch(par->type) 
  {
    case Constant:
    {
      tfree(par->data);
      break;
    }
    case PFBFile: // not implemented yet
    {
      tfree(par->data);
      break;
    }
    default:
    {
      break;
    }
  }

  tfree(par);
  return;
}



/*--------------------------------------------------------------------------
 * GroundwaterFlowEval
 *--------------------------------------------------------------------------*/

void GroundwaterFlowEval(
    void        *groundwater_out, // @return: groundwaterflow evaluation
    int          fcn,           // flag = {CALCFCN , CALCDER}
    BCStruct    *bc_struct,     // boundary condition structure
    Subgrid     *subgrid,       // current subgrid
    Subvector   *p_sub,         // new pressure subvector
    double      *old_pressure,  // old pressure data
    double       dt,            // timestep
    double      *Kr,           // relative permeability
    double      *del_Kr,       // derivative of the relative permebility
    double      *Ks_x,         // permeability_x subvector data
    double      *Ks_y,         // permeability_y subvector data
    double      *z_mult,        // dz multiplication factor
    int          ipatch,        // current patch
    int          isubgrid,      // current subgrid
    ProblemData *problem_data)  // geometry data for problem
{
  if(fcn == CALCFCN) {

    double *fp = (double*)groundwater_out;

    GroundwaterFlowEvalNLFunc(fp, bc_struct, subgrid, 
        p_sub, old_pressure, dt, Kr, del_Kr, Ks_x, Ks_y,
        z_mult, ipatch, isubgrid, problem_data);

  } else { /* fcn == CALCDER */
    
    Submatrix *J_sub = (Submatrix*)groundwater_out;

    GroundwaterFlowEvalJacob(J_sub, bc_struct, subgrid, 
        p_sub, old_pressure, dt, Kr, del_Kr, Ks_x, Ks_y,
        z_mult, ipatch, isubgrid, problem_data);
  }

  return;
}

void GroundwaterFlowEvalNLFunc(
    double      *fp,           // @return: groundwaterflow function evaluation
    BCStruct    *bc_struct,    // boundary condition structure
    Subgrid     *subgrid,      // current subgrid
    Subvector   *p_sub,        // new pressure subvector
    double      *old_pressure, // old pressure data
    double       dt,           // timestep
    double      *Kr,           // relative permeability
    double      *del_Kr,       // derivative of the relative permebility
    double      *Ks_x,         // permeability_x subvector data
    double      *Ks_y,         // permeability_y subvector data
    double      *z_mult,       // dz multiplication factor
    int          ipatch,       // current patch
    int          isubgrid,     // current subgrid
    ProblemData *problem_data) // geometry data for problem
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = 
      (InstanceXtra*)PFModuleInstanceXtra(this_module);

  double *new_pressure = SubvectorData(p_sub);

  Vector    *bottom      = ProblemDataIndexOfDomainBottom(problem_data);
  Subvector *bottom_sub  = VectorSubvector(bottom, isubgrid);
  double    *bottom_dat  = SubvectorData(bottom_sub);

  int i = 0, j = 0, k = 0, ival = 0;
  ForPatchCellsPerFace(
    GroundwaterFlowBC,
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
      double dz = SubgridDZ(subgrid);
      // int nx = SubgridNX(subgrid);
      // int ny = SubgridNY(subgrid);
      double dxdy = dx*dy;
      double dtdx_over_dy = dt * dx / dy;
      double dtdy_over_dx = dt * dy / dx;

      int ibot_mid = 0;
      int ibot_lft = 0;
      int ibot_rgt = 0;
      int ibot_lwr = 0;
      int ibot_upr = 0;

      int k_mid = 0; 
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

      double KrKs_x_mid = 0.0;
      double KrKs_x_lft = 0.0;
      double KrKs_x_rgt = 0.0;
      double KrKs_y_mid = 0.0;
      double KrKs_y_lwr = 0.0;
      double KrKs_y_upr = 0.0;

      double Tx_mid = 0.0, Ty_mid = 0.0;
      double Tx_lft = 0.0, Tx_rgt = 0.0;
      double Ty_lwr = 0.0, Ty_upr = 0.0;

      double old_head_mid = 0.0, new_head_mid = 0.0;
      double new_head_lft = 0.0, new_head_rgt = 0.0;
      double new_head_lwr = 0.0, new_head_upr = 0.0;

      double dh_lft = 0.0, dh_rgt = 0.0;
      double dh_lwr = 0.0, dh_upr = 0.0;
      double dh_dt  = 0.0;
    ),
    CellSetup({
      ip_mid = SubvectorEltIndex(p_sub, i, j, k);
      
      q_storage = 0.0;
      q_divergence = 0.0;

      PF_UNUSED(ival);
    }),
    FACE(LeftFace,  DoNothing),
    FACE(RightFace, DoNothing),
    FACE(DownFace,  DoNothing),
    FACE(UpFace,    DoNothing),
    FACE(BackFace,
    {
      ibot_mid = SubvectorEltIndex(bottom_sub, i,   j, 0);
      ibot_lft = SubvectorEltIndex(bottom_sub, i-1, j, 0);
      ibot_rgt = SubvectorEltIndex(bottom_sub, i+1, j, 0);
      ibot_lwr = SubvectorEltIndex(bottom_sub, i, j-1, 0);
      ibot_upr = SubvectorEltIndex(bottom_sub, i, j+1, 0);

      k_mid = rint(bottom_dat[ibot_mid]);
      k_lft = rint(bottom_dat[ibot_lft]);
      k_rgt = rint(bottom_dat[ibot_rgt]);
      k_lwr = rint(bottom_dat[ibot_lwr]);
      k_upr = rint(bottom_dat[ibot_upr]);

      // amps_Printf("i: %d , j: %d k: %d , k_mid: %d \n", i, j, k, k_mid);
      // amps_Printf("k_mid: %d , k_lft: %d , k_rgt: %d , k_lwr: %d , k_upr: %d \n", k_mid, k_lft, k_rgt, k_lwr, k_upr);

      // find if we are at an edge cell:
      is_lft_edge = (k_lft < 0);
      is_rgt_edge = (k_rgt < 0);
      is_lwr_edge = (k_lwr < 0);
      is_upr_edge = (k_upr < 0);

      ip_lft = SubvectorEltIndex(p_sub, i-1, j, is_lft_edge ? k_mid : k_lft);
      ip_rgt = SubvectorEltIndex(p_sub, i+1, j, is_rgt_edge ? k_mid : k_rgt);
      ip_lwr = SubvectorEltIndex(p_sub, i, j-1, is_lwr_edge ? k_mid : k_lwr);
      ip_upr = SubvectorEltIndex(p_sub, i, j+1, is_upr_edge ? k_mid : k_upr);

      Sy = ParameterAt(instance_xtra->SpecificYield, ibot_mid);
      Ad_mid = ParameterAt(instance_xtra->AquiferDepth, ibot_mid);
      Ad_lft = ParameterAt(instance_xtra->AquiferDepth, 
          is_lft_edge ? ibot_mid : ibot_lft);
      Ad_rgt = ParameterAt(instance_xtra->AquiferDepth, 
          is_rgt_edge ? ibot_mid : ibot_rgt);
      Ad_lwr = ParameterAt(instance_xtra->AquiferDepth, 
          is_lwr_edge ? ibot_mid : ibot_lwr);
      Ad_upr = ParameterAt(instance_xtra->AquiferDepth, 
          is_upr_edge ? ibot_mid : ibot_upr);

      // compute pressure head in adjacent cells

      // no special cases in the middle (everything is known)
      old_head_mid = old_pressure[ip_mid] + 0.5*(Ad_mid + dz*z_mult[ip_mid]);
      new_head_mid = new_pressure[ip_mid] + 0.5*(Ad_mid + dz*z_mult[ip_mid]);

      new_head_lft = is_lft_edge ? new_head_mid : 
          new_pressure[ip_lft] + 0.5*(Ad_lft + dz*z_mult[ip_lft]);
      new_head_rgt = is_rgt_edge ? new_head_mid : 
          new_pressure[ip_rgt] + 0.5*(Ad_rgt + dz*z_mult[ip_rgt]);

      new_head_lwr = is_lwr_edge ? new_head_mid : 
          new_pressure[ip_lwr] + 0.5*(Ad_lwr + dz*z_mult[ip_lwr]);
      new_head_upr = is_upr_edge ? new_head_mid : 
          new_pressure[ip_upr] + 0.5*(Ad_upr + dz*z_mult[ip_upr]);

      // hydraulic conductivity times relative permeability
      // KrKs_x_mid = Kr[ip_mid] * Ks_x[ip_mid];
      // KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Kr[ip_lft] * Ks_x[ip_lft];
      // KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Kr[ip_rgt] * Ks_x[ip_rgt];
      // KrKs_y_mid = Kr[ip_mid] * Ks_y[ip_mid];
      // KrKs_y_lwr = is_lwr_edge ? KrKs_y_mid : Kr[ip_lwr] * Ks_y[ip_lwr];
      // KrKs_y_upr = is_upr_edge ? KrKs_y_mid : Kr[ip_upr] * Ks_y[ip_upr];

      KrKs_x_mid = Ks_x[ip_mid];
      KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Ks_x[ip_lft];
      KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Ks_x[ip_rgt];
      KrKs_y_mid = Ks_y[ip_mid];
      KrKs_y_lwr = is_lwr_edge ? KrKs_y_mid : Ks_y[ip_lwr];
      KrKs_y_upr = is_upr_edge ? KrKs_y_mid : Ks_y[ip_upr];

      // compute transmissivity at the cell faces
      Tx_mid = new_head_mid * KrKs_x_mid;
      Ty_mid = new_head_mid * KrKs_y_mid;

      Tx_lft = is_lft_edge ? Tx_mid : 
        HarmonicMean(new_head_lft * KrKs_x_lft, Tx_mid);
      Tx_rgt = is_rgt_edge ? Tx_mid : 
        HarmonicMean(new_head_rgt * KrKs_x_rgt, Tx_mid);
      Ty_lwr = is_lwr_edge ? Ty_mid : 
        HarmonicMean(new_head_lwr * KrKs_y_lwr, Ty_mid);
      Ty_upr = is_upr_edge ? Ty_mid : 
        HarmonicMean(new_head_upr * KrKs_y_upr, Ty_mid);

      // compute difference in pressure head
      dh_dt  = new_head_mid - old_head_mid;

      if(is_lft_edge) {
        dh_rgt = new_head_rgt - new_head_mid;
        dh_lft = dh_rgt;
      } else if(is_rgt_edge) {
        dh_lft = new_head_mid - new_head_lft;
        dh_rgt = dh_lft;
      } else {
        dh_rgt = new_head_rgt - new_head_mid;
        dh_lft = new_head_mid - new_head_lft;
      }

      if(is_lwr_edge) {
        dh_upr = new_head_upr - new_head_mid;
        dh_lwr = dh_upr;
      } else if(is_upr_edge) {
        dh_lwr = new_head_mid - new_head_lwr;
        dh_upr = dh_lwr;
      } else {
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
      fp[ip_mid] += q_storage - q_divergence;
      // PlusEquals(fp[ip], q_storage - q_divergence + q_divergence);
      // PlusEquals(fp[ip], q_storage - q_divergence);
    }),
    AfterAllCells(DoNothing)
  );    /* End GroundwaterFlow case */
  return;
}

void GroundwaterFlowEvalJacob(
    Submatrix   *J_sub,        // @return: groundwaterflow del evaluation
    BCStruct    *bc_struct,    // boundary condition structure
    Subgrid     *subgrid,      // current subgrid
    Subvector   *p_sub,        // new pressure subvector
    double      *old_pressure, // old pressure data
    double       dt,           // timestep
    double      *Kr,           // relative permeability
    double      *del_Kr,       // derivative of the relative permebility
    double      *Ks_x,         // permeability_x subvector data
    double      *Ks_y,         // permeability_y subvector data
    double      *z_mult,       // dz multiplication factor
    int          ipatch,       // current patch
    int          isubgrid,     // current subgrid
    ProblemData *problem_data) // geometry data for problem
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = 
      (InstanceXtra*)PFModuleInstanceXtra(this_module);

  double *new_pressure = SubvectorData(p_sub);

  Vector    *bottom      = ProblemDataIndexOfDomainBottom(problem_data);
  Subvector *bottom_sub  = VectorSubvector(bottom, isubgrid);
  double    *bottom_dat  = SubvectorData(bottom_sub);
  
  double *cp  = SubmatrixStencilData(J_sub, 0);
  double *wp  = SubmatrixStencilData(J_sub, 1);
  double *ep  = SubmatrixStencilData(J_sub, 2);
  double *sop = SubmatrixStencilData(J_sub, 3);
  double *np  = SubmatrixStencilData(J_sub, 4);

  int i = 0, j = 0, k = 0, ival = 0;
  ForPatchCellsPerFace(
    GroundwaterFlowBC,
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
      double dz = SubgridDZ(subgrid);
      double dxdy = dx*dy;
      double dtdx_over_dy = dt * dx / dy;
      double dtdy_over_dx = dt * dy / dx;

      int im = 0;

      int ibot_mid = 0;
      int ibot_lft = 0;
      int ibot_rgt = 0;
      int ibot_lwr = 0;
      int ibot_upr = 0;

      int k_mid = 0; 
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

      double KrKs_x_mid = 0.0;
      double KrKs_x_lft = 0.0;
      double KrKs_x_rgt = 0.0;
      double KrKs_y_mid = 0.0;
      double KrKs_y_lwr = 0.0;
      double KrKs_y_upr = 0.0;

      double del_KrKs_x_mid = 0.0;
      double del_KrKs_x_lft = 0.0;
      double del_KrKs_x_rgt = 0.0;
      double del_KrKs_y_mid = 0.0;
      double del_KrKs_y_lwr = 0.0;
      double del_KrKs_y_upr = 0.0;

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

      double del_mid_q_storage    = 0.0;
      double del_mid_q_divergence = 0.0;
      double del_lft_q_divergence = 0.0;
      double del_rgt_q_divergence = 0.0;
      double del_lwr_q_divergence = 0.0;
      double del_upr_q_divergence = 0.0;
    ),
    CellSetup({
      ip_mid = SubvectorEltIndex(p_sub, i, j, k);
      im = SubmatrixEltIndex(J_sub, i, j, k);

      del_mid_q_storage    = 0.0;
      del_mid_q_divergence = 0.0;
      del_lft_q_divergence = 0.0;
      del_rgt_q_divergence = 0.0;
      del_lwr_q_divergence = 0.0;
      del_upr_q_divergence = 0.0;

      PF_UNUSED(ival);
    }),
    FACE(LeftFace,  DoNothing),
    FACE(RightFace, DoNothing),
    FACE(DownFace,  DoNothing),
    FACE(UpFace,    DoNothing),
    FACE(BackFace,
    {
      ibot_mid = SubvectorEltIndex(bottom_sub, i,   j, 0);
      ibot_lft = SubvectorEltIndex(bottom_sub, i-1, j, 0);
      ibot_rgt = SubvectorEltIndex(bottom_sub, i+1, j, 0);
      ibot_lwr = SubvectorEltIndex(bottom_sub, i, j-1, 0);
      ibot_upr = SubvectorEltIndex(bottom_sub, i, j+1, 0);

      k_mid = rint(bottom_dat[ibot_mid]);
      k_lft = rint(bottom_dat[ibot_lft]);
      k_rgt = rint(bottom_dat[ibot_rgt]);
      k_lwr = rint(bottom_dat[ibot_lwr]);
      k_upr = rint(bottom_dat[ibot_upr]);

      // find if we are at an edge cell:
      is_lft_edge = (k_lft < 0);
      is_rgt_edge = (k_rgt < 0);
      is_lwr_edge = (k_lwr < 0);
      is_upr_edge = (k_upr < 0);

      ip_lft = SubvectorEltIndex(p_sub, i-1, j, is_lft_edge ? k_mid : k_lft);
      ip_rgt = SubvectorEltIndex(p_sub, i+1, j, is_rgt_edge ? k_mid : k_rgt);
      ip_lwr = SubvectorEltIndex(p_sub, i, j-1, is_lwr_edge ? k_mid : k_lwr);
      ip_upr = SubvectorEltIndex(p_sub, i, j+1, is_upr_edge ? k_mid : k_upr);

      Sy = ParameterAt(instance_xtra->SpecificYield, ibot_mid);
      Ad_mid = ParameterAt(instance_xtra->AquiferDepth, ibot_mid);
      Ad_lft = ParameterAt(instance_xtra->AquiferDepth, 
          is_lft_edge ? ibot_mid : ibot_lft);
      Ad_rgt = ParameterAt(instance_xtra->AquiferDepth, 
          is_rgt_edge ? ibot_mid : ibot_rgt);
      Ad_lwr = ParameterAt(instance_xtra->AquiferDepth, 
          is_lwr_edge ? ibot_mid : ibot_lwr);
      Ad_upr = ParameterAt(instance_xtra->AquiferDepth, 
          is_upr_edge ? ibot_mid : ibot_upr);

      // no special cases in the middle (everything is known)
      new_head_mid = new_pressure[ip_mid] + 0.5*(Ad_mid + dz*z_mult[ip_mid]);

      new_head_lft = is_lft_edge ? new_head_mid : 
          new_pressure[ip_lft] + 0.5*(Ad_lft + dz*z_mult[ip_lft]);
      new_head_rgt = is_rgt_edge ? new_head_mid : 
          new_pressure[ip_rgt] + 0.5*(Ad_rgt + dz*z_mult[ip_rgt]);

      new_head_lwr = is_lwr_edge ? new_head_mid : 
          new_pressure[ip_lwr] + 0.5*(Ad_lwr + dz*z_mult[ip_lwr]);
      new_head_upr = is_upr_edge ? new_head_mid : 
          new_pressure[ip_upr] + 0.5*(Ad_upr + dz*z_mult[ip_upr]);

      // hydraulic conductivity times relative permeability
      // KrKs_x_mid = Kr[ip_mid] * Ks_x[ip_mid];
      // KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Kr[ip_lft] * Ks_x[ip_lft];
      // KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Kr[ip_rgt] * Ks_x[ip_rgt];
      // KrKs_y_mid = Kr[ip_mid] * Ks_y[ip_mid];
      // KrKs_y_lwr = is_lwr_edge ? KrKs_y_mid : Kr[ip_lwr] * Ks_y[ip_lwr];
      // KrKs_y_upr = is_upr_edge ? KrKs_y_mid : Kr[ip_upr] * Ks_y[ip_upr];
      
      KrKs_x_mid = Ks_x[ip_mid];
      KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Ks_x[ip_lft];
      KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Ks_x[ip_rgt];
      KrKs_y_mid = Ks_y[ip_mid];
      KrKs_y_lwr = is_lwr_edge ? KrKs_y_mid : Ks_y[ip_lwr];
      KrKs_y_upr = is_upr_edge ? KrKs_y_mid : Ks_y[ip_upr];
      
      // del_KrKs_x_mid = del_Kr[ip_mid] * Ks_x[ip_mid];
      // del_KrKs_x_lft = is_lft_edge ? 
      //     del_KrKs_x_mid : del_Kr[ip_lft] * Ks_x[ip_lft];
      // del_KrKs_x_rgt = is_rgt_edge ? 
      //     del_KrKs_x_mid : del_Kr[ip_rgt] * Ks_x[ip_rgt];
      // del_KrKs_y_mid = del_Kr[ip_mid] * Ks_y[ip_mid];
      // del_KrKs_y_lwr = is_lwr_edge ? 
      //     del_KrKs_y_mid : del_Kr[ip_lwr] * Ks_y[ip_lwr];
      // del_KrKs_y_upr = is_upr_edge ? 
      //     del_KrKs_y_mid : del_Kr[ip_upr] * Ks_y[ip_upr];

      del_KrKs_x_mid = 0.0;
      del_KrKs_x_lft = 0.0;
      del_KrKs_x_rgt = 0.0;
      del_KrKs_y_mid = 0.0;
      del_KrKs_y_lwr = 0.0;
      del_KrKs_y_upr = 0.0;

      // compute transmissivity at the cell faces
      Tx_mid = new_head_mid * KrKs_x_mid;
      Ty_mid = new_head_mid * KrKs_y_mid;

      Tx_lft = is_lft_edge ? Tx_mid : 
        HarmonicMean(new_head_lft * KrKs_x_lft, Tx_mid);
      Tx_rgt = is_rgt_edge ? Tx_mid : 
        HarmonicMean(new_head_rgt * KrKs_x_rgt, Tx_mid);
      Ty_lwr = is_lwr_edge ? Ty_mid : 
        HarmonicMean(new_head_lwr * KrKs_y_lwr, Ty_mid);
      Ty_upr = is_upr_edge ? Ty_mid : 
        HarmonicMean(new_head_upr * KrKs_y_upr, Ty_mid);


      // dTx[i,j,k] / dp[i,j,k]
      del_mid_Tx_mid = KrKs_x_mid + new_head_mid * del_KrKs_x_mid;
      // dTy[i,j,k] / dp[i,j,k]
      del_mid_Ty_mid = KrKs_y_mid + new_head_mid * del_KrKs_y_mid;

      // dTx[i-1/2,j,k] / dp[i,j,k]
      del_mid_Tx_lft = is_lft_edge ? del_mid_Tx_mid : 
        DelHarmonicMean(new_head_lft * KrKs_x_lft, Tx_mid,
            new_head_lft * del_KrKs_x_lft, del_mid_Tx_mid, Tx_lft);

      // dTx[i+1/2,j,k] / dp[i,j,k]
      del_mid_Tx_rgt = is_rgt_edge ? del_mid_Tx_mid : 
        DelHarmonicMean(new_head_rgt * KrKs_x_rgt, Tx_mid,
            new_head_rgt * del_KrKs_x_rgt, del_mid_Tx_mid, Tx_rgt);

      // dTy[i,j-1/2,k] / dp[i,j,k]
      del_mid_Ty_lwr = is_lwr_edge ? del_mid_Ty_mid : 
        DelHarmonicMean(new_head_lwr * KrKs_y_lwr, Ty_mid,
            new_head_lwr * del_KrKs_y_lwr, del_mid_Ty_mid, Ty_lwr);

      // dTy[i,j+1/2,k] / dp[i,j,k]
      del_mid_Ty_upr = is_upr_edge ? del_mid_Ty_mid : 
        DelHarmonicMean(new_head_upr * KrKs_y_upr, Ty_mid,
            new_head_upr * del_KrKs_y_upr, del_mid_Ty_mid, Ty_upr);

      // Non-diagonal terms

      // dTx[i-1/2,j,k] / dp[i-1,j,k]
      del_lft_Tx_lft = is_lft_edge ? 0.0 : 
        DelHarmonicMean(new_head_lft * KrKs_x_lft, Tx_mid,
            (KrKs_x_lft + new_head_lft * del_KrKs_x_lft), 0.0, Tx_lft);

      // dTx[i+1/2,j,k] / dp[i+1,j,k]
      del_rgt_Tx_rgt = is_rgt_edge ? 0.0 : 
        DelHarmonicMean(new_head_rgt * KrKs_x_rgt, Tx_mid,
            (KrKs_x_rgt + new_head_rgt * del_KrKs_x_rgt), 0.0, Tx_rgt);

      // dTy[i,j-1/2,k] / dp[i,j-1,k]
      del_lwr_Ty_lwr = is_lwr_edge ? 0.0 : 
        DelHarmonicMean(new_head_lwr * KrKs_y_lwr, Ty_mid,
            (KrKs_y_lwr + new_head_lwr * del_KrKs_y_lwr), 0.0, Ty_lwr);

      // dTy[i,j+1/2,k] / dp[i,j+1,k]
      del_upr_Ty_upr = is_upr_edge ? 0.0 : 
        DelHarmonicMean(new_head_upr * KrKs_y_upr, Ty_mid,
            (KrKs_y_upr + new_head_upr * del_KrKs_y_upr), 0.0, Ty_upr);


      if(is_lft_edge) {
        dh_rgt = new_head_rgt - new_head_mid;
        dh_lft = dh_rgt;
      } else if(is_rgt_edge) {
        dh_lft = new_head_mid - new_head_lft;
        dh_rgt = dh_lft;
      } else {
        dh_rgt = new_head_rgt - new_head_mid;
        dh_lft = new_head_mid - new_head_lft;
      }

      del_mid_dh_lft = is_lft_edge ? -1.0 :  1.0;
      del_lft_dh_lft = is_lft_edge ?  0.0 : -1.0;
      del_rgt_dh_lft = is_lft_edge ?  1.0 :  0.0;
      del_mid_dh_rgt = is_rgt_edge ?  1.0 : -1.0;
      del_lft_dh_rgt = is_rgt_edge ? -1.0 :  0.0;
      del_rgt_dh_rgt = is_rgt_edge ?  0.0 :  1.0;

      if(is_lwr_edge) {
        dh_upr = new_head_upr - new_head_mid;
        dh_lwr = dh_upr;
      } else if(is_upr_edge) {
        dh_lwr = new_head_mid - new_head_lwr;
        dh_upr = dh_lwr;
      } else {
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
          -(del_mid_Tx_lft * dh_lft + Tx_lft * del_mid_dh_lft)
          ) + dtdx_over_dy * (
           (del_mid_Ty_upr * dh_upr + Ty_upr * del_mid_dh_upr)
          -(del_mid_Ty_lwr * dh_lwr + Ty_lwr * del_mid_dh_lwr)
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
        IMPORTANT:
        cp[im]  = dF[i,j,k] / dp[i,j,k]
        wp[im]  = dF[i,j,k] / dp[i-1,j,k]
        ep[im]  = dF[i,j,k] / dp[i+1,j,k]
        np[im]  = dF[i,j,k] / dp[i,j-1,k]
        sop[im] = dF[i,j,k] / dp[i,j+1,k]
        lp[im]  = dF[i,j,k] / dp[i,j,k-1]
        up[im]  = dF[i,j,k] / dp[i,j,k+1]
      */
      cp[im]  += del_mid_q_storage - del_mid_q_divergence;
      wp[im]  += -del_lft_q_divergence;
      ep[im]  += -del_rgt_q_divergence;
      sop[im] += -del_lwr_q_divergence;
      np[im]  += -del_upr_q_divergence;
      // PlusEquals(cp[im],  del_mid_q_storage - del_mid_q_divergence+del_mid_q_divergence);
      // PlusEquals(wp[im],  -del_lft_q_divergence+del_lft_q_divergence);
      // PlusEquals(ep[im],  -del_rgt_q_divergence+del_rgt_q_divergence);
      // PlusEquals(sop[im], -del_lwr_q_divergence+del_lwr_q_divergence);
      // PlusEquals(np[im],  -del_upr_q_divergence+del_upr_q_divergence);
    }),
    AfterAllCells(DoNothing)
  );    /* End GroundwaterFlow case */
  return;
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule* GroundwaterFlowEvalInitInstanceXtra(char *patch_name)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = ctalloc(InstanceXtra, 1);

  instance_xtra->SpecificYield = 
      NewGroundwaterFlowParameter(patch_name, "SpecificYield");
  instance_xtra->AquiferDepth  = 
      NewGroundwaterFlowParameter(patch_name, "AquiferDepth");

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/
void GroundwaterFlowEvalFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = 
      (InstanceXtra*)PFModuleInstanceXtra(this_module);

  FreeGroundwaterFlowParameter(instance_xtra->SpecificYield);
  FreeGroundwaterFlowParameter(instance_xtra->AquiferDepth);

  if(instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule* GroundwaterFlowEvalNewPublicXtra()
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * GroundwaterFlowEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void GroundwaterFlowEvalFreePublicXtra()
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if(public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int GroundwaterFlowEvalSizeOfTempData()
{
  return 0;
}
