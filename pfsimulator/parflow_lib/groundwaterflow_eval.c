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

#include "groundwaterflow_eval.h"
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

  Vector    *top      = ProblemDataIndexOfDomainTop(problem_data);
  Subvector *top_sub  = VectorSubvector(top, isubgrid);
  double    *top_data = SubvectorData(top_sub);

  int i = 0, j = 0, k = 0, ival = 0;
  ForPatchCellsPerFace(
    GroundwaterFlowBC,
    BeforeAllCells(DoNothing), 
    LoopVars(i, j, k, ival, bc_struct, ipatch, isubgrid),
    Locals(
      double Sy = 0.0, Ad = 0.0;
      double dx = SubgridDX(subgrid);
      double dy = SubgridDY(subgrid);
      double dz = SubgridDZ(subgrid);
      // int nx = SubgridNX(subgrid);
      // int ny = SubgridNY(subgrid);
      double dxdy = dx*dy;
      double dtdx_over_dy = dt * dx / dy;
      double dtdy_over_dx = dt * dy / dx;

      int stride_xp = 1, stride_yp = SubvectorNX(p_sub);

      int itop = 0;
      int ip = 0;
      int mid = 0; // middle cell index
      int lft = -stride_xp; // left  cell index
      int rgt = +stride_xp; // right cell index
      int dwn = -stride_yp; // down  cell index
      int top = +stride_yp; // top   cell index

      int is_lft_edge = 0;      
      int is_rgt_edge = 0;
      int is_dwn_edge = 0;
      int is_top_edge = 0;

      double q_storage = 0.0, q_divergence = 0.0;

      double KrKs_x_mid = 0.0;
      double KrKs_x_lft = 0.0;
      double KrKs_x_rgt = 0.0;
      double KrKs_y_mid = 0.0;
      double KrKs_y_dwn = 0.0;
      double KrKs_y_top = 0.0;

      double Tx_mid = 0.0, Ty_mid = 0.0;
      double Tx_lft = 0.0, Tx_rgt = 0.0;
      double Ty_dwn = 0.0, Ty_top = 0.0;

      double old_head_mid = 0.0, new_head_mid = 0.0;
      double new_head_lft = 0.0, new_head_rgt = 0.0;
      double new_head_dwn = 0.0, new_head_top = 0.0;

      double dh_lft = 0.0, dh_rgt = 0.0;
      double dh_dwn = 0.0, dh_top = 0.0;
      double dh_dt  = 0.0;
    ),
    CellSetup({
      ip = SubvectorEltIndex(p_sub, i, j, k);
      itop = SubvectorEltIndex(top_sub, i, j, 0);

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
      Sy = ParameterAt(instance_xtra->SpecificYield, itop);
      Ad = ParameterAt(instance_xtra->AquiferDepth, itop);

      // find if we are at an edge cell:
      is_lft_edge = (top_data[itop+lft] < 0);
      is_rgt_edge = (top_data[itop+rgt] < 0);
      is_dwn_edge = (top_data[itop+dwn] < 0);
      is_top_edge = (top_data[itop+top] < 0);

      // compute pressure head in adjacent cells
      // [CAUTION] Ad index is ival (2D), while perm index is ip (3D); 
      // [CAUTION] how to solve this?
      // double hh_mid = pp[ip+mid] + 0.5 * Ad[ival+mid];
      // double hh_lft = pp[ip+lft] + 0.5 * Ad[ival+lft];
      // double hh_rgt = pp[ip+rgt] + 0.5 * Ad[ival+rgt];
      // double hh_dwn = pp[ip+dwn] + 0.5 * Ad[ival+dwn];
      // double hh_top = pp[ip+top] + 0.5 * Ad[ival+top];

      // no special cases in the middle (everything is known)
      old_head_mid = old_pressure[ip+mid] + 0.5*(Ad + dz*z_mult[ip+mid]);
      new_head_mid = new_pressure[ip+mid] + 0.5*(Ad + dz*z_mult[ip+mid]);

      new_head_lft = is_lft_edge ? 
          new_head_mid : new_pressure[ip+lft] + 0.5*(Ad + dz*z_mult[ip+lft]);
      new_head_rgt = is_rgt_edge ? 
          new_head_mid : new_pressure[ip+rgt] + 0.5*(Ad + dz*z_mult[ip+rgt]);

      new_head_dwn = is_dwn_edge ? 
          new_head_mid : new_pressure[ip+dwn] + 0.5*(Ad + dz*z_mult[ip+dwn]);
      new_head_top = is_top_edge ? 
          new_head_mid : new_pressure[ip+top] + 0.5*(Ad + dz*z_mult[ip+top]);

      // hydraulic conductivity times relative permeability
      // KrKs_x_mid = Kr[ip+mid] * Ks_x[ip+mid];
      // KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Kr[ip+lft] * Ks_x[ip+lft];
      // KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Kr[ip+rgt] * Ks_x[ip+rgt];
      // KrKs_y_mid = Kr[ip+mid] * Ks_y[ip+mid];
      // KrKs_y_dwn = is_dwn_edge ? KrKs_y_mid : Kr[ip+dwn] * Ks_y[ip+dwn];
      // KrKs_y_top = is_top_edge ? KrKs_y_mid : Kr[ip+top] * Ks_y[ip+top];

      KrKs_x_mid = Ks_x[ip+mid];
      KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Ks_x[ip+lft];
      KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Ks_x[ip+rgt];
      KrKs_y_mid = Ks_y[ip+mid];
      KrKs_y_dwn = is_dwn_edge ? KrKs_y_mid : Ks_y[ip+dwn];
      KrKs_y_top = is_top_edge ? KrKs_y_mid : Ks_y[ip+top];

      // compute transmissivity at the cell faces
      Tx_mid = new_head_mid * KrKs_x_mid;
      Ty_mid = new_head_mid * KrKs_y_mid;

      Tx_lft = is_lft_edge ? Tx_mid : 
        HarmonicMean(new_head_lft * KrKs_x_lft, Tx_mid);
      Tx_rgt = is_rgt_edge ? Tx_mid : 
        HarmonicMean(new_head_rgt * KrKs_x_rgt, Tx_mid);
      Ty_dwn = is_dwn_edge ? Ty_mid : 
        HarmonicMean(new_head_dwn * KrKs_y_dwn, Ty_mid);
      Ty_top = is_top_edge ? Ty_mid : 
        HarmonicMean(new_head_top * KrKs_y_top, Ty_mid);

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

      if(is_dwn_edge) {
        dh_top = new_head_top - new_head_mid;
        dh_dwn = dh_top;
      } else if(is_top_edge) {
        dh_dwn = new_head_mid - new_head_dwn;
        dh_top = dh_dwn;
      } else {
        dh_dwn = new_head_mid - new_head_dwn;
        dh_top = new_head_top - new_head_mid;
      }

      // compute flux terms
      q_storage = dxdy * Sy * dh_dt;
      q_divergence = dtdy_over_dx * (Tx_rgt * dh_rgt - Tx_lft * dh_lft)
          + dtdx_over_dy * (Ty_top * dh_top - Ty_dwn * dh_dwn);
    }),
    FACE(FrontFace, DoNothing),
    CellFinalize({
      fp[ip] += q_storage - q_divergence;
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

  Vector    *top      = ProblemDataIndexOfDomainTop(problem_data);
  Subvector *top_sub  = VectorSubvector(top, isubgrid);
  double    *top_data = SubvectorData(top_sub);
  
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
      double Sy = 0.0, Ad = 0.0;
      double dx = SubgridDX(subgrid);
      double dy = SubgridDY(subgrid);
      double dz = SubgridDZ(subgrid);
      double dxdy = dx*dy;
      double dtdx_over_dy = dt * dx / dy;
      double dtdy_over_dx = dt * dy / dx;

      int stride_xp = 1, stride_yp = SubvectorNX(p_sub);

      int itop = 0;
      int ip = 0;
      int im = 0;
      int mid = 0; // middle cell index
      int lft = -stride_xp; // left  cell index
      int rgt = +stride_xp; // right cell index
      int dwn = -stride_yp; // down  cell index
      int top = +stride_yp; // top   cell index

      int is_lft_edge = 0;      
      int is_rgt_edge = 0;
      int is_dwn_edge = 0;
      int is_top_edge = 0;

      double Tx_mid = 0.0, Ty_mid = 0.0;
      double Tx_lft = 0.0, Tx_rgt = 0.0;
      double Ty_dwn = 0.0, Ty_top = 0.0;

      double KrKs_x_mid = 0.0;
      double KrKs_x_lft = 0.0;
      double KrKs_x_rgt = 0.0;
      double KrKs_y_mid = 0.0;
      double KrKs_y_dwn = 0.0;
      double KrKs_y_top = 0.0;

      double del_KrKs_x_mid = 0.0;
      double del_KrKs_x_lft = 0.0;
      double del_KrKs_x_rgt = 0.0;
      double del_KrKs_y_mid = 0.0;
      double del_KrKs_y_dwn = 0.0;
      double del_KrKs_y_top = 0.0;

      double new_head_mid = 0.0;
      double new_head_lft = 0.0, new_head_rgt = 0.0;
      double new_head_dwn = 0.0, new_head_top = 0.0;

      double dh_lft = 0.0, dh_rgt = 0.0;
      double dh_dwn = 0.0, dh_top = 0.0;

      double del_mid_dh_lft = 0.0;
      double del_lft_dh_lft = 0.0;
      double del_rgt_dh_lft = 0.0;
      double del_mid_dh_rgt = 0.0;
      double del_rgt_dh_rgt = 0.0;
      double del_lft_dh_rgt = 0.0;

      double del_mid_dh_dwn = 0.0;
      double del_dwn_dh_dwn = 0.0;
      double del_top_dh_dwn = 0.0;
      double del_mid_dh_top = 0.0;
      double del_dwn_dh_top = 0.0;
      double del_top_dh_top = 0.0;

      double del_mid_Tx_mid = 0.0;
      double del_mid_Ty_mid = 0.0;

      double del_mid_Tx_lft = 0.0;
      double del_mid_Tx_rgt = 0.0;
      double del_mid_Ty_dwn = 0.0;
      double del_mid_Ty_top = 0.0;

      double del_lft_Tx_lft = 0.0;
      double del_rgt_Tx_rgt = 0.0;
      double del_dwn_Ty_dwn = 0.0;
      double del_top_Ty_top = 0.0;

      double del_mid_q_storage    = 0.0;
      double del_mid_q_divergence = 0.0;
      double del_lft_q_divergence = 0.0;
      double del_rgt_q_divergence = 0.0;
      double del_dwn_q_divergence = 0.0;
      double del_top_q_divergence = 0.0;
    ),
    CellSetup({
      ip = SubvectorEltIndex(p_sub, i, j, k);
      itop = SubvectorEltIndex(top_sub, i, j, 0);
      im = SubmatrixEltIndex(J_sub, i, j, k);

      del_mid_q_storage    = 0.0;
      del_mid_q_divergence = 0.0;
      del_lft_q_divergence = 0.0;
      del_rgt_q_divergence = 0.0;
      del_dwn_q_divergence = 0.0;
      del_top_q_divergence = 0.0;

      PF_UNUSED(ival);
    }),
    FACE(LeftFace,  DoNothing),
    FACE(RightFace, DoNothing),
    FACE(DownFace,  DoNothing),
    FACE(UpFace,    DoNothing),
    FACE(BackFace,
    {
      // find if we are at an edge cell:
      is_lft_edge = (top_data[itop+lft] < 0);
      is_rgt_edge = (top_data[itop+rgt] < 0);
      is_dwn_edge = (top_data[itop+dwn] < 0);
      is_top_edge = (top_data[itop+top] < 0);

      Sy = ParameterAt(instance_xtra->SpecificYield, itop);
      Ad = ParameterAt(instance_xtra->AquiferDepth, itop);

      // no special cases in the middle (everything is known)
      new_head_mid = new_pressure[ip+mid] + 0.5*(Ad + dz*z_mult[ip+mid]);

      new_head_lft = is_lft_edge ? 
          new_head_mid : new_pressure[ip+lft] + 0.5*(Ad + dz*z_mult[ip+lft]);
      new_head_rgt = is_rgt_edge ? 
          new_head_mid : new_pressure[ip+rgt] + 0.5*(Ad + dz*z_mult[ip+rgt]);

      new_head_dwn = is_dwn_edge ? 
          new_head_mid : new_pressure[ip+dwn] + 0.5*(Ad + dz*z_mult[ip+dwn]);
      new_head_top = is_top_edge ? 
          new_head_mid : new_pressure[ip+top] + 0.5*(Ad + dz*z_mult[ip+top]);

      // hydraulic conductivity times relative permeability
      // KrKs_x_mid = Kr[ip+mid] * Ks_x[ip+mid];
      // KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Kr[ip+lft] * Ks_x[ip+lft];
      // KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Kr[ip+rgt] * Ks_x[ip+rgt];
      // KrKs_y_mid = Kr[ip+mid] * Ks_y[ip+mid];
      // KrKs_y_dwn = is_dwn_edge ? KrKs_y_mid : Kr[ip+dwn] * Ks_y[ip+dwn];
      // KrKs_y_top = is_top_edge ? KrKs_y_mid : Kr[ip+top] * Ks_y[ip+top];
      
      KrKs_x_mid = Ks_x[ip+mid];
      KrKs_x_lft = is_lft_edge ? KrKs_x_mid : Ks_x[ip+lft];
      KrKs_x_rgt = is_rgt_edge ? KrKs_x_mid : Ks_x[ip+rgt];
      KrKs_y_mid = Ks_y[ip+mid];
      KrKs_y_dwn = is_dwn_edge ? KrKs_y_mid : Ks_y[ip+dwn];
      KrKs_y_top = is_top_edge ? KrKs_y_mid : Ks_y[ip+top];
      
      // del_KrKs_x_mid = del_Kr[ip+mid] * Ks_x[ip+mid];
      // del_KrKs_x_lft = is_lft_edge ? 
      //     del_KrKs_x_mid : del_Kr[ip+lft] * Ks_x[ip+lft];
      // del_KrKs_x_rgt = is_rgt_edge ? 
      //     del_KrKs_x_mid : del_Kr[ip+rgt] * Ks_x[ip+rgt];
      // del_KrKs_y_mid = del_Kr[ip+mid] * Ks_y[ip+mid];
      // del_KrKs_y_dwn = is_dwn_edge ? 
      //     del_KrKs_y_mid : del_Kr[ip+dwn] * Ks_y[ip+dwn];
      // del_KrKs_y_top = is_top_edge ? 
      //     del_KrKs_y_mid : del_Kr[ip+top] * Ks_y[ip+top];

      del_KrKs_x_mid = 0.0;
      del_KrKs_x_lft = 0.0;
      del_KrKs_x_rgt = 0.0;
      del_KrKs_y_mid = 0.0;
      del_KrKs_y_dwn = 0.0;
      del_KrKs_y_top = 0.0;

      // compute transmissivity at the cell faces
      Tx_mid = new_head_mid * KrKs_x_mid;
      Ty_mid = new_head_mid * KrKs_y_mid;

      Tx_lft = is_lft_edge ? Tx_mid : 
        HarmonicMean(new_head_lft * KrKs_x_lft, Tx_mid);
      Tx_rgt = is_rgt_edge ? Tx_mid : 
        HarmonicMean(new_head_rgt * KrKs_x_rgt, Tx_mid);
      Ty_dwn = is_dwn_edge ? Ty_mid : 
        HarmonicMean(new_head_dwn * KrKs_y_dwn, Ty_mid);
      Ty_top = is_top_edge ? Ty_mid : 
        HarmonicMean(new_head_top * KrKs_y_top, Ty_mid);


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
      del_mid_Ty_dwn = is_dwn_edge ? del_mid_Ty_mid : 
        DelHarmonicMean(new_head_dwn * KrKs_y_dwn, Ty_mid,
            new_head_dwn * del_KrKs_y_dwn, del_mid_Ty_mid, Ty_dwn);

      // dTy[i,j+1/2,k] / dp[i,j,k]
      del_mid_Ty_top = is_top_edge ? del_mid_Ty_mid : 
        DelHarmonicMean(new_head_top * KrKs_y_top, Ty_mid,
            new_head_top * del_KrKs_y_top, del_mid_Ty_mid, Ty_top);

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
      del_dwn_Ty_dwn = is_dwn_edge ? 0.0 : 
        DelHarmonicMean(new_head_dwn * KrKs_y_dwn, Ty_mid,
            (KrKs_y_dwn + new_head_dwn * del_KrKs_y_dwn), 0.0, Ty_dwn);

      // dTy[i,j+1/2,k] / dp[i,j+1,k]
      del_top_Ty_top = is_top_edge ? 0.0 : 
        DelHarmonicMean(new_head_top * KrKs_y_top, Ty_mid,
            (KrKs_y_top + new_head_top * del_KrKs_y_top), 0.0, Ty_top);


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

      if(is_dwn_edge) {
        dh_top = new_head_top - new_head_mid;
        dh_dwn = dh_top;
      } else if(is_top_edge) {
        dh_dwn = new_head_mid - new_head_dwn;
        dh_top = dh_dwn;
      } else {
        dh_dwn = new_head_mid - new_head_dwn;
        dh_top = new_head_top - new_head_mid;
      }

      del_mid_dh_dwn = is_dwn_edge ? -1.0 :  1.0;
      del_dwn_dh_dwn = is_dwn_edge ?  0.0 : -1.0;
      del_top_dh_dwn = is_dwn_edge ?  1.0 :  0.0;
      del_mid_dh_top = is_top_edge ?  1.0 : -1.0;
      del_dwn_dh_top = is_top_edge ? -1.0 :  0.0;
      del_top_dh_top = is_top_edge ?  0.0 :  1.0;

      // dq_storage[i,j,k] / dp[i,j,k]
      del_mid_q_storage = dxdy * Sy;

      // dq_divergence[i,j,k] / dp[i,j,k]
      del_mid_q_divergence = dtdy_over_dx * (
           (del_mid_Tx_rgt * dh_rgt + Tx_rgt * del_mid_dh_rgt)
          -(del_mid_Tx_lft * dh_lft + Tx_lft * del_mid_dh_lft)
          ) + dtdx_over_dy * (
           (del_mid_Ty_top * dh_top + Ty_top * del_mid_dh_top)
          -(del_mid_Ty_dwn * dh_dwn + Ty_dwn * del_mid_dh_dwn)
          );

      // dq_divergence[i,j,k] / dp[i-1,j,k]
      del_lft_q_divergence = dtdy_over_dx * (Tx_rgt * del_lft_dh_rgt 
          - Tx_lft * del_lft_dh_lft - del_lft_Tx_lft * dh_lft);
      // dq_divergence[i,j,k] / dp[i+1,j,k]
      del_rgt_q_divergence = dtdy_over_dx * (Tx_rgt * del_rgt_dh_rgt 
          + del_rgt_Tx_rgt * dh_rgt - Tx_lft * del_rgt_dh_lft);
      // dq_divergence[i,j,k] / dp[i,j-1,k]
      del_dwn_q_divergence = dtdx_over_dy * (Ty_top * del_dwn_dh_top 
          - Ty_dwn * del_dwn_dh_dwn - del_dwn_Ty_dwn * dh_dwn);
      // dq_divergence[i,j,k] / dp[i,j+1,k]
      del_top_q_divergence = dtdx_over_dy * (Ty_top * del_top_dh_top 
          + del_top_Ty_top * dh_top - Ty_dwn * del_top_dh_dwn);
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
      sop[im] += -del_dwn_q_divergence;
      np[im]  += -del_top_q_divergence;
      // PlusEquals(cp[im],  del_mid_q_storage - del_mid_q_divergence+del_mid_q_divergence);
      // PlusEquals(wp[im],  -del_lft_q_divergence+del_lft_q_divergence);
      // PlusEquals(ep[im],  -del_rgt_q_divergence+del_rgt_q_divergence);
      // PlusEquals(sop[im], -del_dwn_q_divergence+del_dwn_q_divergence);
      // PlusEquals(np[im],  -del_top_q_divergence+del_top_q_divergence);
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
