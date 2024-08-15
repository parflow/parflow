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
    double      *q_groundwater, // @return: groundwaterflow function evaluation
    BCStruct    *bc_struct,     // boundary condition structure
    Subgrid     *subgrid,       // current subgrid
    Subvector   *p_sub,         // new pressure subvector
    double      *old_pressure,  // old pressure data
    double       dt,            // timestep
    double      *perm_x,        // permeability_x subvector data
    double      *perm_y,        // permeability_y subvector data
    double      *z_mult,        // dz multiplication factor
    int          ipatch,        // current patch
    int          isubgrid,      // current subgrid
    ProblemData *problem_data)  // geometry data for problem
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
    }),
    FACE(LeftFace,  DoNothing),
    FACE(RightFace, DoNothing),
    FACE(DownFace,  DoNothing),
    FACE(UpFace,    DoNothing),
    FACE(BackFace,
    {
      Sy = ParameterAt(instance_xtra->SpecificYield, ival);
      Ad = ParameterAt(instance_xtra->AquiferDepth, ival);

      // find if we are at an edge cell:
      // is_lft_edge = (i-SubgridIX(subgrid) == 0);
      // is_rgt_edge = (i-SubgridIX(subgrid) == (nx-1));
      // is_dwn_edge = (j-SubgridIY(subgrid) == 0);
      // is_top_edge = (j-SubgridIY(subgrid) == (ny-1));
      is_lft_edge = (top_data[itop+lft] < 0);
      is_rgt_edge = (top_data[itop+rgt] < 0);
      is_dwn_edge = (top_data[itop+dwn] < 0);
      is_top_edge = (top_data[itop+top] < 0);
      
      // int lft_edge = top_data[itop+lft] < 0;
      // int rgt_edge = top_data[itop+rgt] < 0;
      // int dwn_edge = top_data[itop+dwn] < 0;
      // int top_edge = top_data[itop+top] < 0;

      // amps_Printf(
      //   "Dims: %d %d, Cell: %d %d, IsEdge: %d %d %d %d, Edge: %d %d %d %d\n",
      //   nx, ny, i-SubgridIX(subgrid), j-SubgridIY(subgrid), 
      //   is_lft_edge, is_rgt_edge, is_dwn_edge, is_top_edge,
      //   lft_edge, rgt_edge, dwn_edge, top_edge);

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

      // compute transmissivity at the cell faces
      Tx_mid = new_head_mid * perm_x[ip+mid];
      Ty_mid = new_head_mid * perm_y[ip+mid];

      Tx_lft = is_lft_edge ? 
          Tx_mid : HarmonicMean(new_head_lft * perm_x[ip+lft], Tx_mid);
      Tx_rgt = is_rgt_edge ? 
          Tx_mid : HarmonicMean(new_head_rgt * perm_x[ip+rgt], Tx_mid);
      Ty_dwn = is_dwn_edge ? 
          Ty_mid : HarmonicMean(new_head_dwn * perm_y[ip+dwn], Ty_mid);
      Ty_top = is_top_edge ? 
          Ty_mid : HarmonicMean(new_head_top * perm_y[ip+top], Ty_mid);

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
      q_groundwater[ip] = q_storage - q_divergence;
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
