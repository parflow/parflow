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

void NewGroundwaterFlowParameters(
    void** Sy, // Return: Specific Yield --------- void *var[interval_division]
    void** Ad, // Return: Aquifer Depth ---------- void *var[interval_division]
    void** Ar, // Return: Aquifer Recharge Rate -- void *var[interval_division]
    char* patch_name,      // patch with GroundwaterFlow boundary condition
    int global_cycle,      // current time cycle
    int interval_division) // number of time divisions of the current cycle
{
  Sy = ctalloc(void*, interval_division); // Specific Yield
  Ad = ctalloc(void*, interval_division); // Aquifer Depth
  Ar = ctalloc(void*, interval_division); // Aquifer Recharge

  void **parameters[3] = {Sy, Ad, Ar};
  // Parameter name array
  NameArray parameter_na = 
      NA_NewNameArray("SpecificYield AquiferDepth AquiferRecharge");
  // Valid types of parameter value assignment
  NameArray type_na = NA_NewNameArray("Constant PFBFile");
  // entry key from which to read info
  char key[IDB_MAX_KEY_LEN];

  // loop through the intervals
  int interval_number = 0;
  ForEachInterval(interval_division, interval_number)
  {
    // loop through the parameters
    for(int p = 0; p < NA_Sizeof(parameter_na); ++p) 
    {
      GroundwaterFlowParameter* par = ctalloc(GroundwaterFlowParameter, 1);
      parameters[p][interval_number] = (void*)par;

      sprintf(key, "Patch.%s.BCPressure.%s.%s.Type", patch_name, 
          NA_IndexToName(GlobalsIntervalNames[global_cycle], interval_number),
          NA_IndexToName(parameter_na, p));

      char* type_name = GetString(key);
      int type = NA_NameToIndexExitOnError(type_na, type_name, key);
      par->type = type;

      switch(type) {
        case Constant:
        {
          sprintf(key, "Patch.%s.BCPressure.%s.%s.Value", patch_name, 
              NA_IndexToName(GlobalsIntervalNames[global_cycle], 
              interval_number), NA_IndexToName(parameter_na, p));

          double* data = ctalloc(double, 1);
          (*data) = GetDouble(key);
          par->data = (void*)data;
        }
        case PFBFile: // not implemented yet
        {
          // sprintf(key, "Patch.%s.BCPressure.%s.%s.FileName", patch_name, 
          //     NA_IndexToName(GlobalsIntervalNames[global_cycle], 
          //     interval_number), NA_IndexToName(parameter_na, p));
          break;
        }
      }
    }
  }

  NA_FreeNameArray(parameter_na);
  NA_FreeNameArray(type_na);

  return;
}

void FreeGroundwaterFlowParameters(
    void** Sy, void** Ad, void** Ar, int interval_division) {
  
  int interval_number = 0;
  ForEachInterval(interval_division, interval_number)
  {
    FreeGroundwaterFlowParameter((Sy[interval_number]));
    FreeGroundwaterFlowParameter((Ad[interval_number]));
    FreeGroundwaterFlowParameter((Ar[interval_number]));
  }

  tfree(Sy);
  tfree(Ad);
  tfree(Ar);
}

void FreeGroundwaterFlowParameter(void* parameter) 
{
  GroundwaterFlowParameter *par = (GroundwaterFlowParameter*)parameter;

  switch(par->type) {
    case Constant:
    {
      tfree(par->data);
      break;
    }
    case PFBFile: // not implemented yet
    {
      break;
    }
  }

  tfree(par);
  return;
}

void SetGroundwaterFlowParameter(
    double* values,  // @return: fill with parameter values
    void* parameter, // ptr to GroundwaterFlowParameter object
    BCStruct* bc_struct, // structure that holds boundary condition information
    int ipatch, // current patch
    int is)     // current subgrid
{
  GroundwaterFlowParameter* par = (GroundwaterFlowParameter*)parameter;

  switch(par->type) {
    case Constant:
    {
      int i = 0, j = 0, k = 0, ival = 0;
      ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is, {
        values[ival] = (*((double*)par->data));
      });
      break;
    }
    case PFBFile: // not implemented yet
    {
      break;
    }
  }

  return;
}


/*--------------------------------------------------------------------------
 * GroundwaterFlowEval
 *--------------------------------------------------------------------------*/

typedef GroundwaterFlowPublicXtra PublicXtra;
typedef GroundwaterFlowInstanceXtra InstanceXtra;

void GroundwaterFlowEval(
    double* q_groundwater, // @return: groundwaterflow function evaluation
    BCStruct* bc_struct, // boundary condition structure
    Subgrid* subgrid, // current subgrid
    Subvector* p_sub, // pressure subvector
    double* new_pressure, // new pressure data
    double* old_pressure, // old pressure data
    double dt, // timestep
    double* perm_x, // permeability_x subvector data
    double* perm_y, // permeability_y subvector data
    double* perm_z, // permeability_z subvector data
    double* z_mult, // z coordinate multiplier
    int ipatch,   // current patch
    int isubgrid) // current subgrid
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra = 
      (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int i = 0, j = 0, k = 0, ival = 0;
  ForPatchCellsPerFace(
    GroundwaterFlowBC,
    BeforeAllCells(DoNothing), 
    LoopVars(i, j, k, ival, bc_struct, ipatch, isubgrid),
    Locals(
      double *Sy = instance_xtra->SpecificYield[ipatch][isubgrid];
      double *Ad = instance_xtra->AquiferDepth[ipatch][isubgrid];
      double *Ar = instance_xtra->AquiferRecharge[ipatch][isubgrid];

      double dx = 0.0, dy = 0.0, dz = 0.0;
      double dvol = 0.0, dxdy = 0.0, dxdz = 0.0, dydz = 0.0;

      int stride_xp = 1;
      int stride_yp = SubvectorNX(p_sub);
      int stride_zp = SubvectorNY(p_sub) * stride_yp;

      int ip = 0;
      int mid = 0; // middle cell index
      int lft = -stride_xp; // left  cell index
      int rgt = +stride_xp; // right cell index
      int dwn = -stride_yp; // down  cell index
      int top = +stride_yp; // top   cell index
      int bck = -stride_zp; // back  cell index
      int frt = +stride_zp; // front cell index

      double area = 0.0;

      double q_storage = 0.0, q_divergence = 0.0, q_recharge = 0.0;

      double T_lft = 0.0, T_rgt = 0.0, T_dwn = 0.0;
      double T_top = 0.0, T_bck = 0.0, T_frt = 0.0;

      double old_head_mid = 0.0, new_head_mid = 0.0;
      double new_head_lft = 0.0, new_head_rgt = 0.0, new_head_dwn = 0.0;
      double new_head_top = 0.0, new_head_bck = 0.0, new_head_frt = 0.0;

      double dh_lft = 0.0, dh_rgt = 0.0, dh_dwn = 0.0;
      double dh_top = 0.0, dh_bck = 0.0, dh_frt = 0.0;
      double dh_dt  = 0.0;
    ),
    CellSetup({

      ip = SubvectorEltIndex(p_sub, i, j, k);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid) * z_mult[ip];

      dvol = dx*dy*dz*z_mult[ip]; // infinitesimal volume
      dxdy = dx*dy;
      dxdz = dx*dz*z_mult[ip];
      dydz = dy*dz*z_mult[ip];

      q_storage = 0.0;
      q_divergence = 0.0;
      q_recharge = 0.0;

      // compute transmissivity at the cell faces
      // [CAUTION] Ad index is ival, while perm index is ip; how to solve this?
      // double T_lft = HarmonicMean(
      //     Ad[ival+lft] * perm_x[ip+lft], Ad[ival+mid] * perm_x[ip+mid]);
      // double T_rgt = HarmonicMean(
      //     Ad[ival+mid] * perm_x[ip+mid], Ad[ival+rgt] * perm_x[ip+rgt]);
      // double T_dwn = HarmonicMean(
      //     Ad[ival+dwn] * perm_y[ip+dwn], Ad[ival+mid] * perm_y[ip+mid]);
      // double T_top = HarmonicMean(
      //     Ad[ival+mid] * perm_y[ip+mid], Ad[ival+top] * perm_y[ip+top]);
      T_lft = Ad[ival] * HarmonicMean(perm_x[ip+lft], perm_x[ip+mid]);
      T_rgt = Ad[ival] * HarmonicMean(perm_x[ip+mid], perm_x[ip+rgt]);
      T_dwn = Ad[ival] * HarmonicMean(perm_y[ip+dwn], perm_y[ip+mid]);
      T_top = Ad[ival] * HarmonicMean(perm_y[ip+mid], perm_y[ip+top]);
      // T_bck = Ad[ival] * HarmonicMean(perm_z[ip+bck], perm_z[ip+mid]);
      // T_frt = Ad[ival] * HarmonicMean(perm_z[ip+mid], perm_z[ip+frt]);

      // compute pressure head in adjacent cells
      // [CAUTION] Ad index is ival, while perm index is ip; how to solve this?
      // double hh_mid = pp[ip+mid] + 0.5 * Ad[ival+mid];
      // double hh_lft = pp[ip+lft] + 0.5 * Ad[ival+lft];
      // double hh_rgt = pp[ip+rgt] + 0.5 * Ad[ival+rgt];
      // double hh_dwn = pp[ip+dwn] + 0.5 * Ad[ival+dwn];
      // double hh_top = pp[ip+top] + 0.5 * Ad[ival+top];
      old_head_mid = old_pressure[ip+mid] + 0.5 * Ad[ival];
      new_head_mid = new_pressure[ip+mid] + 0.5 * Ad[ival];
      new_head_lft = new_pressure[ip+lft] + 0.5 * Ad[ival];
      new_head_rgt = new_pressure[ip+rgt] + 0.5 * Ad[ival];
      new_head_dwn = new_pressure[ip+dwn] + 0.5 * Ad[ival];
      new_head_top = new_pressure[ip+top] + 0.5 * Ad[ival];
      // new_head_bck = new_pressure[ip+bck] + 0.5 * Ad[ival];
      // new_head_frt = new_pressure[ip+frt] + 0.5 * Ad[ival];

      // compute difference in pressure head
      dh_lft = new_head_mid - new_head_lft;
      dh_rgt = new_head_rgt - new_head_mid;
      dh_dwn = new_head_mid - new_head_dwn;
      dh_top = new_head_top - new_head_mid;
      // dh_bck = new_head_mid - new_head_bck;
      // dh_frt = new_head_frt - new_head_mid;
      dh_dt  = new_head_mid - old_head_mid;
    }),
    FACE(LeftFace,  { area = +dydz; 
      // q_storage = dydz * Sy[ival] * dh_dt;
      // q_divergence = dt * dy * (T_frt * dh_frt - T_bck * dh_bck)
      //     + dt * dz * (T_top * dh_top - T_dwn * dh_dwn);
    }),
    FACE(RightFace, { area = -dydz; 
      // q_storage = dydz * Sy[ival] * dh_dt;
      // q_divergence = dt * dy * (T_frt * dh_frt - T_bck * dh_bck)
      //     + dt * dz * (T_top * dh_top - T_dwn * dh_dwn);
    }),
    FACE(DownFace,  { area = +dxdz; 
      // q_storage = dxdz * Sy[ival] * dh_dt;
      // q_divergence = dt * dz * (T_rgt * dh_rgt - T_lft * dh_lft)
      //     + dt * dx * (T_frt * dh_frt - T_bck * dh_bck);
    }),
    FACE(UpFace,    { area = -dxdz; 
      // q_storage = dxdz * Sy[ival] * dh_dt;
      // q_divergence = dt * dz * (T_rgt * dh_rgt - T_lft * dh_lft)
      //     + dt * dx * (T_frt * dh_frt - T_bck * dh_bck);
    }),
    FACE(BackFace,  { area = +dxdy;
      q_storage = dxdy * Sy[ival] * dh_dt;
      q_divergence = dt * dy * (T_rgt * dh_rgt - T_lft * dh_lft)
          + dt * dx * (T_top * dh_top - T_dwn * dh_dwn);
    }),
    FACE(FrontFace, { area = -dxdy; 
      // q_storage = dxdy * Sy[ival] * dh_dt;
      // q_divergence = dt * dy * (T_rgt * dh_rgt - T_lft * dh_lft)
      //     + dt * dx * (T_top * dh_top - T_dwn * dh_dwn);
    }),
    CellFinalize(
    {
      q_recharge = dt * area * Ar[ival];
      q_groundwater[ip] = q_storage - q_divergence + q_recharge;
    }),
    AfterAllCells(DoNothing)
  );    /* End GroundwaterFlow case */

  return;
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule* GroundwaterFlowEvalInitInstanceXtra(int num_patches, int subgrid_size)
{
  PFModule* this_module = ThisPFModule;

  double*** Sy = talloc(double**, num_patches);
  double*** Ad = talloc(double**, num_patches);
  double*** Ar = talloc(double**, num_patches);
  memset(Sy, 0, num_patches * sizeof(double **));
  memset(Ad, 0, num_patches * sizeof(double **));
  memset(Ar, 0, num_patches * sizeof(double **));

  for(int ipatch = 0; ipatch < num_patches; ++ipatch) {
    Sy[ipatch] = talloc(double*, subgrid_size);
    Ad[ipatch] = talloc(double*, subgrid_size);
    Ar[ipatch] = talloc(double*, subgrid_size);
    memset(Sy[ipatch], 0, subgrid_size * sizeof(double *));
    memset(Ad[ipatch], 0, subgrid_size * sizeof(double *));
    memset(Ar[ipatch], 0, subgrid_size * sizeof(double *));
  }

  InstanceXtra* instance_xtra = ctalloc(InstanceXtra, 1);
  instance_xtra->SpecificYield   = Sy;
  instance_xtra->AquiferDepth    = Ad;
  instance_xtra->AquiferRecharge = Ar;
  instance_xtra->num_patches = num_patches;
  instance_xtra->subgrid_size = subgrid_size;

  PFModuleInstanceXtra(this_module) = (void*)instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void GroundwaterFlowEvalFreeInstanceXtra()
{
  PFModule* this_module = ThisPFModule;
  InstanceXtra* instance_xtra = 
      (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if(!instance_xtra) return; 
  
  for(int ipatch = 0; ipatch < instance_xtra->num_patches; ++ipatch) {
    tfree(instance_xtra->SpecificYield[ipatch]);
    tfree(instance_xtra->AquiferDepth[ipatch]);
    tfree(instance_xtra->AquiferRecharge[ipatch]);
  }
  tfree(instance_xtra->SpecificYield);
  tfree(instance_xtra->AquiferDepth);
  tfree(instance_xtra->AquiferRecharge);

  tfree(instance_xtra);
  return;
}

/*--------------------------------------------------------------------------
 * GroundwaterFlowEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule* GroundwaterFlowEvalNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * GroundwaterFlowEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void GroundwaterFlowEvalFreePublicXtra()
{
  PFModule* this_module = ThisPFModule;
  PublicXtra* public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if(public_xtra) {
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
