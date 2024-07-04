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

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  NameArray regions;
  int type;
  int FB_in_z;
  void       *data;
} PublicXtra;

typedef void InstanceXtra;


typedef struct {
  char       *filename;
  Vector     *values;
} Type0;                       /* from PFB */



/*--------------------------------------------------------------------------
 * FBz Scaling values
 *--------------------------------------------------------------------------*/
void FBz(ProblemData *problem_data, Vector *FBz)
{
  PFModule       *this_module = ThisPFModule;
  PublicXtra     *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Grid           *grid = VectorGrid(FBz);

  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *FBz_sub;
  Subvector      *val_sub;

  VectorUpdateCommHandle       *handle;

  double         *FBz_dat;
  double         *val_dat;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int is, i, j, k, ips, ipicv;

  /*-----------------------------------------------------------------------
   * FBz Scale
   *-----------------------------------------------------------------------*/

  InitVectorAll(FBz, 1.0);

  if (public_xtra->FB_in_z)
  {
    switch ((public_xtra->type))
    {
      // from PFB
      case 0:
      {
        Type0   *dummy0;
        dummy0 = (Type0*)(public_xtra->data);
        char        *filename = dummy0->filename;
        Vector      *values = dummy0->values;
        GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

        values = NewVectorType(grid, 1, 1, vector_cell_centered);
        ReadPFBinary(filename, values);

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          FBz_sub = VectorSubvector(FBz, is);
          val_sub = VectorSubvector(values, is);
          FBz_dat = SubvectorData(FBz_sub);
          val_dat = SubvectorData(val_sub);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
          {
            ips = SubvectorEltIndex(FBz_sub, i, j, k);
            ipicv = SubvectorEltIndex(val_sub, i, j, k);
            FBz_dat[ips] = val_dat[ipicv];
          });
        }      /* End subgrid loop */

        break;
      }
    }
  }
  handle = InitVectorUpdate(FBz, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
}


/*--------------------------------------------------------------------------
 * FBzInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *FBzInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

#if 0
  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
#endif
  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * FBzFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  FBzFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * FBzNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *FBzNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;

  char *switch_name;
  int switch_value;
  char key[IDB_MAX_KEY_LEN];
  char *name;
  NameArray switch_na;

  public_xtra = ctalloc(PublicXtra, 1);

  /* @RMM added switch for cell face flow barrier */
  /* RMM (default=False) */
  name = "Solver.Nonlinear.FlowBarrierZ";
  switch_na = NA_NewNameArray("False True");
  switch_name = GetStringDefault(name, "False");
  switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, name);
  NA_FreeNameArray(switch_na);

  public_xtra->FB_in_z = switch_value;
  if (public_xtra->FB_in_z == 1)
  {
    name = "FBz.Type";
    switch_na = NA_NewNameArray("PFBFile");
    switch_name = GetString(name);
    public_xtra->type = NA_NameToIndexExitOnError(switch_na, switch_name, name);
    NA_FreeNameArray(switch_na);

    // switch for FBz Type
    //  PFBFile = 0;
    switch ((public_xtra->type))
    {
      // Read from PFB file
      case 0:
      {
        dummy0 = ctalloc(Type0, 1);

        sprintf(key, "Geom.%s.FBz.FileName", "domain");
        dummy0->filename = GetString(key);
        public_xtra->data = (void*)dummy0;

        break;
      }
      default:
      {
	InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
      }
    }   
  }

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * FBzFreePublicXtra
 *--------------------------------------------------------------------------*/
void  FBzFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;

  if (public_xtra)
  {
    if (public_xtra->FB_in_z)
    {
      NA_FreeNameArray(public_xtra->regions);

      switch ((public_xtra->type))
      {
        case 0:
        {
          dummy0 = (Type0*)(public_xtra->data);
          tfree(dummy0);
          break;
        }
      }
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * FBzSizeOfTempData
 *--------------------------------------------------------------------------*/
int  FBzSizeOfTempData()
{
  return 0;
}
