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
#include "parflow_netcdf.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int type;
  void  *data;
  int wcy_exists;              /* check to see whether or not channel width Y is used */
} PublicXtra;

typedef struct {
  Grid   *grid3d;
  Grid   *grid2d;
} InstanceXtra;

typedef struct {
  NameArray regions;
  int num_regions;
  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
  char  *filename;
  Vector *wcy_values;
} Type1;                       /* .pfb file */

typedef struct {
  char  *filename;
  Vector *wcy_values;
} Type2;                       /* .nc file */


/** @brief Populates region data for channel width in Y direction.
 *
 * @param problem_data pointer to ProblemData structure
 * @param wc_y channel width data (in the y direction)
 * @param dummy dummy vector
 *
 */
void YChannelWidth(ProblemData *problem_data, Vector *wc_y, Vector *dummy)
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra *)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);

  Grid *grid_3d = instance_xtra->grid3d;

  GrGeomSolid *gr_solid, *gr_domain;

  Type0 *dummy0;
  Type1 *dummy1;
  Type2 *dummy2;

  VectorUpdateCommHandle *handle;

  SubgridArray *subgrids = GridSubgrids(grid_3d);

  Subgrid *subgrid;
  Subvector *ps_sub;
  Subvector *wcy_values_sub;
  double *psdat, *wcy_values_dat;

  double *data;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  int is, i, j, k, ips, ipicv;

  (void)dummy;

  InitVectorAll(wc_y, 0.0);

  if (public_xtra->wcy_exists == 1)
  {
    switch ((public_xtra->type))
    {
      case 0: {
        int num_regions;
        int *region_indices;
        double *values;
        double value;
        int ir;

        dummy0 = (Type0 *)public_xtra->data;

        num_regions = dummy0->num_regions;
        region_indices = dummy0->region_indices;
        values = dummy0->values;

        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
          value = values[ir];

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);
            ps_sub = VectorSubvector(wc_y, is);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            r = SubgridRX(subgrid);

            data = SubvectorData(ps_sub);

            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps_sub, i, j, 0);
              data[ips] = value;
            });
          }
        }
        break;
      }

      case 1: {
        Vector *wcy_val;

        dummy1 = (Type1 *)public_xtra->data;

        wcy_val = dummy1->wcy_values;

        gr_domain = ProblemDataGrDomain(problem_data);

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          ps_sub = VectorSubvector(wc_y, is);
          wcy_values_sub = VectorSubvector(wcy_val, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          psdat = SubvectorData(ps_sub);
          wcy_values_dat = SubvectorData(wcy_values_sub);

          GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz, {
            ips = SubvectorEltIndex(ps_sub, i, j, 0);
            ipicv = SubvectorEltIndex(wcy_values_sub, i, j, 0);

            psdat[ips] = wcy_values_dat[ipicv];
          })
        }
        break;
      }

      case 2: {
        Vector *wcy_val;

        dummy2 = (Type2 *)public_xtra->data;

        wcy_val = dummy2->wcy_values;

        gr_domain = ProblemDataGrDomain(problem_data);

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          ps_sub = VectorSubvector(wc_y, is);
          wcy_values_sub = VectorSubvector(wcy_val, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          psdat = SubvectorData(ps_sub);
          wcy_values_dat = SubvectorData(wcy_values_sub);

          GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz, {
            ips = SubvectorEltIndex(ps_sub, i, j, 0);
            ipicv = SubvectorEltIndex(wcy_values_sub, i, j, 0);

            psdat[ips] = wcy_values_dat[ipicv];
          });
        }
        break;
      }
    }
  }

  handle = InitVectorUpdate(wc_y, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
}


/** @brief Initializes InstanceXtra object for channel width Y problem
 *
 * @return Modified PFModule with YChannelWidth information
 */
PFModule *YChannelWidthInitInstanceXtra(Grid *grid3d, Grid *grid2d)
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra *)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra;

  Type1 *dummy1;
  Type2 *dummy2;

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    instance_xtra = ctalloc(InstanceXtra, 1);
  }
  else
  {
    instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
  }

  if (grid3d != NULL)
  {
    (instance_xtra->grid3d) = grid3d;
  }

  if (grid2d != NULL)
  {
    (instance_xtra->grid2d) = grid2d;

    if (public_xtra->wcy_exists)
    {
      if (public_xtra->type == 1)
      {
        dummy1 = (Type1 *)(public_xtra->data);

        dummy1->wcy_values = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);

        ReadPFBinary((dummy1->filename), (dummy1->wcy_values));
      }

      if (public_xtra->type == 2)
      {
        dummy2 = (Type2 *)(public_xtra->data);

        dummy2->wcy_values = NewVectorType(grid2d, 1, 1, vector_cell_centered_2D);

        ReadPFNC((dummy2->filename), (dummy2->wcy_values), "wc_y", 0, 2);
      }
    }
  }


  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/** @brief Frees the InstanceXtra object for channel width Y problem
 *
 */
void YChannelWidthFreeInstanceXtra()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra *)PFModulePublicXtra(this_module);
  InstanceXtra *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);

  Type1 *dummy1;
  Type2 *dummy2;

  if (public_xtra->wcy_exists == 1)
  {
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1 *)(public_xtra->data);
      FreeVector(dummy1->wcy_values);
    }

    if (public_xtra->type == 2)
    {
      dummy2 = (Type2 *)(public_xtra->data);
      FreeVector(dummy2->wcy_values);
    }
  }

  if (instance_xtra)
  {
    free(instance_xtra);
  }
}

/** @brief Creates PublicXtra object for channel width
 *
 */
PFModule *YChannelWidthNewPublicXtra()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra;

  Type0 *dummy0;
  Type1 *dummy1;
  Type2 *dummy2;

  int num_regions, ir;

  char key[IDB_MAX_KEY_LEN];

  char *name;

  char *switch_name;
  char *switch_exist_name;

  NameArray type_na;
  NameArray switch_na;
  int switch_val;

  public_xtra = ctalloc(PublicXtra, 1);

  name = "Solver.Nonlinear.ChannelWidthExistY";
  switch_na = NA_NewNameArray("False True");
  switch_exist_name = GetStringDefault(name, "False");
  switch_val = NA_NameToIndexExitOnError(switch_na, switch_exist_name, name);
  NA_FreeNameArray(switch_na);

  public_xtra->wcy_exists = switch_val;

  if (public_xtra->wcy_exists == 1)
  {
    type_na = NA_NewNameArray("Constant PFBFile NCFile");

    switch_name = GetString("ChannelWidthY.Type");

    public_xtra->type = NA_NameToIndexExitOnError(type_na, switch_name, "ChannelWidthY.Type");

    switch ((public_xtra->type))
    {
      case 0: {
        dummy0 = ctalloc(Type0, 1);

        switch_name = GetString("ChannelWidthY.GeomNames");

        dummy0->regions = NA_NewNameArray(switch_name);

        num_regions = (dummy0->num_regions) = NA_Sizeof(dummy0->regions);

        (dummy0->region_indices) = ctalloc(int, num_regions);
        (dummy0->values) = ctalloc(double, num_regions);

        for (ir = 0; ir < num_regions; ir++)
        {
          (dummy0->region_indices)[ir] = NA_NameToIndex(GlobalsGeomNames, NA_IndexToName((dummy0->regions), ir));
          sprintf(key, "ChannelWidthY.Geom.%s.Value", NA_IndexToName((dummy0->regions), ir));
          (dummy0->values)[ir] = GetDouble(key);
        }

        (public_xtra->data) = (void *)dummy0;

        break;
      }

      case 1: {
        dummy1 = ctalloc(Type1, 1);

        dummy1->filename = GetString("ChannelWidthY.FileName");

        (public_xtra->data) = (void *)dummy1;
        break;
      }

      case 2: {
        dummy2 = ctalloc(Type2, 1);

        dummy2->filename = GetString("ChannelWidthY.FileName");

        (public_xtra->data) = (void *)dummy2;
        break;
      }

      default: {
        InputError("Error: invalid type <%s> for key <%s>\n", switch_name, key);
      }
    }

    NA_FreeNameArray(type_na);
  }

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/** @brief Frees the PublicXtra object associated with YChannelWidth.
 *
 */
void YChannelWidthFreePublicXtra()
{
  PFModule *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra *)PFModulePublicXtra(this_module);

  Type0 *dummy0;

  if (public_xtra)
  {
    if (public_xtra->wcy_exists)
    {
      switch ((public_xtra->type))
      {
        case 0: {
          dummy0 = (Type0 *)(public_xtra->data);

          NA_FreeNameArray(dummy0->regions);

          tfree(dummy0->region_indices);
          tfree(dummy0->values);
          tfree(dummy0);
          break;
        }

        case 1: {
          Type1 *dummy1;
          dummy1 = (Type1 *)(public_xtra->data);

          tfree(dummy1);
          break;
        }

        case 2: {
          Type2 *dummy2;
          dummy2 = (Type2 *)(public_xtra->data);

          tfree(dummy2);
          break;
        }
      }
    }

    tfree(public_xtra);
  }
}


int YChannelWidthSizeOfTempData()
{
  return 0;
}
