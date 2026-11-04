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

#include "parflow.h"

#ifdef HAVE_ALQUIMIA

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_geochem_conds;
  int type;
  void      *data;
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  int num_regions;
  NameArray regions;
  int       *region_indices;
  double    *values;
} Type0;                       /* constant regions */

typedef struct {
  char  *filename;
} Type1;                      /* .pfb file */


/** @brief Fill the geochemical-condition index field: map each named initial
 * condition onto its geometries (later names overlay earlier ones), so every
 * cell knows which engine condition initializes it.
 *
 * @param problem_data the problem data
 * @param geochemcond [out] the per-cell geochemical-condition index field
 */
void  GeochemCond(ProblemData *problem_data, Vector *geochemcond)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Grid           *grid = VectorGrid(geochemcond);

  Type0          *dummy0;
  Type1          *dummy1;

  SubgridArray   *subgrids = GridSubgrids(grid);

  Subgrid        *subgrid;
  Subvector      *pc_sub;

  double *data;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int is, i, j, k, ipc;


  /*-----------------------------------------------------------------------
   * Initial conditions for concentrations in each phase
   *-----------------------------------------------------------------------*/
  InitVector(geochemcond, 0);

  switch ((public_xtra->type))
  {
    case 0:
    {
      int num_regions;
      int *region_indices;
      double *values;

      GrGeomSolid  *gr_solid;
      double value;
      int ir;


      dummy0 = (Type0*)(public_xtra->data);

      num_regions = (dummy0->num_regions);
      region_indices = (dummy0->region_indices);
      values = (dummy0->values);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
        value = values[ir];

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          pc_sub = VectorSubvector(geochemcond, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          data = SubvectorData(pc_sub);
          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ipc = SubvectorEltIndex(pc_sub, i, j, k);

            data[ipc] = value;
          });
        }
      }

      break;
    }

    case 1:
    {
      dummy1 = (Type1*)(public_xtra->data);

      ReadPFBinary((dummy1->filename), geochemcond);

      break;
    }
  }
}

/*--------------------------------------------------------------------------
 * GeochemCondInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *GeochemCondInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * GeochemCondFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  GeochemCondFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GeochemCondNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *GeochemCondNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;

  char key[IDB_MAX_KEY_LEN];
  char *switch_name;
  char *region;
  char *geochem_cond_names;
  int num_geochem_conds;

  NameArray type_na;
  NameArray geochem_cond_na;

  type_na = NA_NewNameArray("Constant PFBFile");

  geochem_cond_names = GetStringDefault("GeochemCondition.Names", "");
  geochem_cond_na = NA_NewNameArray(geochem_cond_names);
  num_geochem_conds = NA_Sizeof(geochem_cond_na);

  if (num_geochem_conds > 0)
  {
    public_xtra = ctalloc(PublicXtra, 1);
    switch_name = GetString("GeochemCondition.Type");
    public_xtra->type = NA_NameToIndex(type_na, switch_name);
    switch ((public_xtra->type))
    {
      case 0:
      {
        int num_regions, ir;

        dummy0 = ctalloc(Type0, 1);

        switch_name = GetString("GeochemCondition.GeomNames");

        dummy0->regions = NA_NewNameArray(switch_name);

        num_regions =
          (dummy0->num_regions) = NA_Sizeof(dummy0->regions);

        (dummy0->region_indices) = ctalloc(int, num_regions);
        (dummy0->values) = ctalloc(double, num_regions);

        for (ir = 0; ir < num_regions; ir++)
        {
          region = NA_IndexToName(dummy0->regions, ir);

          dummy0->region_indices[ir] =
            NA_NameToIndex(GlobalsGeomNames, region);

          sprintf(key, "GeochemCondition.Geom.%s.Value", region);
          dummy0->values[ir] = NA_NameToIndex(geochem_cond_na, GetString(key));
        }

        (public_xtra->data) = (void*)dummy0;

        break;
      }

      case 1:
      {
        dummy1 = ctalloc(Type1, 1);
        sprintf(key, "GeochemCondition.FileName");
        dummy1->filename = GetString(key);
        (public_xtra->data) = (void*)dummy1;

        break;
      }

      default:
      {
        InputError("Error: invalid type <%s> for key <%s>\n",
                   switch_name, key);
      }
    }
  }
  else
  {
    public_xtra = NULL;
  }

  NA_FreeNameArray(type_na);
  NA_FreeNameArray(geochem_cond_na);

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}

/*--------------------------------------------------------------------------
 * GeochemCondFreePublicXtra
 *--------------------------------------------------------------------------*/

void  GeochemCondFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;
  Type1       *dummy1;

  if (public_xtra)
  {
    switch ((public_xtra->type))
    {
      case 0:
      {
        dummy0 = (Type0*)(public_xtra->data);

        NA_FreeNameArray(dummy0->regions);

        tfree(dummy0->region_indices);
        tfree(dummy0->values);
        tfree(dummy0);
        break;
      }

      case 1:
      {
        dummy1 = (Type1*)(public_xtra->data);

        tfree(dummy1);
        break;
      }
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GeochemCondSizeOfTempData
 *--------------------------------------------------------------------------*/

int  GeochemCondSizeOfTempData()
{
  return 0;
}
#endif /* HAVE_ALQUIMIA */
