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
  int num_phase
  int      *type;
  void    **data;
} PublicXtra;

typedef struct {
  PFModule *phase_density;

  Grid     *grid;

  double   *temp_data;

  Vector   *temp_new_density;
  Vector   *temp_new_density_der;
  Vector   *temp_fcn;
} InstanceXtra;

typedef struct {
  int num_regions;
  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
  char    *filename;

  /*Vector  *ic_values;*/
  Vector  *s_values;
} Type1;                      /* Spatially varying field over entire domain
                               * read from a file */



/*--------------------------------------------------------------------------
 *    Phase
 *--------------------------------------------------------------------------*/

void         PhaseSource(phase_source, phase, problem, problem_data, time)

Vector * phase _source;
int phase;
ProblemData *problem_data; /* Contains geometry information for the problem */
Problem     *problem;      /* General problem information */
double time;

{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  PFModule      *phase_density = (instance_xtra->phase_density);

  Grid          *grid = VectorGrid(ic_pressure);

  GrGeomSolid   *gr_solid, *gr_domain;

  Type0         *dummy0;
  Type1         *dummy1;
  Type2         *dummy2;
  Type3         *dummy3;

  SubgridArray  *subgrids = GridSubgrids(grid);

  Subgrid       *subgrid;

  Vector        *temp_new_density = (instance_xtra->temp_new_density);
  Vector        *temp_new_density_der =
    (instance_xtra->temp_new_density_der);
  Vector        *temp_fcn = (instance_xtra->temp_fcn);

  Subvector     *ps_sub;
  Subvector     *tf_sub;
  Subvector     *tnd_sub;
  Subvector     *tndd_sub;
  Subvector     *ic_values_sub;

  double        *data;
  double        *fcn_data;
  double        *new_density_data;
  double        *new_density_der_data;
  double        *psdat, *ic_values_dat;

  double gravity = -ProblemGravity(problem);

  int num_regions;
  int           *region_indices;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  int is, i, j, k, ips, iel, ipicv;

  amps_Invoice result_invoice;

  /*-----------------------------------------------------------------------
   * Sources for this phase
   *-----------------------------------------------------------------------*/

  InitVector(phase_source, 0.0);

  switch ((public_xtra->type[phase]))
  {
    case 0: /* Assign constant values within regions. */
    {
      double   *value;
      int ir;

      dummy0 = (Type0*)(public_xtra->data[phase]);

      num_regions = (dummy0->num_regions);
      region_indices = (dummy0->region_indices);
      values = (dummy0->values);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          ps_sub = VectorSubvector(ic_pressure, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          data = SubvectorData(ps_sub);
          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ips = SubvectorEltIndex(ps_sub, i, j, k);
            data[ips] = values[ir];
          });
        }      /* End of subgrid loop */
      }        /* End of region loop */
      break;
    }          /* End of case 0 */

    case 1: /* ParFlow binary file with spatially varying source values */
    {
      Vector *s_values;

      dummy1 = (Type1*)(public_xtra->data[phase]);

      s_values = dummy1->s_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        ps_sub = VectorSubvector(s_pressure, is);
        s_values_sub = VectorSubvector(s_values, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        /* RDF: assume resolution is the same in all 3 directions */
        r = SubgridRX(subgrid);

        psdat = SubvectorData(ps_sub);
        s_values_dat = SubvectorData(s_values_sub);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(ps_sub, i, j, k);
          ipicv = SubvectorEltIndex(s_values_sub, i, j, k);

          psdat[ips] = s_values_dat[ipicv];
        });
      }        /* End subgrid loop */
      break;
    }          /* End case 1 */
  }            /* End of switch statement */
}


/*--------------------------------------------------------------------------
 * PhaseSourceInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseSourceInitInstanceXtra(problem, grid, temp_data)

Problem * problem;
Grid      *grid;
double    *temp_data;
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  Type1         *dummy1;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      FreeTempVector(instance_xtra->temp_new_density);
      FreeTempVector(instance_xtra->temp_new_density_der);
      FreeTempVector(instance_xtra->temp_fcn);
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    (instance_xtra->temp_new_density) = NewTempVector(grid, 1, 1);
    (instance_xtra->temp_new_density_der) = NewTempVector(grid, 1, 1);
    (instance_xtra->temp_fcn) = NewTempVector(grid, 1, 1);

    /* Uses a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);
      (dummy1->s_values) = NewTempVector(grid, 1, 1);
    }
  }

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
    SetTempVectorData((instance_xtra->temp_new_density), temp_data);
    temp_data += SizeOfVector(instance_xtra->temp_new_density);
    SetTempVectorData((instance_xtra->temp_new_density_der), temp_data);
    temp_data += SizeOfVector(instance_xtra->temp_new_density_der);
    SetTempVectorData((instance_xtra->temp_fcn), temp_data);
    temp_data += SizeOfVector(instance_xtra->temp_fcn);

    /* Uses a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data[phase]);
      SetTempVectorData((dummy1->s_values), temp_data);
      temp_data += SizeOfVector(dummy1->s_values);

      ReadPFBinary((dummy1->filename),
                   (dummy1->s_values));
    }
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->phase_density) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->phase_density), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseSourceFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  PhaseSourceFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->phase_density);

    FreeTempVector(instance_xtra->temp_new_density);
    FreeTempVector(instance_xtra->temp_new_density_der);
    FreeTempVector(instance_xtra->temp_fcn);

    free(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * PhaseSourceNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PhaseSourceNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  int num_regions;
  int ir;
  int i;

  Type0         *dummy0;
  Type1         *dummy1;

  char *switch_name;
  char *region;

  char key[IDB_MAX_KEY_LEN];

  NameArray type_na;

  type_na = NA_NewNameArray(
                            "Constant HydroStaticDepth HydroStaticPatch PFBFile");

  public_xtra = ctalloc(PublicXtra, 1);

  switch_name = GetString("ICPressure.Type");

  public_xtra->type = NA_NameToIndexExitOnError(type_na, switch_name, "ICPressure.Type");

  switch_name = GetString("ICPressure.GeomNames");
  public_xtra->regions = NA_NewNameArray(switch_name);

  num_regions = NA_Sizeof(public_xtra->regions);

  switch ((public_xtra->type))
  {
    case 0:
    {
      dummy0 = ctalloc(Type0, 1);

      dummy0->num_regions = num_regions;

      (dummy0->region_indices) = ctalloc(int, num_regions);
      (dummy0->values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy0->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        sprintf(key, "Geom.%s.ICPressure.Value", region);
        dummy0->values[ir] = GetDouble(key);
      }

      (public_xtra->data) = (void*)dummy0;
      break;
    }

    case 1:
    {
      dummy1 = ctalloc(Type1, 1);

      dummy1->num_regions = num_regions;

      (dummy1->region_indices) = ctalloc(int, num_regions);
      (dummy1->reference_elevations) = ctalloc(double, num_regions);
      (dummy1->pressure_values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy1->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        sprintf(key, "Geom.%s.ICPressure.RefElevation", region);
        dummy1->reference_elevations[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.ICPressure.Value", region);
        dummy1->pressure_values[ir] = GetDouble(key);
      }

      (public_xtra->data) = (void*)dummy1;

      break;
    }

    case 2:
    {
      dummy2 = ctalloc(Type2, 1);

      dummy2->num_regions = num_regions;

      (dummy2->region_indices) = ctalloc(int, num_regions);
      (dummy2->geom_indices) = ctalloc(int, num_regions);
      (dummy2->patch_indices) = ctalloc(int, num_regions);
      (dummy2->pressure_values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy2->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        sprintf(key, "Geom.%s.ICPressure.Value", region);
        dummy2->pressure_values[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.ICPressure.RefGeom", region);
        switch_name = GetString(key);

        dummy2->geom_indices[ir] = NA_NameToIndexExitOnError(GlobalsGeomNames,
							     switch_name, key);

        sprintf(key, "Geom.%s.ICPressure.RefPatch", region);
        switch_name = GetString(key);

        dummy2->patch_indices[ir] =
          NA_NameToIndexExitOnError(GeomSolidPatches(
                                          GlobalsGeometries[dummy2->geom_indices[ir]]),
				    switch_name, key);
      }

      (public_xtra->data) = (void*)dummy2;

      break;
    }

    case 3:
    {
      dummy3 = ctalloc(Type3, 1);

      sprintf(key, "Geom.%s.ICPressure.FileName", "domain");
      dummy3->filename = GetString(key);

      public_xtra->data = (void*)dummy3;

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);    }
  }

  NA_FreeNameArray(type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * ICPhasePressureFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ICPhasePressureFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  Type0       *dummy0;
  Type1       *dummy1;
  Type2       *dummy2;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->regions);

    switch ((public_xtra->type))
    {
      case 0:
      {
        dummy0 = (Type0*)(public_xtra->data);

        tfree(dummy0->region_indices);
        tfree(dummy0->values);
        tfree(dummy0);
        break;
      }

      case 1:
      {
        dummy1 = (Type1*)(public_xtra->data);

        tfree(dummy1->region_indices);
        tfree(dummy1->reference_elevations);
        tfree(dummy1->pressure_values);
        tfree(dummy1);
      }

      case 2:
      {
        dummy2 = (Type2*)(public_xtra->data);

        tfree(dummy2->region_indices);
        tfree(dummy2->patch_indices);
        tfree(dummy2->geom_indices);
        tfree(dummy2->pressure_values);
        tfree(dummy2);
      }
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * ICPhasePressureSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ICPhasePressureSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type3         *dummy3;

  int sz = 0;

  /* add local TempData size to `sz' */
  sz += SizeOfVector(instance_xtra->temp_new_density);
  sz += SizeOfVector(instance_xtra->temp_new_density_der);
  sz += SizeOfVector(instance_xtra->temp_fcn);

  if (public_xtra->type == 3)
  {
    dummy3 = (Type3*)(public_xtra->data);
    sz += SizeOfVector(dummy3->ic_values);
  }

  return sz;
}
