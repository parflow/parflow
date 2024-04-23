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
/*****************************************************************************
*
*****************************************************************************/

#include "parflow.h"

#include <float.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  NameArray geo_names;
  int num_geo_indexes;
  int       *geo_indexes;       /* Integer value of each geounit */

  PFModule **PorosityFieldSimulators;

  int time_index;
} PublicXtra;

typedef struct {
  Grid      *grid;

  PFModule **PorosityFieldSimulators;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * Porosity
 *--------------------------------------------------------------------------*/

void Porosity(
              ProblemData * problem_data,
              Vector *      porosity,
              int           num_geounits,
              GeomSolid **  geounits,
              GrGeomSolid **gr_geounits)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int num_geo_indexes = (public_xtra->num_geo_indexes);
  int              *geo_indexes = (public_xtra->geo_indexes);

  PFModule        **PorosityFieldSimulators = (instance_xtra->PorosityFieldSimulators);

  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;

  Grid             *grid;

  SubgridArray     *subgrids;

  Subgrid          *subgrid,
    *well_subgrid,
    *tmp_subgrid;

  Subvector        *subvector;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_p, ny_p, nz_p;
  int i, j, k, pi, sg, well;

  double           *phi;

  (void)num_geounits;

  BeginTiming(public_xtra->time_index);

  InitVectorAll(porosity, 1.0);

  /*------------------------------------------------------------------------
   * Compute porosity in the geounits.
   *------------------------------------------------------------------------*/

  for (i = 0; i < num_geo_indexes; i++)
  {
    j = geo_indexes[i];
    PFModuleInvokeType(PorosityFieldInvoke, PorosityFieldSimulators[i], (geounits[j], gr_geounits[j], porosity));
  }

  /*------------------------------------------------------------------------
   * Reset porosity in wells to 1.
   *------------------------------------------------------------------------*/

  if (WellDataNumWells(well_data) > 0)
  {
    grid = VectorGrid(porosity);

    subgrids = GridSubgrids(grid);

    for (well = 0; well < WellDataNumPressWells(well_data); well++)
    {
      well_data_physical = WellDataPressWellPhysical(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(porosity, sg);

        nx_p = SubvectorNX(subvector);
        ny_p = SubvectorNY(subvector);
        nz_p = SubvectorNZ(subvector);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          phi = SubvectorElt(subvector, ix, iy, iz);

          pi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    pi, nx_p, ny_p, nz_p, 1, 1, 1,
          {
            phi[pi] = 1.0;
          });

          FreeSubgrid(tmp_subgrid);       /* done with temporary subgrid */
        }
      }
    }

    for (well = 0; well < WellDataNumFluxWells(well_data); well++)
    {
      well_data_physical = WellDataFluxWellPhysical(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(porosity, sg);

        nx_p = SubvectorNX(subvector);
        ny_p = SubvectorNY(subvector);
        nz_p = SubvectorNZ(subvector);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          phi = SubvectorElt(subvector, ix, iy, iz);

          pi = 0;
          BoxLoopI1(i, j, k,
                    ix, iy, iz, nx, ny, nz,
                    pi, nx_p, ny_p, nz_p, 1, 1, 1,
          {
            phi[pi] = 1.0;
          });

          FreeSubgrid(tmp_subgrid);       /* done with temporary subgrid */
        }
      }
    }
  }

  EndTiming(public_xtra->time_index);

  return;
}


/*--------------------------------------------------------------------------
 * PorosityInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PorosityInitInstanceXtra(
                                    Grid *  grid,
                                    double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  int num_geo_indexes = (public_xtra->num_geo_indexes);
  int i;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
    (instance_xtra->grid) = grid;

  /*-----------------------------------------------------------------------
   * Initialize module instances
   *-----------------------------------------------------------------------*/

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->PorosityFieldSimulators) = talloc(PFModule *, num_geo_indexes);

    for (i = 0; i < num_geo_indexes; i++)
    {
      (instance_xtra->PorosityFieldSimulators)[i] =
        PFModuleNewInstanceType(PorosityFieldInitInstanceXtraInvoke, (public_xtra->PorosityFieldSimulators)[i], (grid, temp_data));
    }
  }
  else
  {
    for (i = 0; i < num_geo_indexes; i++)
    {
      PFModuleReNewInstanceType(PorosityFieldInitInstanceXtraInvoke, (instance_xtra->PorosityFieldSimulators)[i], (grid, temp_data));
    }
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PorosityFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PorosityFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int i;

  if (instance_xtra)
  {
    for (i = 0; i < (public_xtra->num_geo_indexes); i++)
    {
      PFModuleFreeInstance(instance_xtra->PorosityFieldSimulators[i]);
    }
    tfree(instance_xtra->PorosityFieldSimulators);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PorosityNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PorosityNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  int i, ind, sim_type;
  int num_geo_indexes;

  char          *sim_type_name;
  char key[IDB_MAX_KEY_LEN];

  char *geom_names;
  char *geom_name;

  NameArray switch_na;

  /*----------------------------------------------------------
   * The name array to map names to switch values
   *----------------------------------------------------------*/
  switch_na = NA_NewNameArray("Constant PFBFile");

  public_xtra = ctalloc(PublicXtra, 1);

  /*--------------------------------------------------------------
   * Read in the number of geounits to simulate (num_geo_indexes),
   * and assign properties to each
   *--------------------------------------------------------------*/

  geom_names = GetString("Geom.Porosity.GeomNames");

  public_xtra->geo_names = NA_NewNameArray(geom_names);

  num_geo_indexes = NA_Sizeof(public_xtra->geo_names);

  (public_xtra->num_geo_indexes) = num_geo_indexes;
  (public_xtra->geo_indexes) = ctalloc(int, num_geo_indexes);
  (public_xtra->PorosityFieldSimulators) = ctalloc(PFModule *,
                                                   num_geo_indexes);

  for (i = 0; i < num_geo_indexes; i++)
  {
    geom_name = NA_IndexToName(public_xtra->geo_names, i);
    ind = NA_NameToIndex(GlobalsGeomNames, geom_name);
    if (ind < 0)
    {
      InputError("Error: invalid geometry name <%s> for porosity\n",
                 geom_name, "");
    }
    (public_xtra->geo_indexes)[i] = ind;

    sprintf(key, "Geom.%s.Porosity.Type", geom_name);
    sim_type_name = GetString(key);

    sim_type = NA_NameToIndexExitOnError(switch_na, sim_type_name, key);

    /* Assign the Porosity field simulator method and invoke the "New"
     * function */
    switch (sim_type)
    {
      case 0:
      {
        (public_xtra->PorosityFieldSimulators)[i] = PFModuleNewModuleType(PorosityFieldNewPublicXtraInvoke,
                                                                          ConstantPorosity, (geom_name));
        break;
      }

      case 1:
      {
        (public_xtra->PorosityFieldSimulators)[i] = PFModuleNewModuleType(PorosityFieldNewPublicXtraInvoke,
                                                                          InputPorosity, (geom_name));
        break;
      }

      default:
      {
	InputError("Invalid switch value <%s> for key <%s>", sim_type_name, key);
      }
    }
  }

  (public_xtra->time_index) = RegisterTiming("Porosity");

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PorosityFreePublicXtra
 *--------------------------------------------------------------------------*/

void  PorosityFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  int i;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->geo_names);

    for (i = 0; i < (public_xtra->num_geo_indexes); i++)
    {
      PFModuleFreeModule(public_xtra->PorosityFieldSimulators[i]);
    }
    tfree(public_xtra->PorosityFieldSimulators);

    tfree(public_xtra->geo_indexes);
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * PorositySizeOfTempData
 *--------------------------------------------------------------------------*/

int  PorositySizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  PublicXtra    *instance_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  int sz = 0;

  int n;

  /* set `sz' to max of each of the called modules */
  for (n = 0; n < (public_xtra->num_geo_indexes); n++)
  {
    sz = pfmax(sz, PFModuleSizeOfTempData((instance_xtra->PorosityFieldSimulators)[n]));
  }

  /* add local TempData size to `sz' */
  /*  ---> This module doesn't use temp space */

  return sz;
}
