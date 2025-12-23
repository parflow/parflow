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
  double field_value;
} PublicXtra;

typedef struct {
  Grid   *grid;

  double *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * ConstantPorosity
 *--------------------------------------------------------------------------*/

void    ConstantPorosity(
                         GeomSolid *  geounit,
                         GrGeomSolid *gr_geounit,
                         Vector *     field)
{
  /*-----------------------------------------------------------------------
   * Local variables
   *-----------------------------------------------------------------------*/
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  double field_value = (public_xtra->field_value);

  Grid           *grid = (instance_xtra->grid);
  Subgrid        *subgrid;

  Subvector      *field_sub;
  double         *fieldp;

  int subgrid_loop;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int index;

  (void)geounit;

  /*-----------------------------------------------------------------------
   * Assign constant values to field
   *-----------------------------------------------------------------------*/

  for (subgrid_loop = 0; subgrid_loop < GridNumSubgrids(grid); subgrid_loop++)
  {
    subgrid = GridSubgrid(grid, subgrid_loop);
    field_sub = VectorSubvector(field, subgrid_loop);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    fieldp = SubvectorData(field_sub);
    GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
    {
      index = SubvectorEltIndex(field_sub, i, j, k);

      fieldp[index] = field_value;
    });
  }
}


/*--------------------------------------------------------------------------
 * ConstantPorosityInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *ConstantPorosityInitInstanceXtra(
                                            Grid *  grid,
                                            double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */

    /* set new data */
    (instance_xtra->grid) = grid;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * ConstantPorosityFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  ConstantPorosityFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * ConstantPorosityNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ConstantPorosityNewPublicXtra(char *geom_name)
{
  /* Local variables */
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  /* Allocate space for the public_xtra structure */
  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "Geom.%s.Porosity.Value", geom_name);
  public_xtra->field_value = GetDouble(key);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * ConstantPorosityFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ConstantPorosityFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * ConstantPorositySizeOfTempData
 *--------------------------------------------------------------------------*/

int  ConstantPorositySizeOfTempData()
{
  return 0;
}
