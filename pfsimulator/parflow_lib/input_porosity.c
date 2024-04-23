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
/* helper type for reading from file */
typedef struct {
  char    *filename;
  Vector  *ic_values;
} Type3;                      /* Spatially varying field over entire domain
                               * read from a file */
/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/
typedef struct {
  double field_value;
  /* new field for reading from file */
  void *data;
} PublicXtra;
typedef struct {
  Grid   *grid;
  double *temp_data;
} InstanceXtra;
/*--------------------------------------------------------------------------
 * InputPorosity
 *--------------------------------------------------------------------------*/
void    InputPorosity(
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
  /* extra variables for reading from file */
  Type3 * dummy3;
  dummy3 = (Type3*)(public_xtra->data);
  Vector *ic_values = dummy3->ic_values;
  Subvector *ic_values_sub;
  double  *ic_values_dat;
  for (subgrid_loop = 0; subgrid_loop < GridNumSubgrids(grid); subgrid_loop++)
  {
    subgrid = GridSubgrid(grid, subgrid_loop);
    field_sub = VectorSubvector(field, subgrid_loop);

    /* new subvector from file */
    ic_values_sub = VectorSubvector(ic_values, subgrid_loop);
    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);
    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);
    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);
    fieldp = SubvectorData(field_sub);
    /* new subvector data to read from */
    ic_values_dat = SubvectorData(ic_values_sub);
    GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
    {
      index = SubvectorEltIndex(field_sub, i, j, k);
      fieldp[index] = ic_values_dat[index];
    });
  }
}
/*--------------------------------------------------------------------------
 * InputPorosityInitInstanceXtra
 *--------------------------------------------------------------------------*/
PFModule  *InputPorosityInitInstanceXtra(
                                         Grid *  grid,
                                         double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

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
    /* read from file */
    Type3 * dummy3;
    dummy3 = (Type3*)(public_xtra->data);
    /* Allocate temp vector */
    dummy3->ic_values = NewVectorType(grid, 1, 1, vector_cell_centered);
    ReadPFBinary((dummy3->filename), (dummy3->ic_values));
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
 * InputPorosityFreeInstanceXtra
 *--------------------------------------------------------------------------*/
void  InputPorosityFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (instance_xtra)
  {
    /* free extra data from file */
    Type3 *dummy3;
    dummy3 = (Type3*)(public_xtra->data);
    FreeVector(dummy3->ic_values);
    tfree(instance_xtra);
  }
}
/*--------------------------------------------------------------------------
 * InputPorosityNewPublicXtra
 *--------------------------------------------------------------------------*/
PFModule   *InputPorosityNewPublicXtra(char *geom_name)
{
  /* Local variables */
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;
  char key[IDB_MAX_KEY_LEN];

  /* Allocate space for the public_xtra structure */
  /* modification for loading pdb filename */
  /* sprintf(key, "Geom.%s.Porosity.Value", geom_name); */
  /* public_xtra -> field_value = GetDouble(key); */
  public_xtra = ctalloc(PublicXtra, 1);
  Type3 *dummy3;
  dummy3 = ctalloc(Type3, 1);
  sprintf(key, "Geom.%s.Porosity.FileName", geom_name);
  dummy3->filename = GetString(key);
  public_xtra->data = (void*)dummy3;
  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}
/*--------------------------------------------------------------------------
 * InputPorosityFreePublicXtra
 *--------------------------------------------------------------------------*/
void  InputPorosityFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    /* free extra fields for file */
    Type3 * dummy3;
    dummy3 = (Type3*)(public_xtra->data);
    tfree(dummy3);
    tfree(public_xtra);
  }
}
/*--------------------------------------------------------------------------
 * InputPorositySizeOfTempData
 *--------------------------------------------------------------------------*/
int  InputPorositySizeOfTempData()
{
  return 0;
}
