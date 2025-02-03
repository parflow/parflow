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
*                               Description
*-----------------------------------------------------------------------------
* This file contains a parflow module that will assign values to the
* field vector that is passed in. The values are specified in the
* user-supplied input file. In particular, the user must specify permeability
* data corresponding to the computational grid exactly.
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  char *filename;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid   *grid;
  double *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * InputRF
 *--------------------------------------------------------------------------*/

void    InputRF(
                GeomSolid *  geounit,
                GrGeomSolid *gr_geounit,
                Vector *     field,
                RFCondData * cdata)
{
  /*-----------------------------------------------------------------------
   * Local variables
   *-----------------------------------------------------------------------*/
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Vector    *tmpRF = NULL;

  /* Grid parameters */
  Grid           *grid = (instance_xtra->grid);
  Subgrid        *subgrid;

  Subvector      *field_sub;
  Subvector      *tmpRF_sub;
  double         *fieldp;
  double         *tmpRFp;

  /* Counter, indices, flags */
  int subgrid_loop;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int indexfp, indextp;

  (void)geounit;
  (void)cdata;

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  tmpRF = NewVectorType(grid, 1, 1, vector_cell_centered);

  ReadPFBinary((public_xtra->filename), tmpRF);

  /*-----------------------------------------------------------------------
   * Assign input data values to field
   *-----------------------------------------------------------------------*/

  for (subgrid_loop = 0; subgrid_loop < GridNumSubgrids(grid); subgrid_loop++)
  {
    subgrid = GridSubgrid(grid, subgrid_loop);
    field_sub = VectorSubvector(field, subgrid_loop);
    tmpRF_sub = VectorSubvector(tmpRF, subgrid_loop);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    fieldp = SubvectorData(field_sub);
    tmpRFp = SubvectorData(tmpRF_sub);
    GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
    {
      indexfp = SubvectorEltIndex(field_sub, i, j, k);
      indextp = SubvectorEltIndex(tmpRF_sub, i, j, k);

      fieldp[indexfp] = tmpRFp[indextp];
    });
  }

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(tmpRF);
}


/*--------------------------------------------------------------------------
 * InputRFInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *InputRFInitInstanceXtra(
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
 * InputRFFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  InputRFFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * InputRFNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *InputRFNewPublicXtra(char *geom_name)
{
  /* Local variables */
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char key[IDB_MAX_KEY_LEN];

  /* Allocate space for the public_xtra structure */
  public_xtra = ctalloc(PublicXtra, 1);


  sprintf(key, "Geom.%s.Perm.FileName", geom_name);
  public_xtra->filename = GetString(key);

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}


/*--------------------------------------------------------------------------
 * InputRFFreePublicXtra
 *--------------------------------------------------------------------------*/

void  InputRFFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * InputRFSizeOfTempData
 *--------------------------------------------------------------------------*/

int  InputRFSizeOfTempData()
{
  int size = 0;

  return size;
}
