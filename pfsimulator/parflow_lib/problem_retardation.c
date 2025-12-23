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
  int num_contaminants;

  int num_geo_unit_indexes;
  int    *geo_indexes;
  NameArray geo_indexes_na;

  int    *type;
  void  **data;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  double  *temp_data;
} InstanceXtra;

typedef struct {
  double  *value;
} Type0;                       /* linear retardation */


/*--------------------------------------------------------------------------
 * Retardation
 *--------------------------------------------------------------------------*/

void         Retardation(
                         Vector *     solidmassfactor,
                         int          contaminant,
                         ProblemData *problem_data)
{
  PFModule   *this_module = ThisPFModule;
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  GrGeomSolid **gr_solids = ProblemDataGrSolids(problem_data);

  Vector       *porosity = ProblemDataPorosity(problem_data);

  Grid         *grid = VectorGrid(solidmassfactor);

  Type0        *dummy0;

  GrGeomSolid *gr_solid;

  Subgrid      *subgrid;
  Subvector    *smf_sub, *p_sub;

  int is, ig;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  double       *smfp, *pp;

  int i, j, k;
  int index;
  int ismf, ip;

  /*-----------------------------------------------------------------------
   * Put in any user defined sources for this phase
   *-----------------------------------------------------------------------*/

  InitVectorAll(solidmassfactor, 1.0);

  for (ig = 0; ig < (public_xtra->num_geo_unit_indexes); ig++)
  {
    gr_solid = gr_solids[(public_xtra->geo_indexes)[ig]];
    index = (public_xtra->num_contaminants) * ig + contaminant;

    switch ((public_xtra->type[index]))
    {
      case 0:
      {
        double value;

        dummy0 = (Type0*)(public_xtra->data[index]);

        value = (dummy0->value)[0];

        for (is = 0; is < GridNumSubgrids(grid); is++)
        {
          subgrid = GridSubgrid(grid, is);
          smf_sub = VectorSubvector(solidmassfactor, is);
          p_sub = VectorSubvector(porosity, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          smfp = SubvectorData(smf_sub);
          pp = SubvectorData(p_sub);
          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ismf = SubvectorEltIndex(smf_sub, i, j, k);
            ip = SubvectorEltIndex(p_sub, i, j, k);

            smfp[ismf] = (pp[ip] + (1.0 - pp[ip]) * value);
          });
        }

        break;
      }
    }
  }
}


/*--------------------------------------------------------------------------
 * RetardationInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *RetardationInitInstanceXtra(
                                       double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

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
 * RetardationFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  RetardationFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * RetardationNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *RetardationNewPublicXtra(
                                    int num_contaminants)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;

  int num_geo_unit_indexes;
  int i, ig;
  int ind, index;


  char          *geo_index_names;
  NameArray geo_index_na;

  char          *switch_name;
  char key[IDB_MAX_KEY_LEN];

  NameArray switch_na;

  char *geom_name;

  /*----------------------------------------------------------
   * The name array to map names to switch values
   *----------------------------------------------------------*/
  switch_na = NA_NewNameArray("Linear");

  if (num_contaminants > 0)
  {
    public_xtra = ctalloc(PublicXtra, 1);

    (public_xtra->num_contaminants) = num_contaminants;

    geo_index_names = GetString("Geom.Retardation.GeomNames");
    geo_index_na = (public_xtra->geo_indexes_na) =
      NA_NewNameArray(geo_index_names);
    num_geo_unit_indexes = NA_Sizeof(geo_index_na);

    (public_xtra->num_geo_unit_indexes) = num_geo_unit_indexes;
    (public_xtra->geo_indexes) = ctalloc(int, num_geo_unit_indexes);

    (public_xtra->type) = ctalloc(int, num_geo_unit_indexes * num_contaminants);
    (public_xtra->data) = ctalloc(void *, num_geo_unit_indexes * num_contaminants);

    for (ig = 0; ig < num_geo_unit_indexes; ig++)
    {
      geom_name = NA_IndexToName(geo_index_na, ig);

      ind = NA_NameToIndex(GlobalsGeomNames, geom_name);

      (public_xtra->geo_indexes)[ig] = ind;

      for (i = 0; i < num_contaminants; i++)
      {
        index = num_contaminants * ig + i;

        sprintf(key, "Geom.%s.%s.Retardation.Type",
                geom_name,
                NA_IndexToName(GlobalsContaminatNames, i));
        switch_name = GetString(key);

        public_xtra->type[index] =
          NA_NameToIndexExitOnError(switch_na, switch_name, key);

        switch ((public_xtra->type[index]))
        {
          case 0:
          {
            dummy0 = ctalloc(Type0, 1);

            (dummy0->value) = ctalloc(double, 1);

            sprintf(key, "Geom.%s.%s.Retardation.Rate",
                    geom_name,
                    NA_IndexToName(GlobalsContaminatNames, i));
            *(dummy0->value) = GetDouble(key);

            (public_xtra->data[index]) = (void*)dummy0;

            break;
          }

          default:
          {
            InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
          }
        }
      }
    }
  }
  else
  {
    public_xtra = NULL;
  }

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*-------------------------------------------------------------------------
 * RetardationFreePublicXtra
 *-------------------------------------------------------------------------*/

void  RetardationFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;

  int i, ig;
  int index;


  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->geo_indexes_na);

    for (ig = 0; ig < (public_xtra->num_geo_unit_indexes); ig++)
    {
      for (i = 0; i < (public_xtra->num_contaminants); i++)
      {
        index = (public_xtra->num_contaminants) * ig + i;

        switch ((public_xtra->type[index]))
        {
          case 0:
          {
            dummy0 = (Type0*)(public_xtra->data[index]);

            tfree(dummy0->value);

            tfree(dummy0);

            break;
          }
        }
      }
    }

    tfree(public_xtra->geo_indexes);

    tfree(public_xtra->data);
    tfree(public_xtra->type);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * RetardationSizeOfTempData
 *--------------------------------------------------------------------------*/

int  RetardationSizeOfTempData()
{
  int sz = 0;

  return sz;
}

