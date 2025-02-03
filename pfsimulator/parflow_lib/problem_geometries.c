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

#include <string.h>
#include <assert.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct indicator_data {
  int num_indicators;                       // Number of indicators for the given solid
  int                     *indicators;  //pointer to dynamic object
  char                    *filename;
  struct  indicator_data  *next_indicator_data;

  NameArray indicator_na;
} IndicatorData;

typedef struct {
  GeomSolid     **solids;
  int num_solids;

  IndicatorData  *indicator_data;

  int time_index;
  int pfsol_time_index;

  /* Geometry input names are for each "type" of geometry
   * the user is inputing */
  NameArray geom_input_names;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid     *grid;
  double   *temp_data;
} InstanceXtra;

void resetBoundary(Vector *vector, const double value, const int ghosts)
{
  const Grid           *grid = VectorGrid(vector);

  /* currently only works if ghost layer is 2 */
  assert(ghosts == 2);

  const SubgridArray   *subgrids = GridSubgrids(grid);
  int sg = 0;
  ForSubgridI(sg, subgrids)
  {
    /* Get information about this subgrid */
    const Subgrid *subgrid = SubgridArraySubgrid(subgrids, sg);

    const int ix = SubgridIX(subgrid);
    const int iy = SubgridIY(subgrid);
    const int iz = SubgridIZ(subgrid);

    const int nx = SubgridNX(subgrid);
    const int ny = SubgridNY(subgrid);
    const int nz = SubgridNZ(subgrid);

    /* Compute information about the subgrid and ghost layer */
    const int ix_all = ix - ghosts;
    const int iy_all = iy - ghosts;
    const int iz_all = iz - ghosts;

    const int nx_all = nx + 2 * ghosts;
    const int ny_all = ny + 2 * ghosts;
    const int nz_all = nz + 2 * ghosts;

    /* Get information about this subvector */
    const Subvector *subvector = VectorSubvector(vector, sg);

    const int nx_f = SubvectorNX(subvector);
    const int ny_f = SubvectorNY(subvector);
    const int nz_f = SubvectorNZ(subvector);

    double *data = SubvectorElt(subvector, ix_all, iy_all, iz_all);

    int i, j, k;
    int fi = 0;

    BoxLoopI1(i, j, k,
              ix_all, iy_all, iz_all, nx_all, ny_all, nz_all,
              fi, nx_f, ny_f, nz_f, 1, 1, 1,
    {
      if (i == ix_all ||
          j == iy_all ||
          k == iz_all ||
          (i == (ix_all + nx_all - 1)) ||
          (j == (iy_all + ny_all - 1)) ||
          (k == (iz_all + nz_all - 1)))
      {
        data[fi] = value;
      }
      else if (((i == ix_all + 1) && (j == iy_all + 1)) ||
               ((i == ix_all + 1) && (k == iz_all + 1)) ||
               ((i == ix_all + 1) && (j == iy_all + ny_all - 2)) ||
               ((i == ix_all + 1) && (k == iz_all + nz_all - 2)) ||
               ((i == ix_all + nx_all - 2) && (j == iy_all + 1)) ||
               ((i == ix_all + nx_all - 2) && (k == iz_all + 1)) ||
               ((i == ix_all + nx_all - 2) && (j == iy_all + ny_all - 2)) ||
               ((i == ix_all + nx_all - 2) && (k == iz_all + nz_all - 2)))
      {
        data[fi] = value;
      }
      else if (((j == iy_all + 1) && (k == iz_all + 1)) ||
               ((j == iy_all + 1) && (k == iz_all + nz_all - 2)) ||
               ((j == iy_all + ny_all - 2) && (k == iz_all + 1)) ||
               ((j == iy_all + ny_all - 2) && (k == iz_all + nz_all - 2)))
      {
        data[fi] = value;
      }
    });
  }
}

/*--------------------------------------------------------------------------
 * Geometries
 *--------------------------------------------------------------------------*/

void           Geometries(
                          ProblemData *problem_data)
{
  PFModule           *this_module = ThisPFModule;
  PublicXtra         *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra       *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  GeomSolid         **solids = (public_xtra->solids);
  int num_solids = (public_xtra->num_solids);
  IndicatorData      *indicator_data = (public_xtra->indicator_data);

  Grid               *grid = (instance_xtra->grid);
  Vector             *tmp_indicator_field = NULL;

  GrGeomSolid       **gr_solids;

  GrGeomExtentArray  *extent_array;

  IndicatorData      *current_indicator_data;

  VectorUpdateCommHandle         *handle;

  int i, k;

  BeginTiming(public_xtra->time_index);

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  tmp_indicator_field = NewVectorType(instance_xtra->grid, 1, 2, vector_cell_centered);

  gr_solids = ctalloc(GrGeomSolid *, num_solids);

  /*-----------------------------------------------------------------------
   * Convert Geom solids to GrGeom solids
   *-----------------------------------------------------------------------*/

  extent_array = GrGeomCreateExtentArray(GridSubgrids(grid), 3, 3, 3, 3, 3, 3);
  for (i = 0; i < num_solids; i++)
  {
    if (solids[i])
    {
      GrGeomSolidFromGeom(&gr_solids[i], solids[i], extent_array);
    }
  }
  GrGeomFreeExtentArray(extent_array);

  /*-----------------------------------------------------------------------
   * Convert indicator solids to GrGeom solids
   *-----------------------------------------------------------------------*/

  current_indicator_data = indicator_data;

  i = 0;
  while (current_indicator_data != NULL)
  {
    InitVectorAll(tmp_indicator_field, -1.0);
    ReadPFBinary((current_indicator_data->filename), tmp_indicator_field);
    handle = InitVectorUpdate(tmp_indicator_field, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

#ifdef HAVE_SAMRAI
    /*
     * SGS This should be removed after vector updates
     * have been improved in the SAMRAI port.   This algorithm
     * can't have all the ghost cells with valid data, it depends
     * on the outermost layer to be an invalid indicator value
     */
    resetBoundary(tmp_indicator_field, -1.0, 2);
#endif

    for (k = 0; k < (current_indicator_data->num_indicators); k++)
    {
      while (gr_solids[i])
      {
        i++;
      }
      GrGeomSolidFromInd(&gr_solids[i], tmp_indicator_field, (current_indicator_data->indicators)[k]);
    }

    current_indicator_data = (current_indicator_data->next_indicator_data);
  }

  EndTiming(public_xtra->time_index);

  ProblemDataNumSolids(problem_data) = num_solids;
  ProblemDataSolids(problem_data) = solids;
  ProblemDataGrSolids(problem_data) = gr_solids;

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(tmp_indicator_field);

  return;
}


/*--------------------------------------------------------------------------
 * GeometriesInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *GeometriesInitInstanceXtra(
                                      Grid *grid)
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

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * GeometriesFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  GeometriesFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * GeometriesNewPublicXtra:
 *--------------------------------------------------------------------------*/

PFModule   *GeometriesNewPublicXtra()
{
  PFModule       *this_module = ThisPFModule;
  PublicXtra     *public_xtra;

  int num_solids;
  GeomSolid     **solids = NULL;

  int num_new_solids;
  GeomSolid     **new_solids;

  GeomSolid     **tmp_solids;

  IndicatorData  *indicator_data, *new_indicator_data, *current_indicator_data = NULL;

  int i, g, intype, num_intypes;

  NameArray switch_na;
  NameArray geom_input_na;

  char *geom_input_names;
  char key[IDB_MAX_KEY_LEN];

  char *intype_name;

  char *geom_name;

  char *patch_names;

  /*----------------------------------------------------------
   * The name array to map names to switch values
   *----------------------------------------------------------*/
  switch_na = NA_NewNameArray("IndicatorField SolidFile Box");


  public_xtra = ctalloc(PublicXtra, 1);

  /*----------------------------------------------------------
   * Read in the number of input types (intype).
   *----------------------------------------------------------*/

  geom_input_names = GetString("GeomInput.Names");

  geom_input_na =
    public_xtra->geom_input_names = NA_NewNameArray(geom_input_names);

  num_intypes = NA_Sizeof(geom_input_na);

  /*----------------------------------------------------------
   * Read in the input type, then read in the associated data.
   *----------------------------------------------------------*/

  num_solids = 0;
  indicator_data = NULL;

  for (i = 0; i < num_intypes; i++)
  {
    sprintf(key, "GeomInput.%s.InputType",
            NA_IndexToName(geom_input_na, i));
    intype_name = GetString(key);
    intype = NA_NameToIndexExitOnError(switch_na, intype_name, key);

    num_new_solids = 0;

    switch (intype)
    {
      case 0:    /* indicator field */
      {
        char *indicator_names;
        int solids_index;

        /* create a new indicator data structure */
        new_indicator_data = ctalloc(IndicatorData, 1);

        sprintf(key, "GeomInput.%s.GeomNames",
                NA_IndexToName(geom_input_na, i));
        indicator_names = GetString(key);


        (new_indicator_data->indicator_na) =
          NA_NewNameArray(indicator_names);

        /* Add this geom input geometries to the total array */
        if (GlobalsGeomNames)
          NA_AppendToArray(GlobalsGeomNames, indicator_names);
        else
          GlobalsGeomNames = NA_NewNameArray(indicator_names);

        num_new_solids = NA_Sizeof(new_indicator_data->indicator_na);

        (new_indicator_data->num_indicators) = num_new_solids;

        /* allocate and read in the indicator values */
        (new_indicator_data->indicators) = ctalloc(int, num_new_solids);

        for (solids_index = 0; solids_index < num_new_solids; solids_index++)
        {
          sprintf(key, "GeomInput.%s.Value",
                  NA_IndexToName(new_indicator_data->indicator_na,
                                 solids_index));
          new_indicator_data->indicators[solids_index] = GetInt(key);
        }

        /* read in the number of characters in the filename for the indicator field */

        sprintf(key, "Geom.%s.FileName", NA_IndexToName(geom_input_na, i));
        new_indicator_data->filename = strdup(GetString(key));

        /* make sure and NULL out the next_indicator_data pointer */
        (new_indicator_data->next_indicator_data) = NULL;

        /* do some bookkeeping on the list */
        if (indicator_data == NULL)
        {
          indicator_data = new_indicator_data;
        }
        else
        {
          (current_indicator_data->next_indicator_data) = new_indicator_data;
        }
        current_indicator_data = new_indicator_data;

        /* allocate some new solids */
        new_solids = ctalloc(GeomSolid *, num_new_solids);

        break;
      }

      case 1:    /* `.pfsol' file */
      {
        BeginTiming(PFSOLReadTimingIndex);
        num_new_solids = GeomReadSolids(&new_solids,
                                        NA_IndexToName(geom_input_na, i),
                                        GeomTSolidType);

#if 0
        if (amps_Rank() == 73)
        {
          char filename[2056];
          sprintf(filename, "octree.%d.domain.pfsol", amps_Rank());
          GrGeomOctree  *data;
          data = new_solids[0]->data;
          GrGeomPrintOctree(filename, data);
        }
#endif


        sprintf(key, "GeomInput.%s.GeomNames",
                NA_IndexToName(geom_input_na, i));
        geom_name = GetString(key);

        /* Add this geom input geometries to the total array */
        if (GlobalsGeomNames)
          NA_AppendToArray(GlobalsGeomNames, geom_name);
        else
          GlobalsGeomNames = NA_NewNameArray(geom_name);

        /* Get the patch names; patches exist only on the first
         * object in the input file */
        sprintf(key, "Geom.%s.Patches",
                NA_IndexToName(GlobalsGeomNames, num_solids));
        patch_names = GetString(key);
        new_solids[0]->patches = NA_NewNameArray(patch_names);

        EndTiming(PFSOLReadTimingIndex);

        break;
      }

      case 2:    /* box */
      {
        double xl, yl, zl, xu, yu, zu;

        sprintf(key, "GeomInput.%s.GeomName",
                NA_IndexToName(geom_input_na, i));
        geom_name = GetString(key);

        /* Add this geom input geometries to the total array */
        if (GlobalsGeomNames)
          NA_AppendToArray(GlobalsGeomNames, geom_name);
        else
          GlobalsGeomNames = NA_NewNameArray(geom_name);

        /* Input the box extents */
        sprintf(key, "Geom.%s.Lower.X", geom_name);
        xl = GetDouble(key);
        sprintf(key, "Geom.%s.Lower.Y", geom_name);
        yl = GetDouble(key);
        sprintf(key, "Geom.%s.Lower.Z", geom_name);
        zl = GetDouble(key);

        sprintf(key, "Geom.%s.Upper.X", geom_name);
        xu = GetDouble(key);
        sprintf(key, "Geom.%s.Upper.Y", geom_name);
        yu = GetDouble(key);
        sprintf(key, "Geom.%s.Upper.Z", geom_name);
        zu = GetDouble(key);

        new_solids = ctalloc(GeomSolid *, 1);
        new_solids[0] =
          GeomSolidFromBox(xl, yl, zl, xu, yu, zu, GeomTSolidType);
        num_new_solids = 1;

        /* Get the patch names */
        sprintf(key, "Geom.%s.Patches", geom_name);
        patch_names = GetStringDefault(key,
                                       "left right front back bottom top");
        new_solids[0]->patches = NA_NewNameArray(patch_names);

        break;
      }
    }

    /*-------------------------------------------------------
     * Add the new_solids to the solids list.
     *-------------------------------------------------------*/

    if (num_new_solids)
    {
      tmp_solids = solids;
      solids = ctalloc(GeomSolid *, (num_solids + num_new_solids));

      for (g = 0; g < num_solids; g++)
        solids[g] = tmp_solids[g];
      for (g = 0; g < num_new_solids; g++)
        solids[num_solids + g] = new_solids[g];

      tfree(tmp_solids);
      tfree(new_solids);

      num_solids += num_new_solids;
    }
  }

  (public_xtra->solids) = solids;
  (public_xtra->num_solids) = num_solids;
  (public_xtra->indicator_data) = indicator_data;

  (public_xtra->time_index) = RegisterTiming("Geometries");

  /* Geometries need to be world accessible */
  GlobalsGeometries = solids;

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * GeometriesFreePublicXtra
 *--------------------------------------------------------------------------*/

void  GeometriesFreePublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  IndicatorData *current_indicator_data, *tmp_indicator_data;

  int g;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->geom_input_names);

    NA_FreeNameArray(GlobalsGeomNames);

    for (g = 0; g < (public_xtra->num_solids); g++)
      if (public_xtra->solids[g])
        GeomFreeSolid(public_xtra->solids[g]);
    tfree(public_xtra->solids);


    current_indicator_data = (public_xtra->indicator_data);
    while (current_indicator_data != NULL)
    {
      tmp_indicator_data = (current_indicator_data->next_indicator_data);

      if (current_indicator_data->indicators)
      {
        tfree(current_indicator_data->indicators);
      }

      if (current_indicator_data->filename)
      {
        tfree(current_indicator_data->filename);
      }

      if (current_indicator_data->indicator_na)
      {
        NA_FreeNameArray(current_indicator_data->indicator_na);
      }

      tfree(current_indicator_data);

      current_indicator_data = tmp_indicator_data;
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GeometriesSizeOfTempData
 *--------------------------------------------------------------------------*/

int  GeometriesSizeOfTempData()
{
  int sz = 0;

  return sz;
}

