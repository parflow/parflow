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
* Routines to write a Vector to Silo file.
*
*****************************************************************************/

#include "parflow.h"

#ifdef HAVE_SILO
#include "silo.h"
#endif

#include <string.h>
#include <math.h>
#include <sys/stat.h>

amps_ThreadLocalDcl(int, s_parflow_silo_filetype);

#ifdef HAVE_SILO
void       WriteSilo_Subvector(DBfile *db_file, Subvector *subvector, Subgrid   *subgrid,
                               char *variable_name)
{
  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);
  int nz_v = SubvectorNZ(subvector);

  int i, j, k, ai;
  double         *data;

  int err;

  char varname[512];
  char meshname[512];

  float *coords[3];
  int dims[3];

  int p = amps_Rank(amps_CommWorld);

  dims[0] = nx + 1;
  dims[1] = ny + 1;
  dims[2] = nz + 1;

  /* Write the origin information */
  int origin_dims[1];
  origin_dims[0] = 3;

  int origin[3];
  origin[0] = ix;
  origin[1] = iy;
  origin[2] = iz;
  err = DBWrite(db_file, "index_origin", origin, origin_dims, 1, DB_INT);
  if (err < 0)
  {
    amps_Printf("Error: Silo failed on DBWrite\n");
  }

  // Loops to set coords
  coords[0] = ctalloc(float, dims[0]);
  for (i = 0; i < dims[0]; i++)
  {
    coords[0][i] = SubgridX(subgrid) + SubgridDX(subgrid) * ((float)i - 0.5);
  }

  coords[1] = ctalloc(float, dims[1]);
  for (j = 0; j < dims[1]; j++)
  {
    coords[1][j] = SubgridY(subgrid) + SubgridDY(subgrid) * ((float)j - 0.5);
  }

  coords[2] = ctalloc(float, dims[2]);
  /* z_coord = SubgridZ(subgrid); */
/*  @RMM-- bare bones testing
 * for implementing variable dz into silo output
 * need to brab the vardz vector out of problem data
 * and use in a sum as indicated here  */
  for (k = 0; k < dims[2]; k++)
  {
    /* mult = 1.0;
     * if ( k > 19 ) {
     *   mult = 0.4;
     * }
     * z_coord +=  SubgridDZ(subgrid)*mult;
     * coords[2][k] =  z_coord; */
    coords[2][k] = SubgridZ(subgrid) + SubgridDZ(subgrid) * ((float)k - 0.5);
  }

  sprintf(meshname, "%s_%06u", "mesh", p);

  err = DBPutQuadmesh(db_file, meshname, NULL, coords, dims,
                      3, DB_FLOAT, DB_COLLINEAR, NULL);
  if (err < 0)
  {
    amps_Printf("Error: Silo put quadmesh failed %s\n", meshname);
    exit(1);
  }

  data = SubvectorElt(subvector, ix, iy, iz);

  /* SGS Note:
   * this is way to slow we need better boxloops that operate
   * on arrays.
   * Can this be some kind of copy?  Array ordering, ghost layers?
   */

  double *array = ctalloc(double, nx * ny * nz);
  int array_index = 0;
  ai = 0;
  BoxLoopI1(i, j, k,
            ix, iy, iz, nx, ny, nz,
            ai, nx_v, ny_v, nz_v, 1, 1, 1,
  {
    array[array_index++] = data[ai];
  });

  dims[0] = nx;
  dims[1] = ny;
  dims[2] = nz;

  sprintf(varname, "%s_%06u", variable_name, p);
  err = DBPutQuadvar1(db_file, varname, meshname,
                      (float*)array, dims, 3,
                      NULL, 0, DB_DOUBLE,
                      DB_ZONECENT, NULL);
  if (err < 0)
  {
    amps_Printf("Error: Silo put quadvar1 failed %s\n", varname);
    exit(1);
  }

  free(array);
  free(coords[0]);
  free(coords[1]);
  free(coords[2]);
}
#endif

void pf_mk_dir(char* filename)
{
  struct stat status;
  int err = stat(filename, &status);

  if (err == -1)
  {
    err = mkdir(filename, S_IRUSR | S_IWUSR | S_IXUSR);
    if (err < 0)
    {
      amps_Printf("Error: can't create directory for silo files %s\n", filename);
      exit(1);
    }
  }
  else if (!S_ISDIR(status.st_mode))
  {
    amps_Printf("Error: can't create directory for silo files %s, file exists with that name\n", filename);
    exit(1);
  }
}

/*
 * Silo file writing uses a directory to store files.  This directory
 * needs to be created somewhere so an Init step is required.
 */
void     WriteSiloInit(char *file_prefix)
{
#ifdef HAVE_SILO
  char filename[2048];

  int p = amps_Rank(amps_CommWorld);
  int P = amps_Size(amps_CommWorld);

  char key[IDB_MAX_KEY_LEN];

  /*
   * Get compression options for SILO files
   */

  sprintf(key, "SILO.CompressionOptions");
  char *compression_options = GetStringDefault(key, "");
  if (strlen(compression_options))
  {
    DBSetCompression(compression_options);
    if (db_errno < 0)
    {
      amps_Printf("Error: Compression options failed for SILO.CompressionOptions=%s\n", compression_options);
      amps_Printf("       This may mean SILO was not compiled with compression enabled\n");
      exit(1);
    }
  }

  sprintf(key, "SILO.Filetype");
  char *switch_name = GetStringDefault(key, "PDB");
  NameArray type_na = NA_NewNameArray("PDB HDF5");

  switch (NA_NameToIndexExitOnError(type_na, switch_name, key))
  {
    case 0:
    {
      s_parflow_silo_filetype = DB_PDB;
      break;
    }

    case 1:
    {
      s_parflow_silo_filetype = DB_HDF5;
      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }

  NA_FreeNameArray(type_na);

  if (p == 0)
  {
    sprintf(filename, "%s", file_prefix);
    pf_mk_dir(filename);
    /*
     * SGS This is very hacky
     */
    char* output_types[] = { "perm_x",
                             "perm_y",
                             "perm_z",
                             "porosity",
                             "satur",
                             "concen",
                             "press",
                             "slope_x",
                             "slope_y",
                             "mannings",
                             "specific_storage",
                             "mask",
                             "dz_mult",
                             "top_zindex",
                             "top_patch",
                             "eflx_lh_tot",
                             "eflx_lwrad_out",
                             "eflx_sh_tot",
                             "eflx_soil_grnd",
                             "qflx_evap_tot",
                             "qflx_evap_grnd",
                             "qflx_evap_soi",
                             "qflx_evap_veg",
                             "qflx_tran_veg",
                             "qflx_infl",
                             "swe_out",
                             "t_grnd",
                             "t_soil",
                             "qflx_qirr",
                             "qflx_qirr_inst",
                             "evaptrans",
                             "evaptranssum",
                             "overlandsum",
                             "overland_bc_flux",
                             0 };

    for (int i = 0; output_types[i]; i++)
    {
      sprintf(filename, "%s/%s", file_prefix, output_types[i]);
      pf_mk_dir(filename);

      for (int j = 0; j < P; j++)
      {
        sprintf(filename, "%s/%s/%06u", file_prefix, output_types[i], j);
        pf_mk_dir(filename);
      }
    }
  }
#endif
}


/*
 * Write a Vector to a Silo file.
 *
 * Notes:
 * Silo files can store additinal metadata such as name of variable,
 * simulation time etc.  These should be added.
 */
void     WriteSilo(char *  file_prefix,
                   char *  file_type,
                   char *  file_suffix,
                   Vector *v,
                   double  time,
                   int     step,
                   char *  variable_name)
{
#ifdef HAVE_SILO
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;

  int g;
  int p, P;

  char file_extn[7] = "silo";
  char filename[512];
  int err;
  DBfile *db_file;
#endif

  BeginTiming(PFBTimingIndex);

#ifdef HAVE_SILO
  p = amps_Rank(amps_CommWorld);
  P = amps_Size(amps_CommWorld);
  if (p == 0)
  {
    int i;
    char **meshnames;
    int   *meshtypes;

    char **varnames;
    int   *vartypes;

    DBoptlist *optlist;

    meshnames = ctalloc(char *, P);
    meshtypes = ctalloc(int, P);

    varnames = ctalloc(char *, P);
    vartypes = ctalloc(int, P);

    for (i = 0; i < P; i++)
    {
      char *name = ctalloc(char, 2048);
      if (strlen(file_suffix))
      {
        sprintf(name, "%s/%s/%06u/data.%s.%s:mesh_%06u", file_prefix, file_type, i, file_suffix, file_extn, i);
      }
      else
      {
        sprintf(name, "%s/%s/%06u/data.%s:mesh_%06u", file_prefix, file_type, i, file_extn, i);
      }
      meshnames[i] = name;
      meshtypes[i] = DB_QUADMESH;

      name = ctalloc(char, 2048);
      if (strlen(file_suffix))
      {
        sprintf(name, "%s/%s/%06u/data.%s.%s:%s_%06u", file_prefix, file_type, i, file_suffix, file_extn,
                variable_name, i);
      }
      else
      {
        sprintf(name, "%s/%s/%06u/data.%s:%s_%06u", file_prefix, file_type, i, file_extn,
                variable_name, i);
      }
      varnames[i] = name;
      vartypes[i] = DB_QUADVAR;
    }

    /* open file */
    if (strlen(file_suffix))
    {
      sprintf(filename, "%s.%s.%s.%s", file_prefix, file_type, file_suffix, file_extn);
    }
    else
    {
      sprintf(filename, "%s.%s.%s", file_prefix, file_type, file_extn);
    }

    /* TODO SGS what type? HDF PDB? */
    db_file = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, s_parflow_silo_filetype);

    if (db_file == NULL)
    {
      amps_Printf("Error: can't open silo file %s\n", filename);
      exit(1);
    }

    /* Multimesh information; one file per processor */
    optlist = DBMakeOptlist(2);
    DBAddOption(optlist, DBOPT_CYCLE, &step);
    DBAddOption(optlist, DBOPT_DTIME, &time);

    err = DBPutMultimesh(db_file, "mesh", P, (const char*const*)meshnames, meshtypes, optlist);
    if (err < 0)
    {
      amps_Printf("Error: Silo put multimesh failed %s\n", filename);
      exit(1);
    }

    DBFreeOptlist(optlist);

    optlist = DBMakeOptlist(2);
    DBAddOption(optlist, DBOPT_CYCLE, &step);
    DBAddOption(optlist, DBOPT_DTIME, &time);

    /* Multimesh information; one file per processor */
    err = DBPutMultivar(db_file, variable_name, P, (const char*const*)varnames, vartypes, optlist);
    if (err < 0)
    {
      amps_Printf("Error: Silo put multivar failed %s\n", filename);
      exit(1);
    }

    DBFreeOptlist(optlist);

    /* Write out some meta information for pftools */

    /* Write the origin information */
    int dims[1];
    dims[0] = 3;

    double origin[3];
    origin[0] = SubgridX(GridSubgrid(GlobalsUserGrid, 0));
    origin[1] = SubgridY(GridSubgrid(GlobalsUserGrid, 0));
    origin[2] = SubgridZ(GridSubgrid(GlobalsUserGrid, 0));
    err = DBWrite(db_file, "origin", origin, dims, 1, DB_DOUBLE);
    if (err < 0)
    {
      amps_Printf("Error: Silo failed on DBWrite\n");
    }

    /* Write the size information */
    int size[3];
    size[0] = SubgridNX(GridSubgrid(GlobalsUserGrid, 0));
    size[1] = SubgridNY(GridSubgrid(GlobalsUserGrid, 0));
    size[2] = SubgridNZ(GridSubgrid(GlobalsUserGrid, 0));
    err = DBWrite(db_file, "size", size, dims, 1, DB_INT);
    if (err < 0)
    {
      amps_Printf("Error: Silo failed on DBWrite\n");
    }

    /* Write the delta information */
    double delta[3];
    delta[0] = SubgridDX(GridSubgrid(GlobalsUserGrid, 0));
    delta[1] = SubgridDY(GridSubgrid(GlobalsUserGrid, 0));
    delta[2] = SubgridDZ(GridSubgrid(GlobalsUserGrid, 0));
    err = DBWrite(db_file, "delta", delta, dims, 1, DB_DOUBLE);
    if (err < 0)
    {
      amps_Printf("Error: Silo failed on DBWrite\n");
    }

    err = DBClose(db_file);
    if (err < 0)
    {
      amps_Printf("Error: can't close silo file %s\n", filename);
    }

    /* Free up allocated variables */
    for (i = 0; i < P; i++)
    {
      free(meshnames[i]);
      free(varnames[i]);
    }
    free(meshnames);
    free(meshtypes);
    free(varnames);
    free(vartypes);
  }

  if (strlen(file_suffix))
  {
    sprintf(filename, "%s/%s/%06u/data.%s.%s", file_prefix, file_type, p, file_suffix, file_extn);
  }
  else
  {
    sprintf(filename, "%s/%s/%06u/data.%s", file_prefix, file_type, p, file_extn);
  }

  /* TODO SGS what type? HDF PDB? */
  db_file = DBCreate(filename, DB_CLOBBER, DB_LOCAL, NULL, s_parflow_silo_filetype);
  if (db_file == NULL)
  {
    amps_Printf("Error: can't open silo file %s\n", filename);
  }

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    WriteSilo_Subvector(db_file, subvector, subgrid, variable_name);
  }

  err = DBClose(db_file);
  if (err < 0)
  {
    amps_Printf("Error: can't close silo file %s\n", filename);
  }
#else
  amps_Printf("Parflow not compiled with SILO, can't create SILO file\n");
#endif

  EndTiming(PFBTimingIndex);
}

