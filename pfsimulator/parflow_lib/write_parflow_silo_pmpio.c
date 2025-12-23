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

#if defined(HAVE_SILO) && defined(HAVE_MPI)
#include "silo.h"
#include <mpi.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <pmpio.h>
#pragma GCC diagnostic pop
#endif

#include <string.h>
#include <math.h>
#include <sys/stat.h>

amps_ThreadLocalDcl(int, s_num_silo_files);

/*-----------------------------------------------------------------------------
 * Purpose:     Implement the create callback to initialize pmpio
 *              Will create the silo file and the 'first' directory (namespace)
 *              in it. The driver type (DB_PDB or DB_HDF5) is passed as user
 *              data; a void pointer to the driver determined in main.
 *-----------------------------------------------------------------------------
 */
#if defined(HAVE_SILO) && defined(HAVE_MPI)
void *CreateSiloFile(const char *fname, const char *nsname, void *userData)
{
  int driver = *((int*)userData);
  DBfile *db_file = DBCreate(fname, DB_CLOBBER, DB_LOCAL, "pmpio", driver);

  return (void*)db_file;
}

/*-----------------------------------------------------------------------------
 * Purpose:     Implement the open callback to initialize pmpio
 *              Will open the silo file and, for write, create the new
 *              directory or, for read, just cd into the right directory.
 *-----------------------------------------------------------------------------
 */
void *OpenSiloFile(const char *fname, const char *nsname, PMPIO_iomode_t ioMode,
                   void *userData)
{
  DBfile *db_file = DBOpen(fname, DB_UNKNOWN,
                           ioMode == PMPIO_WRITE ? DB_APPEND : DB_READ);

  return (void*)db_file;
}

/*-----------------------------------------------------------------------------
 * Purpose:     Implement the close callback for pmpio
 *-----------------------------------------------------------------------------
 */
void CloseSiloFile(void *file, void *userData)
{
  DBfile *db_file = (DBfile*)file;

  if (db_file)
    DBClose(db_file);
}

#endif

/*
 * Silo file writing uses a directory to store files.  This directory
 * needs to be created somewhere so an Init step is required.
 */
void     WriteSiloPMPIOInit(char *file_prefix)
{
#if defined(HAVE_SILO) && defined(HAVE_MPI)
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

  /* need database key for number of files, need to check if numgroups > npocs */
  /* should put this in an init routine */
  s_num_silo_files = GetIntDefault("SILO.pmpio.NumFiles", 1);

  if (s_num_silo_files > P + 1)
  {
    amps_Printf("Error: Number of SILO files %d \n", s_num_silo_files);
    amps_Printf("       exceeds the number of processors %d \n", P + 1);
    exit(1);
  }
  if (s_num_silo_files < 1)
  {
    amps_Printf("Error: Number of SILO files %d \n", s_num_silo_files);
    amps_Printf("       is less than 1 \n");
    exit(1);
  }
  /*  @RMM TODO: we should check if only one PMPIO file is to be written, if
   * so we should not use the header-data framework and subdirs, but include just a single
   * file structure for Init and writing */

/* @RMM right now only PDB but no reason to not reinstate this to include HDF5/NetCDF etc
 *       after sufficient testing */
/*    sprintf(key, "SILO.Filetype");
 *  char *switch_name = GetStringDefault(key, "PDB");
 *  NameArray type_na = NA_NewNameArray("PDB HDF5");
 *
 *  switch (NA_NameToIndex(type_na, switch_name)) {
 *      case 0:
 *          s_parflow_silo_filetype = DB_PDB;
 *          break;
 *      case 1:
 *          s_parflow_silo_filetype = DB_HDF5;
 *          break;
 *      default:
 *          amps_Printf("Error: invalid SILO.Filetype %s\n", switch_name);
 *          exit(1);
 *          break;
 *  }
 *
 *  NA_FreeNameArray(type_na);  */

  /* if ( num_silo_files > 1 ) { */
  if (p == 0)
  {
    /* @RMM we only have a subdir for data for nfiles >1 */
    // if (num_silo_files > 1) {
    sprintf(filename, "%s", file_prefix);
    pf_mk_dir(filename);
    /*
     * @RMM-- no subdir structure for PMPIO
     */
    //}
  }
#else
  amps_Printf("Parflow not compiled with SILO and MPI, can't use SILO PMPIO\n");
#endif
}


/*
 * Write a Vector to a Silo file.
 *
 * Notes:
 * Silo files can store additional metadata such as name of variable,
 * simulation time etc.  These should be added.
 */
void     WriteSiloPMPIO(char *  file_prefix,
                        char *  file_type,
                        char *  file_suffix,
                        Vector *v,
                        double  time,
                        int     step,
                        char *  variable_name)
{
#if defined(HAVE_SILO) && defined(HAVE_MPI)
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;

  int g;
  int p, P;

  char file_extn[7] = "silo";
  char filename[512];
  char filename2[512];
  char nsName[256];
  int i, j, k, ai;
  double         *data;

  int err;
  int origin_dims2[1];

  char varname[512];
  char meshname[512];

  float *coords[3];
  int dims[3];
  int origin2[3];



  int driver = DB_PDB;
  int numGroups;
  PMPIO_baton_t *bat;

  DBfile *db_file;
  DBfile *db_header_file;
#endif

  BeginTiming(PFBTimingIndex);

#if defined(HAVE_SILO) && defined(HAVE_MPI)
  p = amps_Rank(amps_CommWorld);
  P = amps_Size(amps_CommWorld);
  numGroups = s_num_silo_files;

  bat = PMPIO_Init(numGroups, PMPIO_WRITE, amps_CommWorld, 1,
                   CreateSiloFile, OpenSiloFile, CloseSiloFile, &driver);
//    if (numGroups > 1) {
  if (strlen(file_suffix))
  {
    sprintf(filename2, "%s/%s.data.%03u.%s.%s", file_prefix, file_type, PMPIO_GroupRank(bat, p), file_suffix, file_extn);
  }
  else
  {
    sprintf(filename2, "%s/%s.data.%03u.%s", file_prefix, file_type, PMPIO_GroupRank(bat, p), file_extn);
  }
  /* } else {
   *   if(strlen(file_suffix)) {
   *       sprintf(filename2, "%s.pmpio.%s.%s.%s", file_prefix, file_type, file_suffix, file_extn);
   *   } else {
   *       sprintf(filename2, "%s.pmpio.%s.%s", file_prefix, file_type, file_extn);
   *   }
   * } */

  //  if (numGroups == 1) {
  sprintf(nsName, "domain_%06u", p);   /* note, even though I set this for the open routine we don't use domain structure for multiple files, all done
                                        * in the mesh.  For a single file (this case) we do use domains */
//    } else {
//        nsName == "";
//    }

  /* Wait for write access to the file. All processors call this.
   * Some processors (the first in each group) return immediately
   * with write access to the file. Other processors wind up waiting
   * until they are given control by the preceding processor in
   * the group when that processor calls "HandOffBaton" */
  db_file = (DBfile*)PMPIO_WaitForBaton(bat, filename2, nsName);


  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);
    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int iz = SubgridIZ(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    int nz = SubgridNZ(subgrid);

    int nx_v = SubvectorNX(subvector);
    int ny_v = SubvectorNY(subvector);
    int nz_v = SubvectorNZ(subvector);

    dims[0] = nx + 1;
    dims[1] = ny + 1;
    dims[2] = nz + 1;

    /* Write the origin information */

    origin_dims2[0] = 3;


    origin2[0] = ix;
    origin2[1] = iy;
    origin2[2] = iz;

    err = DBWrite(db_file, "index_origin", origin2, origin_dims2, 1, DB_INT);
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
    /*  @RMM-- bare bones testing
     * for implementing variable dz into silo output
     * need to brab the vardz vector out of problem data
     * and use in a sum as indicated here  */
    for (k = 0; k < dims[2]; k++)
    {
      /* mult = 1.0;
       * if ( k > 19 ) {
       * mult = 0.4;
       * }
       * z_coord +=  SubgridDZ(subgrid)*mult;
       * coords[2][k] =  z_coord; */
      coords[2][k] = SubgridZ(subgrid) + SubgridDZ(subgrid) * ((float)k - 0.5);
    }

//       if (numGroups > 1) {
    sprintf(meshname, "%s_%06u", "mesh", p);
//       } else {
//           sprintf(meshname,"%s","mesh");
//       }
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

//       if (numGroups > 1 ) {
    sprintf(varname, "%s_%06u", variable_name, p);
    //      } else {
//           sprintf(varname, "%s", variable_name);

//       }
    // sprintf(varname, "press_%05d",  p);

    err = DBPutQuadvar1(db_file, varname, meshname,
                        (float*)array, dims, 3,
                        NULL, 0, DB_DOUBLE,
                        DB_ZONECENT, NULL);

    free(array);

    /* If this is the 'root' processor, also write the main Silo header file */
    if (p == 0)
    {
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
        /* Go to root directory in the silo file */

        int groupRank = PMPIO_GroupRank(bat, i);
        char *name = ctalloc(char, 2048);
        //   if (numGroups > 1) {
        if (strlen(file_suffix))
        {
          /* data file in subdir*/
          sprintf(name, "%s/%s.data.%03u.%s.%s:mesh_%06u", file_prefix, file_type, groupRank, file_suffix, file_extn, i);
        }
        else
        {
          sprintf(name, "%s/%s.data.%03u.%s:mesh_%06u", file_prefix, file_type, groupRank, file_extn, i);
        }
        // *} else {
        /* this file */
        //      sprintf(name, "/domain_%06u/mesh", i);
        //    }


        meshnames[i] = name;
        meshtypes[i] = DB_QUADMESH;
        name = ctalloc(char, 2048);
        //     if (numGroups > 1) {
        if (strlen(file_suffix))
        {
          sprintf(name, "%s/%s.data.%03u.%s.%s:%s_%06u", file_prefix, file_type, groupRank, file_suffix, file_extn, variable_name, i);
        }
        else
        {
          sprintf(name, "%s/%s.data.%03u.%s:%s_%06u", file_prefix, file_type, groupRank, file_extn, variable_name, i);
        }
//               } else {
//
//                       sprintf(name, "/domain_%06u/%s", i, variable_name);
//               }
        varnames[i] = name;
        vartypes[i] = DB_QUADVAR;
      }

      /* open file */
      if (strlen(file_suffix))
      {
        sprintf(filename, "%s.pmpio.%s.%s.%s", file_prefix, file_type, file_suffix, file_extn);
      }
      else
      {
        sprintf(filename, "%s.pmpio.%s.%s", file_prefix, file_type, file_extn);
      }

      /* TODO SGS what type? HDF PDB? */
      //  s_parflow_silo_filetype = DB_PDB;

//                 if (numGroups > 1) {
      db_header_file = DBCreate(filename, DB_CLOBBER, DB_LOCAL, "pmpio", driver);
      if (db_header_file == NULL)
      {
        amps_Printf("Error: can't open silo file %s\n", filename);
        exit(1);
      }
//                 } else {
//                     db_header_file = db_file;
      /* Go to root directory in the silo file */
//                     DBSetDir(db_file, "/");
//                 }


      /* Multimesh information; pmpio */
      optlist = DBMakeOptlist(2);
      DBAddOption(optlist, DBOPT_CYCLE, &step);
      DBAddOption(optlist, DBOPT_DTIME, &time);

      err = DBPutMultimesh(db_header_file, "mesh", P, (const char*const*)meshnames, meshtypes, optlist);
      if (err < 0)
      {
        amps_Printf("Error: Silo put multimesh failed %s\n", filename);
        exit(1);
      }

      DBFreeOptlist(optlist);

      optlist = DBMakeOptlist(2);
      DBAddOption(optlist, DBOPT_CYCLE, &step);
      DBAddOption(optlist, DBOPT_DTIME, &time);

      /* Multimesh information */
      err = DBPutMultivar(db_header_file, variable_name, P, (const char*const*)varnames, vartypes, optlist);
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
      err = DBWrite(db_header_file, "origin", origin, dims, 1, DB_DOUBLE);
      if (err < 0)
      {
        amps_Printf("Error: Silo failed on DBWrite\n");
      }

      /* Write the size information */
      int size[3];
      size[0] = SubgridNX(GridSubgrid(GlobalsUserGrid, 0));
      size[1] = SubgridNY(GridSubgrid(GlobalsUserGrid, 0));
      size[2] = SubgridNZ(GridSubgrid(GlobalsUserGrid, 0));
      err = DBWrite(db_header_file, "size", size, dims, 1, DB_INT);
      if (err < 0)
      {
        amps_Printf("Error: Silo failed on DBWrite\n");
      }

      /* Write the delta information */
      double delta[3];
      delta[0] = SubgridDX(GridSubgrid(GlobalsUserGrid, 0));
      delta[1] = SubgridDY(GridSubgrid(GlobalsUserGrid, 0));
      delta[2] = SubgridDZ(GridSubgrid(GlobalsUserGrid, 0));
      err = DBWrite(db_header_file, "delta", delta, dims, 1, DB_DOUBLE);
      if (err < 0)
      {
        amps_Printf("Error: Silo failed on DBWrite\n");
      }

//        if (numGroups > 1) {
      err = DBClose(db_header_file);
//                            }
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

    /* Hand off the baton to the next processor. This winds up closing
     * the file so that the next processor that opens it can be assured
     * of getting a consistent and up to date view of the file's contents. */
    PMPIO_HandOffBaton(bat, db_file);

    /* We're done using PMPIO, so finish it off */
    PMPIO_Finish(bat);

    free(coords[0]);
    free(coords[1]);
    free(coords[2]);
  }
#else
  amps_Printf("Parflow not compiled with SILO and MPI, can't use SILO PMPIO\n");
#endif

  EndTiming(PFBTimingIndex);
}



