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
* Read routines for pftools.
*
* (C) 1993 Regents of the University of California.
*
*-----------------------------------------------------------------------------
* $Revision: 1.21 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow_config.h"

#include "readdatabox.h"
#include "tools_io.h"

#ifdef HAVE_SILO
#include "silo.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#define round(x) ((x) >= 0 ? (double)((x) + 0.5) : (double)((x) - 0.5))

/*-----------------------------------------------------------------------
 * read a SILO file
 *-----------------------------------------------------------------------*/

Databox         *ReadSilo(char *filename, double default_value)
{
#ifdef HAVE_SILO
  Databox         *v;

  double X, Y, Z;
  int NX, NY, NZ;
  double DX, DY, DZ;

  int x, y, z;
  int nx, ny, nz;

  int j, k;

  double         *ptr;

  int err = -1;

  DBfile         *db;
  char *current_path = NULL;
  char *path = NULL;
  char *slash = strchr(filename, '/');
  float *localcoords[3];

  if (slash)
  {
    path = (char*)malloc(MAXPATHLEN);
    strncpy(path, filename, slash - filename);
    path[slash - filename] = 0;
    filename = strdup(slash + 1);
  }
  else
  {
    filename = strdup(filename);
  }

  if (path)
  {
    current_path = (char*)malloc(MAXPATHLEN);
    getcwd(current_path, MAXPATHLEN);
    chdir(path);
  }

//   db = DBOpen(filename, DB_PDB, DB_READ);
  db = DBOpen(filename, DB_UNKNOWN, DB_READ);
  if (db == NULL)
  {
    printf("Failed to open SILO file %s\n", filename);
    return NULL;
  }

  double origin[3];
  err = DBReadVar(db, "origin", &origin);
  if (err < 0)
  {
    printf("Failed to read meta data\n");
    return NULL;
  }
  X = origin[0];
  Y = origin[1];
  Z = origin[2];

  int size[3];
  err = DBReadVar(db, "size", &size);
  if (err < 0)
  {
    printf("Failed to read meta data\n");
    return NULL;
  }
  NX = size[0];
  NY = size[1];
  NZ = size[2];

  double delta[3];
  err = DBReadVar(db, "delta", &delta);
  if (err < 0)
  {
    printf("Failed to read meta data\n");
    return NULL;
  }
  DX = delta[0];
  DY = delta[1];
  DZ = delta[2];

  /* create the new databox structure */
  if ((v = NewDataboxDefault(NX, NY, NZ, X, Y, Z, DX, DY, DZ, default_value)) == NULL)
  {
    return((Databox*)NULL);
  }

  DBtoc *toc = DBGetToc(db);
  if (toc == NULL)
  {
    printf("Error: Silo get get TOC failed for %s\n", filename);
    return NULL;
  }

  char **multivar_names = toc->multivar_names;
  int nmultivar = toc->nmultivar;

  /* Check to see if file has a multivar in it to determine file format being used */
  if (nmultivar == 0)
  {
    DBquadvar  *var = DBGetQuadvar(db, "variable");
    if (var == NULL)
    {
      printf("Error: Silo failed to get quadvar %s \n", "variable");
      return NULL;
    }

    memcpy(DataboxCoeff(v, 0, 0, 0), var->vals[0], NX * NY * NZ * sizeof(double));

    DBFreeQuadvar(var);
  }
  else
  {
    DBmultivar *multivar = DBGetMultivar(db, multivar_names[0]);
    if (multivar == NULL)
    {
      printf("Error: Silo get multivar failed for %s\n", multivar_names[0]);
      return NULL;
    }


    int nvars = multivar->nvars;
    char **varnames = multivar->varnames;

    int i;

    int m;
    for (m = 0; m < nvars; m++)
    {
      char *proc_filename;
      char *seperator = ":";

      proc_filename = strtok(varnames[m], seperator);
      char *proc_varname;
      proc_varname = strtok(NULL, seperator);
      /*  printf("multivar nvar:  %d  \n", nvars);
       * printf("multivar proc_varname:  %s  \n", proc_varname);
       * printf("multivar proc_filename:  %s  \n", proc_filename); */

      if (proc_filename == NULL)
      {
        printf("Error malformed multivar name %s in SILO file \n", varnames[m]);
        return NULL;
      }

      DBfile *proc_db;
//       proc_db = DBOpen(proc_filename, DB_PDB, DB_READ);
      proc_db = DBOpen(proc_filename, DB_UNKNOWN, DB_READ);
      if (db == NULL)
      {
        printf("Failed to open SILO file %s\n", filename);
        return NULL;
      }

      if (proc_varname == NULL)
      {
        printf("Error malformed multivar name %s in SILO file \n", varnames[m]);
        return NULL;
      }

      DBquadvar  *var = DBGetQuadvar(proc_db, proc_varname);
      if (var == NULL)
      {
        printf("Error: Silo failed to get quadvar %s \n", varnames[m]);
        return NULL;
      }

      /* now we need to get the mesh, for PMPIO compat */
      DBquadmesh  *mesh = DBGetQuadmesh(proc_db, var->meshname);
      if (mesh == NULL)
      {
        printf("Error: Silo failed to get quadmesh %s \n", varnames[m]);
        return NULL;
      }

      nx = var->dims[0];
      ny = var->dims[1];
      nz = var->dims[2];

      /* Casting here is a dangerous */
      localcoords[0] = (float*)mesh->coords[0];
      localcoords[1] = (float*)mesh->coords[1];
      localcoords[2] = (float*)mesh->coords[2];

      int index_origin[3];
      err = DBReadVar(proc_db, "index_origin", &index_origin);
      if (err < 0)
      {
        printf("Failed to read meta data\n");
        return NULL;
      }

      /* becuase of multi or single file compatibility we need to
       * grab the actual mesh from that variable.  Then we need to
       * determine the origin from the mesh[0][0], [1][0] and [2][0]
       * divided by DX, DY and DZ since there are multiple origins
       * in a single file and this is now ambiguous. */

      x = index_origin[0];
      y = index_origin[1];
      z = index_origin[2];

      x = round((localcoords[0][0] - (X - DX / 2.0)) / DX);
      y = round((localcoords[1][0] - (Y - DY / 2.0)) / DY);
      z = round((localcoords[2][0] - (Z - DZ / 2.0)) / DZ);
      int index = 0;
      double *vals = (double*)var->vals[0];
      for (k = 0; k < nz; k++)
      {
        for (j = 0; j < ny; j++)
        {
          for (i = 0; i < nx; i++)
          {
            ptr = DataboxCoeff(v, x + i, y + j, z + k);
            *ptr = vals[index];
            index++;
          }
        }
      }

      DBFreeQuadvar(var);

      DBClose(proc_db);
    }

    DBFreeMultivar(multivar);
  }


  DBClose(db);



  if (path)
  {
    free(path);
  }

  if (current_path)
  {
    chdir(current_path);
    free(current_path);
  }

  free(filename);
  /*   free(localcoords[0]);
  * free(localcoords[1]);
  * free(localcoords[2]); */

  return v;
#else
  printf("Error: Silo was not used in build\n");
  return NULL;
#endif
}

/*-----------------------------------------------------------------------
 * read a binary `parflow' file
 *-----------------------------------------------------------------------*/

Databox         *ReadParflowB(
                              char * file_name,
                              double default_value)
{
  Databox         *v;

  FILE           *fp;

  double X, Y, Z;
  int NX, NY, NZ;
  double DX, DY, DZ;
  int num_subgrids;

  int x, y, z;
  int nx, ny, nz;
  int rx, ry, rz;

  int nsg, j, k;

  double         *ptr;


  /* open the input file */
  if ((fp = fopen(file_name, "rb")) == NULL)
    return NULL;

  /* read in header info */
  tools_ReadDouble(fp, &X, 1);
  tools_ReadDouble(fp, &Y, 1);
  tools_ReadDouble(fp, &Z, 1);

  tools_ReadInt(fp, &NX, 1);
  tools_ReadInt(fp, &NY, 1);
  tools_ReadInt(fp, &NZ, 1);

  tools_ReadDouble(fp, &DX, 1);
  tools_ReadDouble(fp, &DY, 1);
  tools_ReadDouble(fp, &DZ, 1);

  tools_ReadInt(fp, &num_subgrids, 1);

  /* create the new databox structure */
  if ((v = NewDataboxDefault(NX, NY, NZ, X, Y, Z, DX, DY, DZ, default_value)) == NULL)
  {
    fclose(fp);
    return((Databox*)NULL);
  }

  /* read in the databox data */
  for (nsg = num_subgrids; nsg--;)
  {
    tools_ReadInt(fp, &x, 1);
    tools_ReadInt(fp, &y, 1);
    tools_ReadInt(fp, &z, 1);

    tools_ReadInt(fp, &nx, 1);
    tools_ReadInt(fp, &ny, 1);
    tools_ReadInt(fp, &nz, 1);

    tools_ReadInt(fp, &rx, 1);
    tools_ReadInt(fp, &ry, 1);
    tools_ReadInt(fp, &rz, 1);



    for (k = 0; k < nz; k++)
      for (j = 0; j < ny; j++)
      {
        ptr = DataboxCoeff(v, x, (y + j), (z + k));
        tools_ReadDouble(fp, ptr, nx);
      }
  }

  fclose(fp);
  return v;
}


/*-----------------------------------------------------------------------
 * read a scattered binary `parflow' file
 *-----------------------------------------------------------------------*/

Databox         *ReadParflowSB(
                               char * file_name,
                               double default_value)
{
  Databox         *v;

  FILE           *fp;

  double X, Y, Z;
  int NX, NY, NZ;
  double DX, DY, DZ;
  int num_subgrids;

  int x, y, z;
  int nx, ny, nz;
  int rx, ry, rz;

  double value;

  int nsg, i, j, k, m, n;


  /* open the input file */
  if ((fp = fopen(file_name, "rb")) == NULL)
    return NULL;

  /* read in header info */
  tools_ReadDouble(fp, &X, 1);
  tools_ReadDouble(fp, &Y, 1);
  tools_ReadDouble(fp, &Z, 1);

  tools_ReadInt(fp, &NX, 1);
  tools_ReadInt(fp, &NY, 1);
  tools_ReadInt(fp, &NZ, 1);

  tools_ReadDouble(fp, &DX, 1);
  tools_ReadDouble(fp, &DY, 1);
  tools_ReadDouble(fp, &DZ, 1);

  tools_ReadInt(fp, &num_subgrids, 1);

  /* create the new databox structure */
  if ((v = NewDataboxDefault(NX, NY, NZ, X, Y, Z, DX, DY, DZ, default_value)) == NULL)
  {
    fclose(fp);
    return((Databox*)NULL);
  }

  /* read in the databox data */
  for (nsg = num_subgrids; nsg--;)
  {
    tools_ReadInt(fp, &x, 1);
    tools_ReadInt(fp, &y, 1);
    tools_ReadInt(fp, &z, 1);

    tools_ReadInt(fp, &nx, 1);
    tools_ReadInt(fp, &ny, 1);
    tools_ReadInt(fp, &nz, 1);

    tools_ReadInt(fp, &rx, 1);
    tools_ReadInt(fp, &ry, 1);
    tools_ReadInt(fp, &rz, 1);

    tools_ReadInt(fp, &n, 1);

    for (m = 0; m < n; m++)
    {
      tools_ReadInt(fp, &i, 1);
      tools_ReadInt(fp, &j, 1);
      tools_ReadInt(fp, &k, 1);
      tools_ReadDouble(fp, &value, 1);

      *DataboxCoeff(v, i, j, k) = value;
    }
  }

  fclose(fp);
  return v;
}


/*-----------------------------------------------------------------------
 * read a `simple ascii' file
 *-----------------------------------------------------------------------*/

Databox         *ReadSimpleA(
                             char * file_name,
                             double default_value)
{
  Databox         *v;

  FILE           *fp;

  int nx, ny, nz;

  int m;

  double         *ptr;


  /* open the input file */
  if ((fp = fopen(file_name, "r")) == NULL)
    return((Databox*)NULL);

  /* read in header info */
  fscanf(fp, "%d %d %d", &nx, &ny, &nz);

  /* create the new databox structure */
  if ((v = NewDataboxDefault(nx, ny, nz, 0, 0, 0, 0, 0, 0, default_value)) == NULL)
  {
    fclose(fp);
    return((Databox*)NULL);
  }

  /* read in the databox data */
  ptr = DataboxCoeffs(v);
  for (m = nx * ny * nz; m--;)
    fscanf(fp, "%lf", ptr++);

  fclose(fp);
  return v;
}

/*-----------------------------------------------------------------------
 * read a `real scattered ascii' file
 *-----------------------------------------------------------------------*/

Databox        *ReadRealSA(
                           char * file_name,
                           double default_value)
{
  Databox         *v;

  FILE           *fp;

  double x0, y0, z0;
  int nx, ny, nz;
  double dx, dy, dz;

  double x, y, z, value;
  int ix, iy, iz;

  /* open the input file */
  if ((fp = fopen(file_name, "r")) == NULL)
    return((Databox*)NULL);

  /* read in header info */
  fscanf(fp, "%lf %lf %lf", &x0, &y0, &z0);
  fscanf(fp, "%d %d %d", &nx, &ny, &nz);
  fscanf(fp, "%lf %lf %lf", &dx, &dy, &dz);

  /* create the new databox structure */
  if ((v = NewDataboxDefault(nx, ny, nz, 0, 0, 0, 0, 0, 0, default_value)) == NULL)
  {
    fclose(fp);
    return((Databox*)NULL);
  }

  while (fscanf(fp, "%lf %lf %lf %lf", &x, &y, &z, &value) != EOF)
  {
    ix = (int)((x - x0) / dx + 0.5);
    iy = (int)((y - y0) / dy + 0.5);
    iz = (int)((z - z0) / dz + 0.5);
    if ((ix > -1) && (ix < nx) &&
        (iy > -1) && (iy < ny) &&
        (iz > -1) && (iz < nz))
    {
      *DataboxCoeff(v, ix, iy, iz) = value;
    }
  }

  fclose(fp);
  return v;
}

/*-----------------------------------------------------------------------
 * read a `simple binary' file
 *-----------------------------------------------------------------------*/

Databox         *ReadSimpleB(
                             char * file_name,
                             double default_value)
{
  Databox         *v;

  FILE           *fp;

  int nx, ny, nz;

  double         *ptr;


  /* open the input file */
  if ((fp = fopen(file_name, "rb")) == NULL)
    return((Databox*)NULL);

  /* read in header info */
  tools_ReadInt(fp, &nx, 1);
  tools_ReadInt(fp, &ny, 1);
  tools_ReadInt(fp, &nz, 1);

  /* create the new databox structure */
  if ((v = NewDataboxDefault(nx, ny, nz, 0, 0, 0, 0, 0, 0, default_value)) == NULL)
  {
    fclose(fp);
    return((Databox*)NULL);
  }

  /* read in the databox data */
  ptr = DataboxCoeffs(v);

  tools_ReadDouble(fp, ptr, nx * ny * nz);

  fclose(fp);
  return v;
}


/*-----------------------------------------------------------------------
 * read an AVS Field file
 *-----------------------------------------------------------------------*/

/* Note: These functions will read an AVS field file when the file satisfies
 * the following conditions:
 *    1. it is in the correct format
 *    2. field = uniform or rectilinear
 *    3. the binary data is in the file itself (i.e. not included from
 *       another file), after the two ^L's at the end of the header
 *    4. data = byte, int, float, or double
 *
 * Rectilinear data is munged into a uniform format by maintaining correct
 * coordinates on the faces of the grid, and disregarding correctness within
 * the grid.
 *
 * For files with veclen > 1, the component used is min(COMPONENT,veclen-1).
 * COMPONENT is defined below.
 */

#define COMPONENT 0

/* parse header into count token/value pairs (must be free'd afterwards) */
static int parse_fld_header(char *header, int *count,
                            char ***tokens, char ***values)
{
  char *ptr;
  int newlines = 0;

  *count = 0;

  for (ptr = header; *ptr; ptr++)
    if (*ptr == '\n')
      newlines++;

  *tokens = (char**)malloc(newlines * sizeof(char **));
  *values = (char**)malloc(newlines * sizeof(char **));

  if ((ptr = strtok(header, "\n")) != NULL)
  {
    do
    {
      char *token, *value;
      char *token_ptr, *value_ptr;
      char *line = ptr;
      int len = strlen(ptr);

      /* kill comments */
      ptr = line;
      while (*ptr && *ptr != '#')
        ptr++;
      *ptr = '\0';

      /* get rid of trailing whitespace */
      ptr = line + strlen(line) - 1;
      while (ptr >= line && isspace(*ptr))
        ptr--;
      *++ptr = '\0';

      if (*line == '\0')
        continue;

      token = (char*)malloc(len + 1);
      token_ptr = token;
      value = (char*)malloc(len + 1);
      value_ptr = value;

      ptr = line;
      while (isspace(*ptr))
        ptr++;
      while (!isspace(*ptr) && *ptr != '=')
        *token_ptr++ = *ptr++;
      *token_ptr = '\0';
      while (isspace(*ptr))
        ptr++;
      if (*ptr != '=')
      {
        return(0);
      }
      ptr++;
      while (isspace(*ptr))
        ptr++;
      while (*ptr)
        *value_ptr++ = *ptr++;
      *value_ptr = '\0';

      (*tokens)[*count] = token;
      (*values)[*count] = value;
      (*count)++;
    }
    while ((ptr = strtok(NULL, "\n")) != NULL);
  }

  return(1);
}

static void free_pairs(int count, char **token, char **value)
{
  int i;

  for (i = 0; i < count; i++)
  {
    free(token[i]);
    free(value[i]);
  }
  free(token);
  free(value);
}

#define HEADER_SIZE_INIT 512
#define HEADER_SIZE_INC 512

Databox         *ReadAVSField(
                              char * file_name,
                              double default_value)
{
  Databox         *v;

  FILE           *fp;

  char *header = NULL;
  int header_size = 0;
  int count;
  int tokens;
  int c;
  char **token, **value;
  char begin[64];

  int i, j;
  int NX = -1, NY = -1, NZ = -1;
  int datatype = -1;    /* 0 = byte, 1 = int, 2 = float, 3 = double */
  int is_rect = 0;
  int ndim = -1;
  int veclen = 1;
  double min[3], max[3];

  double X, Y, Z;
  double DX, DY, DZ;

  double          *ptr;

  /* open the input file */
  if ((fp = fopen(file_name, "rb")) == NULL)
    return NULL;

  /* read in header info */
  fgets(begin, 64, fp);
  if (strncmp(begin, "# AVS", 5) != 0)
    return NULL;
  while (begin[strlen(begin) - 1] != '\n')
    fgets(begin, 64, fp);

  header_size = HEADER_SIZE_INIT;
  header = (char*)malloc(header_size);

  count = 0;
  while ((c = fgetc(fp)) != EOF)
  {
    if (count == header_size)
    {
      header_size += HEADER_SIZE_INC;
      header = (char*)realloc(header, header_size);
    }
    if (c == 0xC)
    {
      /* end of header */
      header[count] = '\0';
      break;
    }
    else
      header[count++] = c;
  }
  fgetc(fp);    /* == 0xC */

  if (parse_fld_header(header, &tokens, &token, &value) != 1)
    return NULL;

  /* extract relevant info from token/value pairs */
  for (i = 0; i < tokens; i++)
  {
    if (strcmp(token[i], "dim1") == 0)
      NX = atoi(value[i]);
    else if (strcmp(token[i], "dim2") == 0)
      NY = atoi(value[i]);
    else if (strcmp(token[i], "dim3") == 0)
      NZ = atoi(value[i]);

    else if (strcmp(token[i], "ndim") == 0)
    {
      switch (ndim = atoi(value[i]))
      {
        case 1: NY = 1;

        case 2: NZ = 1;

        case 3: break;

        default: free_pairs(tokens, token, value); return NULL;
      }
    }
    else if (strcmp(token[i], "veclen") == 0)
    {
      if ((veclen = atoi(value[i])) < 1)
      {
        free_pairs(tokens, token, value); return NULL;
      }
    }
    else if (strcmp(token[i], "field") == 0 && strcmp(value[i], "uniform") != 0)
    {
      if (strcmp(value[i], "rectilinear") == 0)
        /* treat rectilinear grid as uniform to convert */
        is_rect = 1;
      else
      {
        free_pairs(tokens, token, value); return NULL;
      }
    }
    else if (strcmp(token[i], "data") == 0)
    {
      if (strcmp(value[i], "byte") == 0)
        datatype = 0;
      else if (strcmp(value[i], "int") == 0 || strcmp(value[i], "integer") == 0)
        datatype = 1;
      else if (strcmp(value[i], "float") == 0 || strcmp(value[i], "single") == 0)
        datatype = 2;
      else if (strcmp(value[i], "double") == 0)
        datatype = 3;
      else
      {
        free_pairs(tokens, token, value); return NULL;
      }
    }
  }

  free_pairs(tokens, token, value);
  free(header);

  if (NX == -1 || NY == -1 || NZ == -1)
  {
    return NULL;
  }
  if (datatype == -1)
  {
    /* datatype was not set */
    return NULL;
  }

  /* create the new databox structure */
  /* set X, Y, Z, DX, DY, DZ to 0 initially; will calculate and set later */
  if ((v = NewDataboxDefault(NX, NY, NZ, 0, 0, 0, 0, 0, 0, default_value)) == NULL)
  {
    fclose(fp);
    return((Databox*)NULL);
  }

  {
    /* get data values */
    char *barray;
    int *iarray;
    float *farray;
    double *darray;

    barray = (char*)malloc(veclen);
    iarray = (int*)malloc(veclen * sizeof(int));
    farray = (float*)malloc(veclen * sizeof(float));
    darray = (double*)malloc(veclen * sizeof(double));

    ptr = DataboxCoeffs(v);
    for (i = 0; i < NX * NY * NZ; i++)
    {
      switch (datatype)
      {
        case 0:
        {
          fread(barray, sizeof(char), veclen, fp);
          for (j = 0; j < veclen; j++)
            darray[j] = barray[j];
          break;
        }

        case 1:
        {
          fread(iarray, sizeof(int), veclen, fp);
          for (j = 0; j < veclen; j++)
            darray[j] = iarray[j];
          break;
        }

        case 3:
        {
          fread(darray, sizeof(double), veclen, fp);
          break;
        }

        case 2:
        default:
        {
          fread(farray, sizeof(float), veclen, fp);
          for (j = 0; j < veclen; j++)
            darray[j] = farray[j];
          break;
        }
      }
      /* use first component by default */
      ptr[i] = darray[0];
    }
  }

  if (is_rect)
  {
    /* hack in rectilinear data by keeping the min and max coordinates */
    float *f;
    int xcoords, ycoords, zcoords;
    int coords;

    xcoords = NX; ycoords = NY; zcoords = NZ;
    switch (ndim)
    {
      case 1: ycoords = 0;

      case 2: zcoords = 0; break;

      case 3: break;

      default: return NULL;
    }
    coords = xcoords + ycoords + zcoords;

    f = (float*)malloc((coords) * sizeof(float));
    fread(f, sizeof(float), coords, fp);
    min[0] = f[0]; max[0] = f[xcoords - 1];
    if (ndim == 2)
      min[2] = max[2] = 0;
    else
      min[2] = f[xcoords + ycoords];
    max[2] = f[xcoords + ycoords + zcoords - 1];
    if (ndim == 1)
      min[1] = max[1] = 0;
    else
      min[1] = f[xcoords];
    max[1] = f[xcoords + ycoords - 1];

    free(f);
  }
  else
  {
    /* true uniform data */
    float f[6];

    fread(f, sizeof(float), 6, fp);
    min[0] = f[0]; min[1] = f[2]; min[2] = f[4];
    max[0] = f[1]; max[1] = f[3]; max[2] = f[5];
  }

  X = min[0]; Y = min[1]; Z = min[2];
  /* set DX/DY/DZ to 1 if grid is only 1 wide in that dimension */
  if (NX != 1)
    DX = (max[0] - min[0]) / (NX - 1);
  else
    DX = 1;
  if (NY != 1)
    DY = (max[1] - min[1]) / (NY - 1);
  else
    DY = 1;
  if (NZ != 1)
    DZ = (max[2] - min[2]) / (NZ - 1);
  else
    DZ = 1;

  DataboxX(v) = X;
  DataboxY(v) = Y;
  DataboxZ(v) = Z;
  DataboxDx(v) = DX;
  DataboxDy(v) = DY;
  DataboxDz(v) = DZ;

  fclose(fp);
  return v;
}

#undef COMPONENT
#undef HEADER_SIZE_INIT
#undef HEADER_SIZE_INC


#ifdef HAVE_HDF4
/*-----------------------------------------------------------------------
 * read an HDF file
 *-----------------------------------------------------------------------*/

Databox         *ReadSDS(char *filename, int ds_num,
                         double default_value)
{
  Databox         *v;

  int32 dim[MAX_VAR_DIMS];
  int32 edges[3];
  int32 start[3];
  int i;
  int z;

  int32 type;

  char name[MAX_NC_NAME];

  int32 sd_id;
  int32 sds_id;


  int32 rank, nt, nattrs;

  int nx, ny, nz;

  int m;

  double         *double_ptr;

  sd_id = SDstart(filename, DFACC_RDONLY);

  sds_id = SDselect(sd_id, ds_num);

  SDgetinfo(sds_id, name, &rank, dim, &type, &nattrs);

  start[0] = start[1] = start[2] = 0;


  /* create the new databox structure */
  if ((v = NewDatabox(dim[2], dim[1], dim[0], 0, 0, 0, 0, 0, 0, default_value)) == NULL)
    return((Databox*)NULL);


  double_ptr = DataboxCoeffs(v);

  edges[0] = 1;
  edges[1] = DataboxNy(v);
  edges[2] = DataboxNx(v);

  switch (type)
  {
    case DFNT_FLOAT32:
    {
      float32  *convert_ptr, *data;

      if ((data = convert_ptr = (float32*)malloc(dim[1] * dim[2] *
                                                 sizeof(float32))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_FLOAT64:
    {
      float64  *convert_ptr, *data;

      if ((data = convert_ptr = (float64*)malloc(dim[1] * dim[2] *
                                                 sizeof(float64))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_INT8:
    {
      int8  *convert_ptr, *data;

      if ((data = convert_ptr = (int8*)malloc(dim[1] * dim[2] *
                                              sizeof(int8))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_UINT8:
    {
      uint8  *convert_ptr, *data;

      if ((data = convert_ptr = (uint8*)malloc(dim[1] * dim[2] *
                                               sizeof(uint8))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_INT16:
    {
      int16  *convert_ptr, *data;

      if ((data = convert_ptr = (int16*)malloc(dim[1] * dim[2] *
                                               sizeof(int16))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_UINT16:
    {
      uint16  *convert_ptr, *data;

      if ((data = convert_ptr = (uint16*)malloc(dim[1] * dim[2] *
                                                sizeof(uint16))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_INT32:
    {
      int32  *convert_ptr, *data;

      if ((data = convert_ptr = (int32*)malloc(dim[1] * dim[2] *
                                               sizeof(int32))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };

    case DFNT_UINT32:
    {
      uint32  *convert_ptr, *data;

      if ((data = convert_ptr = (uint32*)malloc(dim[1] * dim[2] *
                                                sizeof(uint32))) == NULL)
      {
        exit(1);
      }

      for (z = 0; z < dim[0]; z++)
      {
        start[0] = z;

        SDreaddata(sds_id, start, NULL, edges, data);

        convert_ptr = data;
        for (i = dim[1] * dim[2]; i--;)
          *double_ptr++ = *convert_ptr++;
      }

      free(data);
      break;
    };
  }

  SDendaccess(sds_id);

  SDend(sd_id);

  return v;
}

#endif
