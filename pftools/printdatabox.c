/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/
/******************************************************************************
 * Print routines for pftools
 *
 * (C) 1993 Regents of the University of California.
 *
 * $Revision: 1.16 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/
#include "parflow_config.h"

#include <stdlib.h>

#ifdef HAVE_SILO
#include <silo.h>
#endif

#include "printdatabox.h"
#include "tools_io.h"

#include "string.h"
#include "unistd.h"

/*-----------------------------------------------------------------------
 * print a Databox in `simple' ascii format
 *-----------------------------------------------------------------------*/

void            PrintSimpleA(fp, v)
FILE           *fp;
Databox         *v;
{
   int             nx, ny, nz;

   int             m;

   double         *ptr;


   nx = DataboxNx(v);
   ny = DataboxNy(v);
   nz = DataboxNz(v);

   fprintf(fp, "%d %d %d\n", nx, ny, nz);

   ptr = DataboxCoeffs(v);
   for (m = nx * ny * nz; m--;)
      fprintf(fp, "% e\n", *ptr++);
}

/*-----------------------------------------------------------------------
 * IMF: print a Databox in `simple' ascii format -- FORCE NZ -> 1!
        (added this b/c converting CLM fluxes from silo to ascii or pfb 
         generated files w/ NZ = <problem NZ> instead of NZ = 1 for flux 
         fields)
 *-----------------------------------------------------------------------*/

void            PrintSimpleA2D(fp, v)
FILE           *fp;
Databox         *v;
{
   int             nx, ny, nz;

   int             m;

   double         *ptr;


   nx = DataboxNx(v);
   ny = DataboxNy(v);
   nz = 1;                // Force NZ -> 1 (instead of DataboxNz)

   fprintf(fp, "%d %d %d\n", nx, ny, nz);

   ptr = DataboxCoeffs(v);
   for (m = nx * ny * nz; m--;)
      fprintf(fp, "% e\n", *ptr++);
}

/*-----------------------------------------------------------------------
 * print a Databox in `simple' binary format
 *-----------------------------------------------------------------------*/

void            PrintSimpleB(fp, v)
FILE           *fp;
Databox         *v;
{
   int             nx, ny, nz;


   nx = DataboxNx(v);
   ny = DataboxNy(v);
   nz = DataboxNz(v);


   tools_WriteInt(fp, &nx, 1);
   tools_WriteInt(fp, &ny, 1);
   tools_WriteInt(fp, &nz, 1);

   tools_WriteDouble(fp, DataboxCoeffs(v), nx*ny*nz);
}

/*-----------------------------------------------------------------------
 * print a Databox in `parflow' binary format
 *-----------------------------------------------------------------------*/

void            PrintParflowB(fp, v)
FILE           *fp;
Databox        *v;
{
   double  X  = DataboxX(v);     /* These were being set to 0.0 for some
				    reason...changed to set them to the
				    values in the Databox  */
   double  Y  = DataboxY(v);
   double  Z  = DataboxZ(v);
   int     NX = DataboxNx(v);
   int     NY = DataboxNy(v);
   int     NZ = DataboxNz(v);
   double  DX = DataboxDx(v);
   double  DY = DataboxDy(v);
   double  DZ = DataboxDz(v);

   int     ns = 1;               /* num_subgrids */

   int     x  = 0;
   int     y  = 0;
   int     z  = 0;
   int     nx = NX;
   int     ny = NY;
   int     nz = NZ;
   int     rx = 1;
   int     ry = 1;
   int     rz = 1;


   tools_WriteDouble(fp, &X, 1);
   tools_WriteDouble(fp, &Y, 1);
   tools_WriteDouble(fp, &Z, 1);
   tools_WriteInt(fp, &NX, 1);
   tools_WriteInt(fp, &NY, 1);
   tools_WriteInt(fp, &NZ, 1);
   tools_WriteDouble(fp, &DX, 1);
   tools_WriteDouble(fp, &DY, 1);
   tools_WriteDouble(fp, &DZ, 1);

   tools_WriteInt(fp, &ns, 1);

   tools_WriteInt(fp, &x, 1);
   tools_WriteInt(fp, &y, 1);
   tools_WriteInt(fp, &z, 1);
   tools_WriteInt(fp, &nx, 1);
   tools_WriteInt(fp, &ny, 1);
   tools_WriteInt(fp, &nz, 1);
   tools_WriteInt(fp, &rx, 1);
   tools_WriteInt(fp, &ry, 1);
   tools_WriteInt(fp, &rz, 1);

   tools_WriteDouble(fp, DataboxCoeffs(v), nx*ny*nz);
}

/*-----------------------------------------------------------------------
 * print a Databox in AVS .fld format
 *-----------------------------------------------------------------------*/

void            PrintAVSField(fp, v)
FILE           *fp;
Databox        *v;
{
   double  X  = DataboxX(v);
   double  Y  = DataboxY(v);
   double  Z  = DataboxZ(v);
   int     NX = DataboxNx(v);
   int     NY = DataboxNy(v);
   int     NZ = DataboxNz(v);
   double  DX = DataboxDx(v);
   double  DY = DataboxDy(v);
   double  DZ = DataboxDz(v);
   float  coord[6];

   coord[0] = X; coord[2] = Y; coord[4] = Z;
   coord[1] = X+(NX-1)*DX; coord[3] = Y+(NY-1)*DY; coord[5] = Z+(NZ-1)*DZ;

   fprintf(fp,"# AVS\n");
   fprintf(fp,"ndim=3\n");
   fprintf(fp,"dim1=%d\n",NX);
   fprintf(fp,"dim2=%d\n",NY);
   fprintf(fp,"dim3=%d\n",NZ);
   fprintf(fp,"nspace=3\n");
   fprintf(fp,"veclen=1\n");
   fprintf(fp,"data=double\n");
   fprintf(fp,"field=uniform\n");

   /* print the necessary two ^L's */
   fprintf(fp,"%c%c",0xC,0xC);

   fwrite(DataboxCoeffs(v),sizeof(double),NX*NY*NZ,fp);
   fwrite(coord,sizeof(*coord),6,fp);
}



#ifdef HAVE_HDF4
/*-----------------------------------------------------------------------
 * print a Databox in HDF SDS format
 *-----------------------------------------------------------------------*/

int             PrintSDS(filename, type,  v)
char            *filename;
int             type;
Databox         *v;
{
  int32  sd_id;
  int32  sds_id;

  int32           dim[3];
  int32           edges[3];
  int32           start[3];
  
  VOIDP          data;
  
  double         *double_ptr;
  int           i;
  int           z;

  
  dim[0] = DataboxNz(v);
  dim[1] = DataboxNy(v);
  dim[2] = DataboxNx(v);

  edges[0] = 1;
  edges[1] = DataboxNy(v);
  edges[2] = DataboxNx(v);

  start[0] = start[1] = start[2] = 0;
  
  double_ptr = DataboxCoeffs(v);

  sd_id = SDstart(filename, DFACC_CREATE);

  sds_id = SDcreate(sd_id, "ParFlow Data", type, 3, dim);

  switch (type) {
  case DFNT_FLOAT32: 
    {
      float32  *convert_ptr;
      
      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(float32))) == NULL)
         return (0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (float32 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_FLOAT64: 
    {
      float64  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(float64))) == NULL)
         return (0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (float64 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_INT8: 
    {
      int8  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(int8))) == NULL)
         return (0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (int8 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_UINT8: 
    {
      uint8  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(uint8))) == NULL)
         return (0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (uint8 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_INT16: 
    {
      int16  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(int16))) == NULL)
         return (0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (int16 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_UINT16: 
    {
      uint16  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(uint16))) == NULL)
         return(0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (uint16 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_INT32: 
    {
      int32  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(int32))) == NULL)
         return(0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (int32 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }
  case DFNT_UINT32: 
    {
      uint32  *convert_ptr;

      if( (data = (VOIDP)malloc(dim[1]*dim[2] * sizeof(uint32))) == NULL)
         return(0);

      for(z=0; z < DataboxNz(v); z++) 
	{
	  convert_ptr = (uint32 *)data;
	  for(i=dim[1]*dim[2]; i--;)
	    *convert_ptr++ = *double_ptr++;

	  start[0] = z;
	  SDwritedata(sds_id, start , NULL, edges, data);
	}
      
      free(data);
      break;
    }

  }
  
  SDendaccess(sds_id);

  SDend(sd_id);

  return(1);
}
#endif

/*-----------------------------------------------------------------------
 * print a Databox in `vizamrai' format
 *-----------------------------------------------------------------------*/

void            PrintVizamrai(fp, v)
FILE           *fp;
Databox        *v;
{
   int     NX = DataboxNx(v);
   int     NY = DataboxNy(v);
   int     NZ = DataboxNz(v);
   double  DX = DataboxDx(v);
   double  DY = DataboxDy(v);
   double  DZ = DataboxDz(v);

   int     nx = NX;
   int     ny = NY;
   int     nz = NZ;

   int FileFormat = -5;

   double dzero = 0.0;

   int izero = 0;
   int ione = 1;

   double temp;

   int itemp;

   int string_size = 8;
   char *string = "variable";


   tools_WriteInt(fp, &FileFormat, 1);             /* File Format */
   tools_WriteDouble(fp, &dzero, 1);           /* Time stamp (we don't have this */

   tools_WriteInt(fp, &ione, 1);              /* Number of patches */

   tools_WriteInt(fp, &izero, 1);              /* Double format */

   tools_WriteInt(fp, &ione, 1);              /* One variable */

   tools_WriteInt(fp, &string_size, 1);
   fwrite(string, sizeof(char), 8, fp);

   tools_WriteInt(fp, &ione, 1);              /* Number Of Levels */

   tools_WriteInt(fp, &ione, 1);              /* Number Of PatchBoundaries */

   tools_WriteDouble(fp, &dzero, 1);           /* Lower corner patch bondary */
   tools_WriteDouble(fp, &dzero, 1);           
   tools_WriteDouble(fp, &dzero, 1);           

   temp = DX * NX;
   tools_WriteDouble(fp, &temp, 1);           /* upper corner patch boundary */

   temp = DY * NY;
   tools_WriteDouble(fp, &temp, 1);

   temp = DZ * NZ;
   tools_WriteDouble(fp, &temp, 1);

   tools_WriteInt(fp, &izero, 1);                /* Patch level */

   /* For each patch */
   

   tools_WriteInt(fp, &izero, 1);                /* Patch Level */

   tools_WriteInt(fp, &izero, 1);                /* Lower Index Space */
   tools_WriteInt(fp, &izero, 1);
   tools_WriteInt(fp, &izero, 1);

   
   itemp= NX - 1;
   tools_WriteInt(fp, &itemp, 1);                /* Upper Index Space */

   itemp= NY - 1;
   tools_WriteInt(fp, &itemp, 1);                /* Upper Index Space */

   itemp= NZ - 1;
   tools_WriteInt(fp, &itemp, 1);                /* Upper Index Space */

   tools_WriteDouble(fp, &DX, 1);
   tools_WriteDouble(fp, &DY, 1);
   tools_WriteDouble(fp, &DZ, 1);

   temp = DX / 2.0;
   tools_WriteDouble(fp, &temp, 1);
   temp = DY / 2.0;
   tools_WriteDouble(fp, &temp, 1);
   temp = DZ / 2.0;
   tools_WriteDouble(fp, &temp, 1);

   tools_WriteDouble(fp, DataboxCoeffs(v), nx*ny*nz);
}

/*-----------------------------------------------------------------------
 * print a Databox in `silo' format
 *-----------------------------------------------------------------------*/


void            PrintSilo(
   char           *filename,
   Databox        *v)
{

#ifdef HAVE_SILO
   double   X = DataboxX(v);
   double   Y = DataboxY(v);
   double   Z = DataboxZ(v);
   int     NX = DataboxNx(v);
   int     NY = DataboxNy(v);
   int     NZ = DataboxNz(v);
   double  DX = DataboxDx(v);
   double  DY = DataboxDy(v);
   double  DZ = DataboxDz(v);

   double  *x, *y, *z;
   double *coords[3];
   DBfile *db;
   int dims[3];
   int ndims;
   int err;

   int i;

   char *current_path = NULL;
   char *path = NULL;
   char *slash = strchr(filename, '/');

   if(slash) {
      path = malloc(MAXPATHLEN);
      strncpy(path, filename, slash - filename);
      path[slash - filename] = 0;
      filename = strdup(slash + 1);
   } else {
      filename = strdup(filename);
   }

   if(path) { 	
      current_path = malloc(MAXPATHLEN);
      getwd(current_path);
      chdir(path);
   }

    dims[0] = NX+1;
    dims[1] = NY+1;
    dims[2] = NZ+1;
    ndims = 3;

    x = (double*) malloc(sizeof(double) * dims[0]);
    y = (double*) malloc(sizeof(double) * dims[1]);
    z = (double*) malloc(sizeof(double) * dims[2]);
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;

    for (i = 0 ; i < dims[0]; i++)
    {
       x[i] = X + ((double)(i) - 0.5)*DX;
    }

    for (i = 0 ; i < dims[1]; i++)
    {
       y[i] = Y + ((double)(i) - 0.5)*DY;
    }

    for (i = 0 ; i < dims[2]; i++)
    {
       z[i] = Z + ((double)(i) - 0.5)*DZ;
    }
    

    db = DBCreate(filename, DB_CLOBBER, DB_LOCAL, filename, DB_PDB);

    /* Write the origin information */
    int db_dims[1];
    db_dims[0] = 3;

    double origin[3];
    origin[0] = X;
    origin[1] = Y;
    origin[2] = Z;

    err = DBWrite(db, "origin", origin, db_dims, 1, DB_DOUBLE);
    if(err < 0) {
       printf("Error: Silo failed on DBWrite\n");
    } 

    /* Write the size information */
    int size[3];
    size[0] = NX;
    size[1] = NY;
    size[2] = NZ;
    err = DBWrite(db, "size", size, db_dims, 1, DB_INT);
    if(err < 0) {
       printf("Error: Silo failed on DBWrite\n");
    } 
    
    /* Write the delta information */
    double delta[3];
    delta[0] = DX;
    delta[1] = DY;
    delta[2] = DZ;
    err = DBWrite(db, "delta", delta, db_dims, 1, DB_DOUBLE);
    if(err < 0) {
       printf("Error: Silo failed on DBWrite\n");
    } 

    DBPutQuadmesh(db, "mesh", NULL, (float **)coords, dims, ndims, DB_DOUBLE, DB_COLLINEAR, NULL);

    DBPutQuadvar1(db, "variable", "mesh", (float *)DataboxCoeff(v, 0, 0, 0), size, ndims, NULL,0,
                  DB_DOUBLE, DB_ZONECENT, NULL);

    DBClose(db);

   if(path) { 	
      free(path);
   }

   if(current_path) { 	
      chdir(current_path);
      free(current_path);
   }

   free(filename);

#endif
}




