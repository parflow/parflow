/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.16 $
 *********************************************************************EHEADER*/
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

#include "printdatabox.h"
#include "tools_io.h"

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



#ifdef PF_HAVE_HDF
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

