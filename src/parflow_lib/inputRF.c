/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
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

#include <string.h>

#include "parflow.h"




/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   char *filename;

} PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid   *grid;
   double *temp_data;

   /* Instance data */
   Vector *tmpRF;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * InputRF
 *--------------------------------------------------------------------------*/

void    InputRF(geounit, gr_geounit, field, cdata)
GeomSolid    *geounit;
GrGeomSolid  *gr_geounit;
Vector       *field;
RFCondData   *cdata;
{
   /*-----------------------------------------------------------------------
    * Local variables 
    *-----------------------------------------------------------------------*/
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   Vector    *tmpRF         = (instance_xtra -> tmpRF);

   /* Grid parameters */
   Grid           *grid = (instance_xtra -> grid);
   Subgrid        *subgrid;

   Subvector      *field_sub;
   Subvector      *tmpRF_sub;
   double         *fieldp;
   double         *tmpRFp;

   /* Counter, indices, flags */
   int             subgrid_loop;
   int		   i, j, k;
   int             ix, iy, iz;
   int             nx, ny, nz;
   int             r;
   int             indexfp,indextp;
  
   ReadPFBinary((public_xtra -> filename),tmpRF);

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
}


/*--------------------------------------------------------------------------
 * InputRFInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *InputRFInitInstanceXtra(grid, temp_data)
Grid      *grid;
double    *temp_data;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;


   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `grid'
    *-----------------------------------------------------------------------*/

   if ( grid != NULL)
   {
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
         FreeTempVector(instance_xtra -> tmpRF);
      }

      /* set new data */
      (instance_xtra -> grid) = grid;

      (instance_xtra -> tmpRF)        = NewTempVector(grid, 1, 1);
   }


   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;

      SetTempVectorData((instance_xtra -> tmpRF), temp_data);
      temp_data += SizeOfVector(instance_xtra -> tmpRF);

   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * InputRFFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  InputRFFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      FreeTempVector(instance_xtra -> tmpRF);
      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * InputRFNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *InputRFNewPublicXtra(char *geom_name)
{
   /* Local variables */
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char                 key[IDB_MAX_KEY_LEN];

   /* Allocate space for the public_xtra structure */
   public_xtra = ctalloc(PublicXtra, 1);


   sprintf(key, "Geom.%s.Perm.FileName", geom_name);
   public_xtra -> filename = GetString(key);

   PFModulePublicXtra(this_module) = public_xtra;

   return this_module;
}


/*--------------------------------------------------------------------------
 * InputRFFreePublicXtra
 *--------------------------------------------------------------------------*/

void  InputRFFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


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
   PFModule      *this_module = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);
   int           size; 

/* Set size equal to local TempData size */
   size = SizeOfVector(instance_xtra -> tmpRF);

   return (size);
}
