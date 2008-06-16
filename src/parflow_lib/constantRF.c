/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
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
 * This file contains a parflow module that will assign a constant
 * value to the field vector that is passed in. The constant is 
 * assigned only to those points whose indicator value matches the
 * "key" integer that is passed in. Modifications may easily be made
 * to allow more than one field to be passed in and different constants
 * to be assigned to each of the field vectors that are passed in. The
 * only stipulation is that the values will be assigned to only one
 * geounit. That is, values will be assigned only to those grid points
 * whose indicators match the (single) key.
 * 
 * The constant values needed are read from standard.input. 
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  field_value;

} PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid   *grid;
   double *temp_data;

   /* Instance data */

} InstanceXtra;


/*--------------------------------------------------------------------------
 * ConstantRF
 *--------------------------------------------------------------------------*/

void    ConstantRF(geounit, gr_geounit, field, cdata)
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

   double	   value = (public_xtra -> field_value);

   Grid           *grid = (instance_xtra -> grid);
   Subgrid        *subgrid;

   Subvector      *field_sub;
   double         *fieldp;

   int             subgrid_loop;
   int		   i, j, k;
   int             ix, iy, iz;
   int             nx, ny, nz;
   int             r;
   int             index;

   /*
    * Note: the cdata argument is not used here.
    * It's there only for consistency with the other
    * random field functions (turning_bandsRF and pgsRF).
    */


   /*-----------------------------------------------------------------------
    * Assign constant values to field
    *-----------------------------------------------------------------------*/

   for (subgrid_loop = 0; subgrid_loop < GridNumSubgrids(grid); subgrid_loop++)
   {
      subgrid = GridSubgrid(grid, subgrid_loop);
      field_sub = VectorSubvector(field, subgrid_loop);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      /* RDF: assume resolution is the same in all 3 directions */
      r = SubgridRX(subgrid);

      fieldp = SubvectorData(field_sub);
      GrGeomInLoop(i, j, k, gr_geounit, r, ix, iy, iz, nx, ny, nz,
      {
	 index = SubvectorEltIndex(field_sub, i, j, k);

	 fieldp[index] = value;
      });
   }
}


/*--------------------------------------------------------------------------
 * ConstantRFInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *ConstantRFInitInstanceXtra(grid, temp_data)
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

      /* set new data */
      (instance_xtra -> grid) = grid;
   }

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * ConstantRFFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  ConstantRFFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * ConstantRFNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ConstantRFNewPublicXtra(char *geom_name)
{
   /* Local variables */
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char key[IDB_MAX_KEY_LEN];

   /* Allocate space for the public_xtra structure */
   public_xtra = ctalloc(PublicXtra, 1);

   sprintf(key, "Geom.%s.Perm.Value", geom_name);
   (public_xtra -> field_value) = GetDouble(key);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * ConstantRFFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ConstantRFFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if (public_xtra)
   {
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * ConstantRFSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ConstantRFSizeOfTempData()
{
   return 0;
}
