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
 * Routines to help with boundary conditions
 *
 * NOTE :
 *        This routine assumes that the domain structure was set up with
 *        surfaces that are rectilinear areas lying in a plane.
 * 
 *****************************************************************************/

#include "parflow.h"
#include "problem_bc.h"

/*--------------------------------------------------------------------------
 * NewBCStruct
 *--------------------------------------------------------------------------*/

BCStruct *NewBCStruct(subgrids, gr_domain,
		      num_patches, patch_indexes, bc_types, values)
SubgridArray *subgrids;
GrGeomSolid  *gr_domain;
int           num_patches;
int          *patch_indexes;
int          *bc_types;
double     ***values;
{
   BCStruct       *new;


   new = talloc(BCStruct, 1);

   (new -> subgrids)      = subgrids;
   (new -> gr_domain)     = gr_domain;
   (new -> num_patches)   = num_patches;
   (new -> patch_indexes) = patch_indexes;
   (new -> bc_types)      = bc_types;
   (new -> values)        = values;

   return new;
}

/*--------------------------------------------------------------------------
 * FreeBCStruct
 *--------------------------------------------------------------------------*/

void      FreeBCStruct(bc_struct)
BCStruct *bc_struct;
{
   double  ***values;

   int        ipatch, is;


   values = BCStructValues(bc_struct);
   if (values)
   {
      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
         ForSubgridI(is, BCStructSubgrids(bc_struct))
         {
	    tfree(values[ipatch][is]);
         }
         tfree(values[ipatch]);
      }
      tfree(values);
   }

   tfree(bc_struct);
}
