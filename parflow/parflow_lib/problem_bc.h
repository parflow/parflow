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
 *****************************************************************************/

#ifndef _PROBLEM_BC_HEADER
#define _PROBLEM_BC_HEADER

#define DirichletBC  0
#define FluxBC       1
#define OverlandBC   2   //sk 
 
/*----------------------------------------------------------------
 * BCStruct structure
 *----------------------------------------------------------------*/

typedef struct
{
   SubgridArray    *subgrids;  /* subgrids that BC data is defined on */

   GrGeomSolid     *gr_domain;

   int              num_patches;
   int             *patch_indexes; /* num_patches patch indexes */
   int             *bc_types;         /* num_patches BC types */

   double        ***values;  /* num_patches x num_subgrids data arrays */

} BCStruct;


/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define BCStructSubgrids(bc_struct)           ((bc_struct) -> subgrids)
#define BCStructNumSubgrids(bc_struct) \
SubgridArrayNumSubgrids(BCStructSubgrids(bc_struct))
#define BCStructGrDomain(bc_struct)           ((bc_struct) -> gr_domain)
#define BCStructNumPatches(bc_struct)         ((bc_struct) -> num_patches)
#define BCStructPatchIndexes(bc_struct)       ((bc_struct) -> patch_indexes)
#define BCStructBCTypes(bc_struct)            ((bc_struct) -> bc_types)
#define BCStructValues(bc_struct)             ((bc_struct) -> values)
#define BCStructSubgrid(bc_struct, i) \
SubgridArraySubgrid(BCStructSubgrids(bc_struct), i)
#define BCStructPatchIndex(bc_struct, p)      ((bc_struct) -> patch_indexes[p])
#define BCStructBCType(bc_struct, p)          ((bc_struct) -> bc_types[p])
#define BCStructPatchValues(bc_struct, p, s)  ((bc_struct) -> values[p][s])

/*--------------------------------------------------------------------------
 * Looping macro:
 *--------------------------------------------------------------------------*/

#define BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is, body)\
{\
   GrGeomSolid  *PV_gr_domain   = BCStructGrDomain(bc_struct);\
   int           PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);\
   Subgrid      *PV_subgrid     = BCStructSubgrid(bc_struct, is);\
\
   int           PV_r  = SubgridRX(PV_subgrid);\
   int           PV_ix = SubgridIX(PV_subgrid);\
   int           PV_iy = SubgridIY(PV_subgrid);\
   int           PV_iz = SubgridIZ(PV_subgrid);\
   int           PV_nx = SubgridNX(PV_subgrid);\
   int           PV_ny = SubgridNY(PV_subgrid);\
   int           PV_nz = SubgridNZ(PV_subgrid);\
\
   ival = 0;\
   GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,\
		   PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,\
   {\
      body;\
      ival++;\
   });\
}

#define BCStructPatchLoopOvrlnd(i, j, k, fdir, ival, bc_struct, ipatch, is, body)\
{\
   GrGeomSolid  *PV_gr_domain   = BCStructGrDomain(bc_struct);\
   int           PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);\
   Subgrid      *PV_subgrid     = BCStructSubgrid(bc_struct, is);\
\
   int           PV_r  = SubgridRX(PV_subgrid);\
   int           PV_ix = SubgridIX(PV_subgrid) - 1;\
   int           PV_iy = SubgridIY(PV_subgrid) - 1;\
   int           PV_iz = SubgridIZ(PV_subgrid) - 1;\
   int           PV_nx = SubgridNX(PV_subgrid) + 2;\
   int           PV_ny = SubgridNY(PV_subgrid) + 2;\
   int           PV_nz = SubgridNZ(PV_subgrid) + 2;\
\
   ival = 0;\
   GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,\
		   PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,\
   {\
      body;\
      ival++;\
   });\
}


#endif
