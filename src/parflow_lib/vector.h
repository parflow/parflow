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
 * Header info for the Vector data structures
 *
 *****************************************************************************/

#ifndef _VECTOR_HEADER
#define _VECTOR_HEADER

#include "grid.h"

/*--------------------------------------------------------------------------
 * Subvector
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;             /* Pointer to Vector data */
    
   Subgrid *data_space;

} Subvector;

/*--------------------------------------------------------------------------
 * Vector
 *--------------------------------------------------------------------------*/

typedef struct _Vector
{
   Subvector    **subvectors;   /* Array of pointers to subvectors */

   double        *data;         /* Pointer to Vector data */
   int            data_size;    /* Size of data; includes ghost points */

#ifdef SHMEM_OBJECTS
   int            shmem_offset; /* For the shared memory imp the offset
				   from the "shared" data space */
#endif

   Grid          *grid;         /* Grid that this vector is on */

   SubgridArray  *data_space;   /* Description of Vector data */

   int            size;         /* Total number of coefficients */

                                /* Information on how to update boundary */
   CommPkg *comm_pkg[NumUpdateModes]; 

} Vector;

typedef struct _Multispecies
{
  Vector **specie;              /* Array of pointers to vectors */
} Multispecies;

/*--------------------------------------------------------------------------
 * Accessor functions for the Subvector structure
 *--------------------------------------------------------------------------*/

#define SubvectorData(subvector) ((subvector)-> data)

#define SubvectorDataSpace(subvector)  ((subvector) -> data_space)

#define SubvectorIX(subvector)   (SubgridIX(SubvectorDataSpace(subvector)))
#define SubvectorIY(subvector)   (SubgridIY(SubvectorDataSpace(subvector)))
#define SubvectorIZ(subvector)   (SubgridIZ(SubvectorDataSpace(subvector)))

#define SubvectorNX(subvector)   (SubgridNX(SubvectorDataSpace(subvector)))
#define SubvectorNY(subvector)   (SubgridNY(SubvectorDataSpace(subvector)))
#define SubvectorNZ(subvector)   (SubgridNZ(SubvectorDataSpace(subvector)))

#define SubvectorEltIndex(subvector, x, y, z) \
(((x) - SubvectorIX(subvector)) + \
 (((y) - SubvectorIY(subvector)) + \
  (((z) - SubvectorIZ(subvector))) * \
  SubvectorNY(subvector)) * \
 SubvectorNX(subvector))

#define SubvectorElt(subvector, x, y, z) \
(SubvectorData(subvector) + SubvectorEltIndex(subvector, x, y, z))

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define VectorSubvector(vector, n)  ((vector)-> subvectors[(n)])
#define VectorData(vector)          ((vector)-> data)
#define VectorGrid(vector)          ((vector)-> grid)
#define VectorDataSpace(vector)     ((vector)-> data_space)
#define VectorSize(vector)          ((vector)-> size)
#define VectorCommPkg(vector, mode) ((vector) -> comm_pkg[mode])

/*--------------------------------------------------------------------------
 * SizeOfVector macro
 *--------------------------------------------------------------------------*/

#define SizeOfVector(vector)  ((vector) -> data_size)


#endif


