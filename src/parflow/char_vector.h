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
 * Header info for the CharVector data structures
 *
 *****************************************************************************/

#ifndef _CHAR_VECTOR_HEADER
#define _CHAR_VECTOR_HEADER

#include "grid.h"


/*--------------------------------------------------------------------------
 * Subcharvector
 *--------------------------------------------------------------------------*/

typedef struct
{
   char  *data;             /* Pointer to CharVector data */
   int     *data_index;       /* Array of indexes corresponding to
				 component positions in data array */
    
   Subgrid *data_space;

   int      nc;

} Subcharvector;

/*--------------------------------------------------------------------------
 * CharVector
 *--------------------------------------------------------------------------*/

typedef struct _CharVector
{
   Subcharvector    **subcharvectors;   /* Array of pointers to subcharvectors */

   char        *data;         /* Pointer to CharVector data */
   int            data_size;    /* Size of data; includes ghost points */

   Grid          *grid;         /* Grid that this charvector is on */

   SubgridArray  *data_space;   /* Description of CharVector data */

   int            nc;

   int            size;         /* Total number of coefficients */

                                /* Information on how to update boundary */
   CommPkg *comm_pkg[NumUpdateModes]; 

} CharVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Subcharvector structure
 *--------------------------------------------------------------------------*/

#define SubcharvectorData(subcharvector) ((subcharvector)-> data)
#define SubcharvectorCompData(subcharvector, ci) \
(((subcharvector) -> data) + ((subcharvector) -> data_index[ci]))

#define SubcharvectorDataSpace(subcharvector)  ((subcharvector) -> data_space)

#define SubcharvectorIX(subcharvector)   (SubgridIX(SubcharvectorDataSpace(subcharvector)))
#define SubcharvectorIY(subcharvector)   (SubgridIY(SubcharvectorDataSpace(subcharvector)))
#define SubcharvectorIZ(subcharvector)   (SubgridIZ(SubcharvectorDataSpace(subcharvector)))

#define SubcharvectorNX(subcharvector)   (SubgridNX(SubcharvectorDataSpace(subcharvector)))
#define SubcharvectorNY(subcharvector)   (SubgridNY(SubcharvectorDataSpace(subcharvector)))
#define SubcharvectorNZ(subcharvector)   (SubgridNZ(SubcharvectorDataSpace(subcharvector)))

#define SubcharvectorNC(subcharvector)   ((subcharvector) -> nc)

#define SubcharvectorEltIndex(subcharvector, x, y, z) \
(((x) - SubcharvectorIX(subcharvector)) + \
 (((y) - SubcharvectorIY(subcharvector)) + \
  (((z) - SubcharvectorIZ(subcharvector))) * \
  SubcharvectorNY(subcharvector)) * \
 SubcharvectorNX(subcharvector))

#define SubcharvectorElt(subcharvector, ci, x, y, z) \
(SubcharvectorCompData(subcharvector, ci) + \
 SubcharvectorEltIndex(subcharvector, x, y, z))

/*--------------------------------------------------------------------------
 * Accessor functions for the CharVector structure
 *--------------------------------------------------------------------------*/

#define CharVectorSubcharvector(charvector, n)  ((charvector)-> subcharvectors[(n)])
#define CharVectorData(charvector)          ((charvector)-> data)
#define CharVectorGrid(charvector)          ((charvector)-> grid)
#define CharVectorDataSpace(charvector)     ((charvector)-> data_space)
#define CharVectorNC(charvector)            ((charvector)-> nc)
#define CharVectorSize(charvector)          ((charvector)-> size)
#define CharVectorCommPkg(charvector, mode) ((charvector) -> comm_pkg[mode])

/*--------------------------------------------------------------------------
 * SizeOfCharVector macro
 *--------------------------------------------------------------------------*/

#define SizeOfCharVector(charvector)  ((charvector) -> data_size)


#endif


