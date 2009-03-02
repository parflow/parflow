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
 *
 * Header info for the Vector data structures
 *
 *****************************************************************************/

#ifndef _VECTOR_HEADER
#define _VECTOR_HEADER

#include "grid.h"
#include "n_vector.h"

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


typedef Vector *N_Vector;


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


