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

#ifndef _GMS_HEADER
#define _GMS_HEADER

#include "general.h"
#include "geometry.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures:
 *--------------------------------------------------------------------------*/

typedef char gms_CardType[80];

typedef struct {
  char solid_name[80];
  int mat_id;

  Vertex      **vertices;
  int nvertices;

  Triangle    **triangles;
  int ntriangles;
} gms_Solid;

typedef struct {
  char TIN_name[80];
  int mat_id;

  Vertex      **vertices;
  int nvertices;

  Triangle    **triangles;
  int ntriangles;
} gms_TIN;


/*--------------------------------------------------------------------------
 * Prototypes:
 *--------------------------------------------------------------------------*/

/* gmsreadSOL.c */
void gms_ReadSolids(gms_Solid * **solids_ptr, int *nsolids_ptr, char *filename);

/* gmsreadTIN.c */
void gms_ReadTINs(gms_TIN * **TINs_ptr, int *nTINs_ptr, char *filename);

/* gmswriteTIN.c */
void gms_WriteTINs(gms_TIN * *TINs, int nTINs, char *filename);

#ifdef __cplusplus
}
#endif

#endif
