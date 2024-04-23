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
/*****************************************************************************
* Databox header file
*
* (C) 1993 Regents of the University of California.
*
*-----------------------------------------------------------------------------
* $Revision: 1.9 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#ifndef DATABOX_HEADER
#define DATABOX_HEADER

#include "parflow_config.h"

#ifdef HAVE_HDF
#include <hdf.h>
#endif

#include <tcl.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif

#define MAX_LABEL_SIZE 128

/*-----------------------------------------------------------------------
 * Databox structure and accessor functions
 *-----------------------------------------------------------------------*/

typedef struct {
  double         *coeffs;

  int nx, ny, nz;

  /* other info that may not be available */
  double x, y, z;
  double dx, dy, dz;

  char label[MAX_LABEL_SIZE];
}               Databox;

#define DataboxCoeffs(databox)        ((databox)->coeffs)

#define DataboxNx(databox)            ((databox)->nx)
#define DataboxNy(databox)            ((databox)->ny)
#define DataboxNz(databox)            ((databox)->nz)

#define DataboxX(databox)             ((databox)->x)
#define DataboxY(databox)             ((databox)->y)
#define DataboxZ(databox)             ((databox)->z)

#define DataboxDx(databox)            ((databox)->dx)
#define DataboxDy(databox)            ((databox)->dy)
#define DataboxDz(databox)            ((databox)->dz)

#define DataboxLabel(databox)   ((databox)->label)

#define DataboxCoeff(databox, i, j, k) \
  (DataboxCoeffs(databox) +            \
   (k) * DataboxNy(databox) * DataboxNx(databox) + (j) * DataboxNx(databox) + (i))


/* Defines how a grid definition is */
/* to be interpreted.               */

typedef enum {
  vertex,
  cell
} GridType;



/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

/* databox.c */
Databox *NewDatabox(int nx, int ny, int nz, double x, double y, double z, double dx, double dy, double dz);
Databox *NewDataboxDefault(int nx, int ny, int nz, double x, double y, double z, double dx, double dy, double dz,
                           double default_value);
void GetDataboxGrid(Tcl_Interp *interp, Databox *databox);
void SetDataboxGrid(Databox *databox, int nx, int ny, int nz, double x, double y, double z,
                    double dx, double dy, double dz);
void FreeDatabox(Databox *databox);

#ifdef __cplusplus
}
#endif

#endif

