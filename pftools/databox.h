/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.9 $
 *********************************************************************EHEADER*/
/******************************************************************************
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

#ifndef NULL
#define NULL ((void *)0)
#endif

#define MAX_LABEL_SIZE 128

/*-----------------------------------------------------------------------
 * Databox structure and accessor functions
 *-----------------------------------------------------------------------*/

typedef struct
{
   double         *coeffs;

   int             nx, ny, nz;

   /* other info that may not be available */
   double          x,  y,  z;
   double          dx, dy, dz;

   char           label[MAX_LABEL_SIZE];

}               Databox;

#define DataboxCoeffs(databox)        ((databox) -> coeffs)

#define DataboxNx(databox)            ((databox) -> nx)
#define DataboxNy(databox)            ((databox) -> ny)
#define DataboxNz(databox)            ((databox) -> nz)

#define DataboxX(databox)             ((databox) -> x)
#define DataboxY(databox)             ((databox) -> y)
#define DataboxZ(databox)             ((databox) -> z)

#define DataboxDx(databox)            ((databox) -> dx)
#define DataboxDy(databox)            ((databox) -> dy)
#define DataboxDz(databox)            ((databox) -> dz)

#define DataboxLabel(databox)   ((databox) -> label)

#define DataboxCoeff(databox, i, j, k) \
   (DataboxCoeffs(databox) + \
    (k)*DataboxNy(databox)*DataboxNx(databox) + (j)*DataboxNx(databox) + (i))


/* Defines how a grid definition is */
/* to be interpreted.               */

typedef enum
{
   vertex,
   cell
} GridType;



/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* databox.c */
Databox *NewDatabox ANSI_PROTO((int nx , int ny , int nz , double x , double y , double z , double dx , double dy , double dz ));
void GetDataboxGrid ANSI_PROTO((Tcl_Interp *interp , Databox *databox ));
void FreeDatabox ANSI_PROTO((Databox *databox ));

#undef ANSI_PROTO

#endif

