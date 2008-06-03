/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.2 $
 *********************************************************************EHEADER*/

#ifndef _GMS_HEADER
#define _GMS_HEADER

#include "general.h"
#include "geometry.h"


/*--------------------------------------------------------------------------
 * Structures:
 *--------------------------------------------------------------------------*/

typedef char gms_CardType[80];

typedef struct
{
   char          solid_name[80];
   int           mat_id;

   Vertex      **vertices;
   int           nvertices;

   Triangle    **triangles;
   int           ntriangles;

} gms_Solid;

typedef struct
{
   char          TIN_name[80];
   int           mat_id;

   Vertex      **vertices;
   int           nvertices;

   Triangle    **triangles;
   int           ntriangles;

} gms_TIN;


/*--------------------------------------------------------------------------
 * Prototypes:
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* gmsreadSOL.c */
void gms_ReadSolids P((gms_Solid ***solids_ptr , int *nsolids_ptr , char *filename ));

/* gmsreadTIN.c */
void gms_ReadTINs P((gms_TIN ***TINs_ptr , int *nTINs_ptr , char *filename ));

/* gmswriteTIN.c */
void gms_WriteTINs P((gms_TIN **TINs , int nTINs , char *filename ));

#undef P


#endif
