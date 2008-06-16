/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * Header info for the Background structure
 *
 *****************************************************************************/

#ifndef _BACKGROUND_HEADER
#define _BACKGROUND_HEADER

/*--------------------------------------------------------------------------
 * Background:
 *   Structure describing a uniform grid in real-space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   X,  Y,  Z;    /* Anchor point in real-space */
   double   DX, DY, DZ;   /* Spacing in each coordinate direction */

   /* Bounding information (essentially a level 0 subgrid) */
   int  IX, IY, IZ;
   int  NX, NY, NZ;

} Background;

/*--------------------------------------------------------------------------
 * Accessor macros: Background
 *--------------------------------------------------------------------------*/

#define BackgroundX(bg)   ((bg) -> X)
#define BackgroundY(bg)   ((bg) -> Y)
#define BackgroundZ(bg)   ((bg) -> Z)

#define BackgroundDX(bg)  ((bg) -> DX)
#define BackgroundDY(bg)  ((bg) -> DY)
#define BackgroundDZ(bg)  ((bg) -> DZ)

#define BackgroundIX(bg)  ((bg) -> IX)
#define BackgroundIY(bg)  ((bg) -> IY)
#define BackgroundIZ(bg)  ((bg) -> IZ)

#define BackgroundNX(bg)  ((bg) -> NX)
#define BackgroundNY(bg)  ((bg) -> NY)
#define BackgroundNZ(bg)  ((bg) -> NZ)

#define BackgroundXLower(bg) \
(BackgroundX(bg) + BackgroundIX(bg)*BackgroundDX(bg))
#define BackgroundYLower(bg) \
(BackgroundY(bg) + BackgroundIY(bg)*BackgroundDY(bg))
#define BackgroundZLower(bg) \
(BackgroundZ(bg) + BackgroundIZ(bg)*BackgroundDZ(bg))

#define BackgroundXUpper(bg) \
(BackgroundX(bg) + (BackgroundIX(bg) + BackgroundNX(bg))*BackgroundDX(bg))
#define BackgroundYUpper(bg) \
(BackgroundY(bg) + (BackgroundIY(bg) + BackgroundNY(bg))*BackgroundDY(bg))
#define BackgroundZUpper(bg) \
(BackgroundZ(bg) + (BackgroundIZ(bg) + BackgroundNZ(bg))*BackgroundDZ(bg))


#endif
