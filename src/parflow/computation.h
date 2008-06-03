/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * Header info for compute package data structures.
 *
 *****************************************************************************/

#ifndef _COMPUTATION_HEADER
#define _COMPUTATION_HEADER

#include "region.h"


/*--------------------------------------------------------------------------
 *  ComputePkg:
 *    Structure defining various computational regions.
 *--------------------------------------------------------------------------*/

typedef struct
{
   Region  *send_region; /* send region */
   Region  *recv_region; /* receive region */

   Region  *ind_region;  /* neighbor independent computational region */
   Region  *dep_region;  /* neighbor dependent computational region */

} ComputePkg;


/*--------------------------------------------------------------------------
 * Accessor functions for ComputePkg structure
 *--------------------------------------------------------------------------*/

#define ComputePkgSendRegion(compute_pkg) ((compute_pkg) -> send_region)
#define ComputePkgRecvRegion(compute_pkg) ((compute_pkg) -> recv_region)

#define ComputePkgIndRegion(compute_pkg) ((compute_pkg) -> ind_region)
#define ComputePkgDepRegion(compute_pkg) ((compute_pkg) -> dep_region)


#endif

