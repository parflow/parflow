/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header info for data structures for communication of information between
 * processes.
 *
 *****************************************************************************/

#ifndef _COMMUNICATION_HEADER
#define _COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 *  Update mode stuff
 *--------------------------------------------------------------------------*/

#define NumUpdateModes 10

#define VectorUpdateAll      0
#define VectorUpdateAll2     1
#define VectorUpdateRPoint   2
#define VectorUpdateBPoint   3
#define VectorUpdateGodunov  4
#define VectorUpdateVelZ     5
#define VectorUpdatePGS1     6
#define VectorUpdatePGS2     7
#define VectorUpdatePGS3     8
#define VectorUpdatePGS4     9

/*--------------------------------------------------------------------------
 * InitCommunication:
 *--------------------------------------------------------------------------*/

#define InitCommunication(comm_pkg) \
   (CommHandle *)amps_IExchangePackage((comm_pkg) -> package);



/*--------------------------------------------------------------------------
 * FinalizeCommunication:
 *--------------------------------------------------------------------------*/

#define  FinalizeCommunication(handle) amps_Wait((amps_Handle)(handle))



/*--------------------------------------------------------------------------
 * CommPkg:
 *   Structure containing information for communicating subregions of
 *   vector data.
 *--------------------------------------------------------------------------*/

typedef struct
{
   int            num_send_invoices;
   int           *send_ranks;
   amps_Invoice  *send_invoices;

   int            num_recv_invoices;
   int           *recv_ranks;
   amps_Invoice  *recv_invoices;

   amps_Package package;
 
   int     *loop_array; /* Contains the data-array index, offset, length,
			 * and stride factors for setting up the invoices
			 * in this package.
			 * One array to save mallocing many small ones */

} CommPkg;

/*--------------------------------------------------------------------------
 * CommHandle:
 *   This structure is just a CommPkg.  The difference, at a
 *   logical level, is that Handle's point to a specific vector's data,
 *   and Pkg's do not.
 *
 *   Note: CommPkg's currently do point to specific vector data.
 *--------------------------------------------------------------------------*/

typedef amps_Handle CommHandle;

#endif
