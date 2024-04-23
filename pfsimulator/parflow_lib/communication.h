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
 * CommPkg:
 *   Structure containing information for communicating subregions of
 *   vector data.
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_send_invoices;
  int           *send_ranks;
  amps_Invoice  *send_invoices;

  int num_recv_invoices;
  int           *recv_ranks;
  amps_Invoice  *recv_invoices;

  amps_Package package;

  int     *loop_array;  /* Contains the data-array index, offset, length,
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
