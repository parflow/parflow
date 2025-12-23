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
#ifndef _hbt_HEADER
#define _hbt_HEADER

#include <stdio.h>

/****************************************************************************/
/* Height Balanced Tree Information                                          */
/****************************************************************************/

/*===========================================================================*/
/* These are the return codes.                                               */
/*===========================================================================*/
#define HBT_NOT_FOUND 0
#define HBT_FOUND 1
#define HBT_INSERTED 2
#define HBT_DELETED 3
#define HBT_MEMORY_ERROR 4

#define TRUE 1
#define FALSE 0

/*===========================================================================*/
/* This is an actual node on the HBT.                                    */
/*===========================================================================*/
typedef struct _HBT_element {
  void *obj;

  short balance;

  struct _HBT_element *left;
  struct _HBT_element *right;
} HBT_element;

/*===========================================================================*/
/* This is the "head" structure for HBT's.                               */
/*===========================================================================*/
typedef struct _HBT {
  unsigned int height;
  unsigned int num;

  HBT_element *root;


  int (*compare)(void *, void *);
  void (*free)(void *);
  void (*printf)(FILE *file, void *);
  int (*scanf)(FILE *file, void **);

  int malloc_flag;
} HBT;

#endif

