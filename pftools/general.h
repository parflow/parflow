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

#ifndef _GENERAL_HEADER
#define _GENERAL_HEADER

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Define memory allocation routines
 *--------------------------------------------------------------------------*/

#define talloc(type, count) \
        ((count > 0) ? (type*)malloc((unsigned int)(sizeof(type) * (count))) : NULL)

#define ctalloc(type, count) \
        ((count > 0) ? (type*)calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)

/* note: the `else' is required to guarantee termination of the `if' */
#define tfree(ptr) if (ptr) free(ptr); else


/*--------------------------------------------------------------------------
 * Define max and min functions
 *--------------------------------------------------------------------------*/

#ifndef max
#define max(a, b)  (((a) < (b)) ? (b) : (a))
#endif
#ifndef min
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#endif

/*
 * round is defined in math.h and it may not be a macro so this does not work
 #ifndef round
 #define round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
 #endif
 #endif
 */

/*--------------------------------------------------------------------------
 * Define various flags
 *--------------------------------------------------------------------------*/

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ON  1
#define OFF 0

#ifdef __cplusplus
}
#endif

#endif

