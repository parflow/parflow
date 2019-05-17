
/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2019, Lawrence Livermore National Security,
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

#ifndef _INDEX_SPACE_HEADER
#define _INDEX_SPACE_HEADER

/**
 * Structures for index space.
 */

#define DIM 3

/**
 * Hold a point in index space.
 */
// SGS rename to Point?
typedef int Index[DIM];

/**
 * A box in index space.
 */
typedef struct 
{
  Index lo;
  Index up;
} Box;

typedef struct _BoxListElement
{
  Box box;
  struct _BoxListElement* next;
  struct _BoxListElement* prev;
} BoxListElement;

typedef struct _BoxList
{
  BoxListElement* head;
  BoxListElement* tail;
  unsigned int size;
} BoxList;


void IndexCopy(Index index, Index src);
int BoxSize(Box *box);
void BoxNumberCells(Box* box, Index* number_cells);
void BoxClear(Box *box);
void BoxSet(Box *box, Index lo, Index up);
void BoxCopy(Box *dst, Box *src);
void BoxPrint(Box* box);
BoxList* NewBoxList(void);
int BoxListSize(BoxList *box_list);
int BoxListIsEmpty(BoxList *box_list);
Box* BoxListFront(BoxList *box_list);
void BoxListAppend(BoxList* box_list, Box* box);
void BoxListConcatenate(BoxList *box_list, BoxList *concatenate_list);
void BoxListClearItems(BoxList* box_list);
void BoxListPrint(BoxList* box_list);

#endif 

