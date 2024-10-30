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

#include "parflow.h"
#include "index_space.h"

#include <limits.h>

void PointCopy(Point point, Point src)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    point[dim] = src[dim];
  }
}

int BoxSize(Box *box)
{
  return (box->up[0] - box->lo[0] + 1) * (box->up[1] - box->lo[1] + 1) * (box->up[2] - box->lo[2] + 1);
}

void BoxNumberCells(Box* box, Point* number_cells)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    (*number_cells)[dim] = box->up[dim] - box->lo[dim] + 1;
  }
}

void BoxClear(Box *box)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    box->lo[dim] = INT_MAX;
    box->up[dim] = INT_MAX;
  }
}

void BoxSet(Box *box, Point lo, Point up)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    box->lo[dim] = lo[dim];
    box->up[dim] = up[dim];
  }
}

void BoxCopy(Box *dst, Box *src)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    dst->lo[dim] = src->lo[dim];
    dst->up[dim] = src->up[dim];
  }
}

void BoxPrint(Box* box)
{
  printf("[%04d,%04d,%04d][%04d,%04d,%04d]", box->lo[0], box->lo[1], box->lo[2], box->up[0], box->up[1], box->up[2]);
}

BoxArray* NewBoxArray(BoxList *box_list)
{
  BoxArray* box_array = ctalloc(BoxArray, 1);

  if (box_list)
  {
    box_array->size = BoxListSize(box_list);
    box_array->boxes = ctalloc(Box, box_array->size);

    int i = 0;
    BoxListElement* element = box_list->head;
    while (element)
    {
      BoxCopy(&(box_array->boxes[i]), &(element->box));
      for (int dim = 0; dim < DIM; dim++)
      {
        if (i == 0 || box_array->boxlimits[dim] > box_array->boxes[i].lo[dim])
          box_array->boxlimits[dim] = box_array->boxes[i].lo[dim];
        if (i == 0 || box_array->boxlimits[DIM + dim] < box_array->boxes[i].up[dim])
          box_array->boxlimits[DIM + dim] = box_array->boxes[i].up[dim];
      }
      element = element->next;
      i++;
    }
  }

  return box_array;
}

void FreeBoxArray(BoxArray* box_array)
{
  if (box_array)
  {
    if (box_array->boxes)
    {
      tfree(box_array->boxes);
    }

    tfree(box_array);
  }
}

BoxList* NewBoxList(void)
{
  return ctalloc(BoxList, 1);
}

void FreeBoxList(BoxList *box_list)
{
  if (box_list)
  {
    BoxListElement* element = box_list->head;
    while (element)
    {
      BoxListElement* next = element->next;
      tfree(element);
      element = next;
    }

    tfree(box_list);
  }
}

int BoxListSize(BoxList *box_list)
{
  return box_list->size;
}

int BoxListIsEmpty(BoxList *box_list)
{
  return box_list->size > 0 ? 0 : 1;
}

Box* BoxListFront(BoxList *box_list)
{
  return &(box_list->head->box);
}

void BoxListAppend(BoxList* box_list, Box* box)
{
  if (box_list->size == 0)
  {
    box_list->head = ctalloc(BoxListElement, 1);
    BoxCopy(&(box_list->head->box), box);

    box_list->tail = box_list->head;
    box_list->head->next = NULL;
    box_list->head->prev = NULL;

    box_list->size = 1;
  }
  else
  {
    BoxListElement* new_element = ctalloc(BoxListElement, 1);
    BoxCopy(&(new_element->box), box);

    new_element->next = NULL;
    new_element->prev = box_list->tail;
    box_list->tail->next = new_element;
    box_list->tail = new_element;

    box_list->size++;
  }
}

void BoxListConcatenate(BoxList *box_list, BoxList *concatenate_list)
{
  BoxListElement* element = concatenate_list->head;

  while (element)
  {
    BoxListAppend(box_list, &(element->box));
    element = element->next;
  }
}

void BoxListClearItems(BoxList* box_list)
{
  BoxListElement* element = box_list->head;

  while (element)
  {
    BoxListElement* next = element->next;
    tfree(element);
    element = next;
  }

  box_list->head = NULL;
  box_list->tail = NULL;
  box_list->size = 0;
}

void BoxListPrint(BoxList* box_list)
{
  BoxListElement* element = box_list->head;

  while (element)
  {
    printf("\t");
    BoxPrint(&(element->box));
    printf("\n");
    element = element->next;
  }
}
