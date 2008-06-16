/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _GRGEOM_LIST_HEADER
#define _GRGEOM_LIST_HEADER

#include "parflow.h"

/*----------------------------------------------------------------
 * List Structure
 *----------------------------------------------------------------*/

typedef struct list_member
{
   double                value;
   int                   normal_component;
   int                   triangle_id;
   struct list_member   *next_list_member;
} ListMember;


/*--------------------------------------------------------------------------
 * Accessors : ListMember
 *--------------------------------------------------------------------------*/
#define ListMemberValue(list_member)           ((list_member) -> value)
#define ListMemberNormalComponent(list_member) ((list_member) -> normal_component)
#define ListMemberTriangleID(list_member)      ((list_member) -> triangle_id)
#define ListMemberNextListMember(list_member)  ((list_member) -> next_list_member)

/*--------------------------------------------------------------------------
 * Macros : ListMember
 *--------------------------------------------------------------------------*/
#define ListValueInClosedInterval(current_member, lower_point, upper_point) \
(((current_member) != NULL) ? (((ListMemberValue((current_member)) >= (lower_point)) && (ListMemberValue((current_member)) <= (upper_point))) ? TRUE : FALSE) : FALSE)

#define ListValueLEPoint(current_member, point) \
(((current_member) != NULL) ? ((ListMemberValue((current_member)) <= (point)) ? TRUE : FALSE) : FALSE)

#define ListValueLTPoint(current_member, point) \
(((current_member) != NULL) ? ((ListMemberValue((current_member)) < (point)) ? TRUE : FALSE) : FALSE)

#define ListValueEQPoint(current_member, point) \
(((current_member) != NULL) ? ((ListMemberValue((current_member)) == (point)) ? TRUE : FALSE) : FALSE)

#define ListValueGTPoint(current_member, point) \
(((current_member) != NULL) ? ((ListMemberValue((current_member)) > (point)) ? TRUE : FALSE) : FALSE)

#define ListValueGEPoint(current_member, point) \
(((current_member) != NULL) ? ((ListMemberValue((current_member)) >= (point)) ? TRUE : FALSE) : FALSE)

#endif
