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
* Header file for `well.c'
*
* (C) 1996 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Revision: 1.3 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#ifndef PFWELL_DATA_HEADER
#define PFWELL_DATA_HEADER

#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

#include "general.h"

#include "well.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------------------------------------
 * Well List Member structure
 *----------------------------------------------------------------*/

typedef struct well_list_member {
  double time;
  WellDataPhysical        **well_data_physicals;
  WellDataValue           **well_data_values;
  WellDataStat            **well_data_stats;
  struct well_list_member  *next_well_list_member;
} WellListMember;


/*--------------------------------------------------------------------------
 * Accessor macros: WellListMember
 *--------------------------------------------------------------------------*/
#define WellListMemberTime(well_list_member) \
  ((well_list_member)->time)

#define WellListMemberWellDataPhysicals(well_list_member) \
  ((well_list_member)->well_data_physicals)
#define WellListMemberWellDataPhysical(well_list_member, i) \
  ((well_list_member)->well_data_physicals[i])

#define WellListMemberWellDataValues(well_list_member) \
  ((well_list_member)->well_data_values)
#define WellListMemberWellDataValue(well_list_member, i) \
  ((well_list_member)->well_data_values[i])

#define WellListMemberWellDataStats(well_list_member) \
  ((well_list_member)->well_data_stats)
#define WellListMemberWellDataStat(well_list_member, i) \
  ((well_list_member)->well_data_stats[i])

#define WellListMemberNextWellListMember(well_list_member) \
  ((well_list_member)->next_well_list_member)

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

void WriteWellListData(FILE *fd, double time, WellDataStat *well_data_stat, int num_phases, int num_components);

#ifdef __cplusplus
}
#endif

#endif
