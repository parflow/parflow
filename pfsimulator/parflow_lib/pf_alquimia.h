/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
 *
 *
 **********************************************************************EHEADER */

#ifndef PF_ALQUIMIA_H
#define PF_ALQUIMIA_H

#ifdef HAVE_ALQUIMIA
#include "chem_advance.h"
#include "chem_allocate.h"
#include "chem_datastructs.h"
#include "chem_datatransfer.h"
#include "chem_destroy.h"
#include "chem_initialize.h"
#include "chem_printdata.h"
#include "chem_processconds.h"
#include "chem_utilities.h"
#include "problem_bc_concen.h"
#include "problem_geochem_cond.h"
#include "set_chem_data.h"
#else
#include "chem_utilities.h"
#include "problem_bc_concen.h"
#endif



#endif
