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
* Header file to include all header information for parflow.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#ifndef _PARFLOW_HEADER
#define _PARFLOW_HEADER

#ifndef _WIN32
#include <sys/param.h>
#endif

#include "amps.h"

#include "info_header.h"
#include "general.h"
#include "file_versions.h"
#include "input_database.h"
#include "logging.h"
#include "timing.h"
#include "loops.h"
#include "background.h"
#include "communication.h"
#include "computation.h"
#include "region.h"
#include "grid.h"
#include "matrix.h"
#include "vector.h"
#include "pf_module.h"
#include "geometry.h"
#include "grgeometry.h"
#include "geostats.h"
#include "lb.h"
#include "globals.h"
#include "time_cycle_data.h"
#include "problem_bc.h"
#include "problem_eval.h"
#include "well.h"
#include "bc_pressure.h"
#include "problem.h"
#include "solver.h"
#include "nl_function_eval.h"
#include "parflow_proto.h"
#include "parflow_proto_f.h"

// SGS FIXME this should not be here, in fact this whole parflow.h file is dumb.
#include <math.h>

// backend_mapping.h must be included as the last header
#include "backend_mapping.h"

#endif
