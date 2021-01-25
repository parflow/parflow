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
 **********************************************************************EHEADER*/
#ifndef TOP_HEADER
#define TOP_HEADER

#include "databox.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute indices of the top of the domain.
 *
 * Computes a 2D top Databox with integer values that are the indices
 * in Z of the top of the domain.   Each (i,j,0) is the z index of
 * the top of the domain.
 * 
 * The 3D Mask input is one of the optional outputs from a ParFlow
 * run.  The mask has values 0 outside of domain so top most non-zero
 * entry in each (i,j) column is the top of the domain.
 * 
 * @param [in] mask 3D mask Databox from the ParFlow run.
 * @param [out] top 2D Databox with z indices 
 */
void ComputeTop(Databox *mask, Databox  *top);

/**
 * Compute indices of the bottom of the domain.
 *
 * Computes a 2D bottom Databox with integer values that are the indices
 * in Z of the bottom of the domain.   Each (i,j,0) is the z index of
 * the bottom of the domain.
 * 
 * The 3D Mask input is one of the optional outputs from a ParFlow
 * run.  The mask has values 0 outside of domain so bottom most
 * non-zero entry in each (i,j) column is the bottom of the domain.
 * 
 * @param [in] mask 3D mask Databox from the ParFlow run.
 * @param [out] bottom 2D Databox with z indices 
 */
void ComputeBottom(Databox *mask, Databox  *bottom);

/**
 * Extracts the top domain surface values of a dataset.
 *
 * Returns 2D DataBox from a 3D Databox with data
 * that lies on the top domain surface boundary.
 *
 * The top is defined by the z indices in the provided 2D top databox
 * which can be computed with ComputeTop.
 *
 * @param [in] top 2D Databox with z index values that define the top of the domain
 * @param [in] data 3D Databox with data that should be sampled
 * @param [out] topdata 2D Databox extracted from the data input on top of the domain
 */
void ExtractTop(Databox *top, Databox *data, Databox *topdata);

/**
 * Extracts the boundary values of a dataset.
 *
 * Returns 2D DataBox from a 2D Databox with data that lies along the
 * domain boundary in X/Y.  This could be used in sequence with
 * ExtractTop to for a 3D dataset.  First extract the top and then
 * extract the boundary.
 *
 * The domain boundary is defined using a 2D top Databox which can 
 * be computed with ComputeTop.
 *
 * @param [in] top 2D Databox with z index values that define the top of the domain
 * @param [in] data 2D Databox 
 * @param [out] boundarydata 2D Databox sampled from the data input that lies on X/Y boundaries.
 */
void ExtractTopBoundary(Databox *top, Databox *data, Databox *boundarydata);

#ifdef __cplusplus
}
#endif

#endif

