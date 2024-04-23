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
*
* Routines to help with boundary conditions
*
* NOTE :
*        This routine assumes that the domain structure was set up with
*        surfaces that are rectilinear areas lying in a plane.
*
*****************************************************************************/

#include "parflow.h"
#include "problem_bc.h"

/*--------------------------------------------------------------------------
 * NewBCStruct
 *--------------------------------------------------------------------------*/

BCStruct *NewBCStruct(
                      SubgridArray *subgrids,
                      GrGeomSolid * gr_domain,
                      int           num_patches,
                      int *         patch_indexes,
                      int *         bc_types,
                      double ***    values)
{
  BCStruct       *new_bcstruct;


  new_bcstruct = talloc(BCStruct, 1);

  (new_bcstruct->subgrids) = subgrids;
  (new_bcstruct->gr_domain) = gr_domain;
  (new_bcstruct->num_patches) = num_patches;
  (new_bcstruct->patch_indexes) = patch_indexes;
  (new_bcstruct->bc_types) = bc_types;
  (new_bcstruct->values) = values;

  return new_bcstruct;
}

/*--------------------------------------------------------------------------
 * FreeBCStruct
 *--------------------------------------------------------------------------*/

void      FreeBCStruct(BCStruct *bc_struct)
{
  double  ***values;

  int ipatch, is;


  values = BCStructValues(bc_struct);
  if (values)
  {
    for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
    {
      ForSubgridI(is, BCStructSubgrids(bc_struct))
      {
        tfree(values[ipatch][is]);
      }
      tfree(values[ipatch]);
    }
    tfree(values);
  }

  tfree(bc_struct);
}
