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
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * NewComputePkg
 *--------------------------------------------------------------------------*/

ComputePkg  *NewComputePkg(
                           Region *send_reg,
                           Region *recv_reg,
                           Region *dep_reg,
                           Region *ind_reg)
{
  ComputePkg      *new_compute_pkg;


  new_compute_pkg = talloc(ComputePkg, 1);

  ComputePkgSendRegion(new_compute_pkg) = send_reg;
  ComputePkgRecvRegion(new_compute_pkg) = recv_reg;

  ComputePkgDepRegion(new_compute_pkg) = dep_reg;
  ComputePkgIndRegion(new_compute_pkg) = ind_reg;

  return new_compute_pkg;
}


/*--------------------------------------------------------------------------
 * FreeComputePkg
 *--------------------------------------------------------------------------*/

void         FreeComputePkg(
                            ComputePkg *compute_pkg)
{
  if (ComputePkgSendRegion(compute_pkg))
    FreeRegion(ComputePkgSendRegion(compute_pkg));
  if (ComputePkgRecvRegion(compute_pkg))
    FreeRegion(ComputePkgRecvRegion(compute_pkg));

  if (ComputePkgIndRegion(compute_pkg))
    FreeRegion(ComputePkgIndRegion(compute_pkg));
  if (ComputePkgDepRegion(compute_pkg))
    FreeRegion(ComputePkgDepRegion(compute_pkg));

  tfree(compute_pkg);
}


