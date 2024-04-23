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
* Routines associated with the Background structure.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * ReadBackground
 *--------------------------------------------------------------------------*/

Background  *ReadBackground()
{
  Background    *background;

  background = talloc(Background, 1);

  BackgroundX(background) = GetDouble("ComputationalGrid.Lower.X");
  BackgroundY(background) = GetDouble("ComputationalGrid.Lower.Y");
  BackgroundZ(background) = GetDouble("ComputationalGrid.Lower.Z");

  BackgroundDX(background) = GetDouble("ComputationalGrid.DX");
  BackgroundDY(background) = GetDouble("ComputationalGrid.DY");
  BackgroundDZ(background) = GetDouble("ComputationalGrid.DZ");

  return background;
}


/*--------------------------------------------------------------------------
 * FreeBackground
 *--------------------------------------------------------------------------*/

void         FreeBackground(
                            Background *background)
{
  tfree(background);
}


/*--------------------------------------------------------------------------
 * SetBackgroundBounds
 *--------------------------------------------------------------------------*/

void         SetBackgroundBounds(
                                 Background *background,
                                 Grid *      grid)
{
  int ix_lower = 0, iy_lower = 0, iz_lower = 0;
  int ix_upper = 0, iy_upper = 0, iz_upper = 0;

  Subgrid  *subgrid;
  int is;


  /*---------------------------------------------------
   * Look at only level 0 subgrids to get bounds:
   * all finer subgrids are contained within level 0
   * subgrids by definition.
   *---------------------------------------------------*/

  for (is = 0; is < GridNumSubgrids(grid); is++)
  {
    subgrid = GridSubgrid(grid, is);

    if (SubgridLevel(subgrid) == 0)
    {
      ix_lower = SubgridIX(subgrid);
      iy_lower = SubgridIY(subgrid);
      iz_lower = SubgridIZ(subgrid);
      ix_upper = SubgridIX(subgrid) + SubgridNX(subgrid) - 1;
      iy_upper = SubgridIY(subgrid) + SubgridNY(subgrid) - 1;
      iz_upper = SubgridIZ(subgrid) + SubgridNZ(subgrid) - 1;

      break;
    }
  }

  for (; is < GridNumSubgrids(grid); is++)
  {
    subgrid = GridSubgrid(grid, is);

    if (SubgridLevel(subgrid) == 0)
    {
      ix_lower = pfmin(ix_lower, SubgridIX(subgrid));
      iy_lower = pfmin(iy_lower, SubgridIY(subgrid));
      iz_lower = pfmin(iz_lower, SubgridIZ(subgrid));
      ix_upper = pfmax(ix_upper, SubgridIX(subgrid) + SubgridNX(subgrid) - 1);
      iy_upper = pfmax(iy_upper, SubgridIY(subgrid) + SubgridNY(subgrid) - 1);
      iz_upper = pfmax(iz_upper, SubgridIZ(subgrid) + SubgridNZ(subgrid) - 1);
    }
  }

  BackgroundIX(background) = ix_lower;
  BackgroundIY(background) = iy_lower;
  BackgroundIZ(background) = iz_lower;

  BackgroundNX(background) = ix_upper - ix_lower + 1;
  BackgroundNY(background) = iy_upper - iy_lower + 1;
  BackgroundNZ(background) = iz_upper - iz_lower + 1;
}
