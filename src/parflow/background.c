/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
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

void         FreeBackground(background)
Background  *background;
{
   tfree(background);
}


/*--------------------------------------------------------------------------
 * SetBackgroundBounds
 *--------------------------------------------------------------------------*/

void         SetBackgroundBounds(background, grid)
Background  *background;
Grid        *grid;
{
   int       ix_lower, iy_lower, iz_lower;
   int       ix_upper, iy_upper, iz_upper;

   Subgrid  *subgrid;
   int       is;


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
	 ix_lower = min(ix_lower, SubgridIX(subgrid));
	 iy_lower = min(iy_lower, SubgridIY(subgrid));
	 iz_lower = min(iz_lower, SubgridIZ(subgrid));
	 ix_upper = max(ix_upper, SubgridIX(subgrid) + SubgridNX(subgrid) - 1);
	 iy_upper = max(iy_upper, SubgridIY(subgrid) + SubgridNY(subgrid) - 1);
	 iz_upper = max(iz_upper, SubgridIZ(subgrid) + SubgridNZ(subgrid) - 1);
      }
   }

   BackgroundIX(background) = ix_lower;
   BackgroundIY(background) = iy_lower;
   BackgroundIZ(background) = iz_lower;

   BackgroundNX(background) = ix_upper - ix_lower + 1;
   BackgroundNY(background) = iy_upper - iy_lower + 1;
   BackgroundNZ(background) = iz_upper - iz_lower + 1;
}
