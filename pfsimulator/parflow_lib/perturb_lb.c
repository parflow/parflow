/*****************************************************************************
*
* This function calculates the perturbation
* to a pressure field.
*
*
* (C) 1993 Regents of the University of California.
*
* $Revision: 1.1.1.1 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/


/*--------------------*
* Include files
*--------------------*/
#include "parflow.h"


/**************************************************************************
* Perturbation function for lattice Boltzmann solver.
**************************************************************************/
void PerturbSystem(
                   Lattice *lattice,
                   Problem *problem)
{
  /*------------------------------------------------------------*
  * Local variables
  *------------------------------------------------------------*/

  /* Lattice variables */
  Grid  *grid = (lattice->grid);
  Vector *pressure = (lattice->pressure);
  CharVector *cellType = (lattice->cellType);

  /* Grid parameters */
  Subgrid   *subgrid;
  int ix, iy, iz;
  int nx, ny, nz;

  /* Physical variables and coefficients */
  Subvector *sub_p;
  Subcharvector *sub_cellType;
  double    *pp;
  char      *cellTypep;

  /* Indices */
  int i, j, k;
  int index;
  int gridloop;

  for (gridloop = 0; gridloop < GridNumSubgrids(grid); gridloop++)
  {
    subgrid = GridSubgrid(grid, gridloop);

    sub_p = VectorSubvector(pressure, gridloop);
    sub_cellType = CharVectorSubcharvector(cellType, gridloop);
    pp = SubvectorData(sub_p);
    cellTypep = SubcharvectorData(sub_cellType);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

#if 0
    /* Reset the interior to a lower, constant value */
    for (i = ix; i < ix + nx; i++)
      for (j = iy; j < iy + ny; j++)
        for (k = iz; k < iz + nz; k++)
        {
          index = SubvectorEltIndex(sub_p, i, j, k);
          if (cellTypep[index])
            pp[index] = 500.0 - rho * g * ((nz - k) * dz);
          cellTypep[index] = 1;
        }
#endif

#if 1
    /* Single pulse plane in the center of the x axis */
    i = ix + nx / 2;
    j = iy + ny / 2;
    k = iz + nz / 2;
/*
 *    for(j=iy; j<iy+ny; j++)
 *    for(k=iz; k<iz+nz; k++)
 */
    {
      index = SubvectorEltIndex(sub_p, i, j, k);
      pp[index] = 1.0e1;
      cellTypep[index] = 1;
    }
#endif

#if 0
    /* Single pulse in center of domain */
    i = nx / 2;
    j = ny / 2;
    k = nz / 2;

    index = SubvectorEltIndex(sub_p, i, j, k);
    pp[index] += 100.0;
#endif
  }
}
