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
* This is a simple kriging routine that computes a probability
* field or p-field for the geounit of interest. A p-field is
* and array of means and variances that are determined solely from
* external conditioning data.
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * PField
 *--------------------------------------------------------------------------*/

void         PField(
                    Grid *       grid,
                    GeomSolid *  geounit,
                    GrGeomSolid *gr_geounit,
                    Vector *     field,
                    RFCondData * cdata,
                    Statistics * stats)
{
  /*-----------------*
  * Local variables *
  *-----------------*/

  /* Conditioning and statistical data */
  int nc = (cdata->nc);
  double    *x = (cdata->x);
  double    *y = (cdata->y);
  double    *z = (cdata->z);
  double    *v = (cdata->v);
  double lambdaX = (stats->lambdaX);
  double lambdaY = (stats->lambdaY);
  double lambdaZ = (stats->lambdaZ);
  double mean = (stats->mean);
  double sigma = (stats->sigma);
  int lognormal = (stats->lognormal);
  int nc_sub;
  double    *v_sub;

  /* Grid parameters */
  Subgrid   *subgrid;
  Subvector *sub_field;
  double    *fieldp;

  /* Subgrid parameters */
  int nx, ny, nz;
  double dx, dy, dz;

  /* Counters, indices, flags */
  int gridloop;
  int i, j, k, n, m, nn;
  int       *i_array;
  int cpts;
  int index1;

  /* Spatial variables */
  int iLx, iLy, iLz;            /* Correlation length in terms of grid points */
  int nLx, nLy, nLz;            /* Size of correlation neighborhood in grid pts. */
  int nLxyz;                    /* nLxyz = nLx*nLy*nLz */
  int ix, iy, iz;
  int ref;
  int max_search_radius;
  double X0, Y0, Z0;
  double cx, cy, cz;
  double a1, a2, a3;

  /* Variables used in kriging algorithm */
  double cmean, csigma;         /* Conditional mean and std. dev. from kriging */
  double    *A;
  double    *b;                 /* Covariance vector for conditioning points */
  double    *w;                 /* Solution vector to Aw=b */
  double    *value;
  int di, dj, dk;
  double    ***cov;
  int ierr;

  /* Conditioning data variables */
  int       *ci, *cj, *ck;      /* Indices for conditioning data points */

  (void)geounit;

  /*-----------------------------------------------------------------------
   * Start p-field algorithm
   *-----------------------------------------------------------------------*/

  /* For now, we will assume that all subgrids have
   * the same uniform spacing */
  subgrid = GridSubgrid(grid, 0);

  dx = SubgridDX(subgrid);
  dy = SubgridDY(subgrid);
  dz = SubgridDZ(subgrid);

  /* Size of search neighborhood */
  iLx = (int)(lambdaX / dx);
  iLy = (int)(lambdaY / dy);
  iLz = (int)(lambdaZ / dz);

  /* Define the size of a correlation neighborhood */
  nLx = 2 * iLx + 1;
  nLy = 2 * iLy + 1;
  nLz = 2 * iLz + 1;
  nLxyz = nLx * nLy * nLz;

  /* Allocate memory for conditioning data indices */
  ci = ctalloc(int, nc);
  cj = ctalloc(int, nc);
  ck = ctalloc(int, nc);
  i_array = ctalloc(int, nc);
  v_sub = ctalloc(double, nc);

  /*-----------------------------------------------------------------------
   * Compute correlation lookup table
   *-----------------------------------------------------------------------*/
  /* First compute a covariance lookup table */
  cov = talloc(double**, nLx);
  for (i = 0; i < nLx; i++)
  {
    cov[i] = talloc(double*, nLy);
    for (j = 0; j < nLy; j++)
      cov[i][j] = ctalloc(double, nLz);
  }

  /* Allocate memory for variables that will be used in kriging */
  A = ctalloc(double, nLxyz * nLxyz);
  b = ctalloc(double, nLxyz);
  w = ctalloc(double, nLxyz);
  value = ctalloc(double, nLxyz);

  /*--------------------------------------------------------------------
   * Start kriging algorithm
   *--------------------------------------------------------------------*/
  for (gridloop = 0; gridloop < GridNumSubgrids(grid); gridloop++)
  {
    subgrid = GridSubgrid(grid, gridloop);
    sub_field = VectorSubvector(field, gridloop);
    fieldp = SubvectorData(sub_field);

    X0 = RealSpaceX(0, SubgridRX(subgrid));
    Y0 = RealSpaceY(0, SubgridRY(subgrid));
    Z0 = RealSpaceZ(0, SubgridRZ(subgrid));

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    /* Size of search neighborhood */
    iLx = (int)(lambdaX / dx);
    iLy = (int)(lambdaY / dy);
    iLz = (int)(lambdaZ / dz);

    max_search_radius = iLx;
    if (iLy > iLx)
      max_search_radius = iLy;
    if (iLz > iLy)
      max_search_radius = iLz;

    /* Define the size of a correlation neighborhood */
    nLx = 2 * iLx + 1;
    nLy = 2 * iLy + 1;
    nLz = 2 * iLz + 1;
    nLxyz = nLx * nLy * nLz;

    /* RDF: assume resolution is the same in all 3 directions */
    ref = SubgridRX(subgrid);

    /* Note that in the construction of the covariance matrix
     * the max_search_rad is not used. Covariance depends upon
     * the correlation lengths, lambdaX/Y/Z, and the grid spacing.
     * The max_search_rad can be longer or shorter than the correlation
     * lengths. The bigger the search radius, the more accurately
     * the random field will match the correlation structure of the
     * covariance function. But the run time will increase greatly
     * as max_search_rad gets bigger because of the kriging matrix
     * that must be solved (see below).
     */
    cx = 0.0;
    cy = 0.0;
    cz = 0.0;
    if (lambdaX != 0.0)
      cx = dx * dx / (lambdaX * lambdaX);
    if (lambdaY != 0.0)
      cy = dy * dy / (lambdaY * lambdaY);
    if (lambdaZ != 0.0)
      cz = dz * dz / (lambdaZ * lambdaZ);

    for (k = 0; k < nLz; k++)
      for (j = 0; j < nLy; j++)
        for (i = 0; i < nLx; i++)
        {
          a1 = i * i * cx;
          a2 = j * j * cy;
          a3 = k * k * cz;
          cov[i][j][k] = exp(-sqrt(a1 + a2 + a3));
        }

    /* Map data to grid and shift to N(0,1) distribution.
     * Keep only points that are within max_search_radius
     * of this subgrid. */
    nc_sub = 0;
    for (n = 0; n < nc; n++)
    {
      i = (int)((x[n] - X0) / dx + 0.5);
      j = (int)((y[n] - Y0) / dy + 0.5);
      k = (int)((z[n] - Z0) / dz + 0.5);

      if ((ix - max_search_radius <= i && i <= ix + nx + max_search_radius) &&
          (iy - max_search_radius <= j && j <= iy + ny + max_search_radius) &&
          (iz - max_search_radius <= k && k <= iz + nz + max_search_radius))
      {
        ci[nc_sub] = i;
        cj[nc_sub] = j;
        ck[nc_sub] = k;
        if (lognormal)
          v_sub[nc_sub] = (log(v[n] / mean)) / sigma;
        else
          v_sub[nc_sub] = (v[n] - mean) / sigma;
        nc_sub++;
      }
    }

    if (nc_sub)
    {
      GrGeomInLoop(i, j, k, gr_geounit, ref, ix, iy, iz, nx, ny, nz,
      {
        index1 = SubvectorEltIndex(sub_field, i, j, k);

        /* Construct the input matrix and vector for kriging */
        cpts = 0;
        n = 0;
        while (n < nc_sub)
        {
          di = abs(i - ci[n]);
          dj = abs(j - cj[n]);
          dk = abs(k - ck[n]);

          if ((di + dj + dk) == 0)    /* that is, if di=dj=dk=0 */
          {
            fieldp[index1] = v_sub[n];
            n = nc_sub;
            cpts = 0;
          }

          else if ((di <= iLx) && (dj <= iLy) && (dk <= iLz))
          {
            i_array[cpts] = n;
            value[cpts] = v_sub[n];
            b[cpts++] = cov[di][dj][dk];
          }

          n++;
        }

        if (cpts > 0)
        {
          nn = 0;
          for (n = 0; n < cpts; n++)
            for (m = 0; m < cpts; m++)
            {
              di = abs(ci[i_array[n]] - ci[i_array[m]]);
              dj = abs(cj[i_array[n]] - cj[i_array[m]]);
              dk = abs(ck[i_array[n]] - ck[i_array[m]]);
              A[nn++] = cov[di][dj][dk];
            }

          /* Solve the linear system and compute the
           * conditional mean and standard deviation
           * for the RV to be simulated.*/
          cmean = 0.0;
          csigma = 0.0;
          for (n = 0; n < cpts; n++)
            w[n] = b[n];

          dpofa_(A, &cpts, &cpts, &ierr);
          dposl_(A, &cpts, &cpts, w);

          for (m = 0; m < cpts; m++)
            cmean += w[m] * value[m];

          for (m = 0; m < cpts; m++)
            csigma += w[m] * b[m];
          csigma = sqrt(cov[0][0][0] - csigma);

          fieldp[index1] = cmean + csigma * fieldp[index1];
        }     /* if(cpts ...  */
      });    /* GrGeomInLoop */
    }   /* if(nc_sub) */
  }  /* gridloop */
     /*-----------------------------------------------------------------------
      * END grid loop
      *-----------------------------------------------------------------------*/

  /* Free up local arrays */
  for (i = 0; i < nLx; i++)
  {
    for (j = 0; j < nLy; j++)
      tfree(cov[i][j]);
    tfree(cov[i]);
  }
  tfree(cov);

  tfree(b);
  tfree(w);
  tfree(value);
}


