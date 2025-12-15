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
* Routines to generate a correlated normal or log-normal
* random field using the sequential Gaussian simulator method.
* The random path has been modified for parallel processors.
*
*****************************************************************************/

#include "parflow.h"

#include <limits.h>
#include <float.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  double lambdaX;
  double lambdaY;
  double lambdaZ;
  double mean;
  double sigma;
  int seed;
  int dist_type;
  int strat_type;
  double low_cutoff;
  double high_cutoff;
  int time_index;
  int max_search_rad;
  int max_npts;
  int max_cpts;
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid   *grid;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * PGSRF
 *--------------------------------------------------------------------------*/

void         PGSRF(
                   GeomSolid *  geounit,
                   GrGeomSolid *gr_geounit,
                   Vector *     field,
                   RFCondData * cdata)
{
  /*-----------------*
  * Local variables *
  *-----------------*/
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /* Input parameters (see PGSRFNewPublicXtra() below) */
  double lambdaX = (public_xtra->lambdaX);
  double lambdaY = (public_xtra->lambdaY);
  double lambdaZ = (public_xtra->lambdaZ);
  double mean = (public_xtra->mean);
  double sigma = (public_xtra->sigma);
  int dist_type = (public_xtra->dist_type);
  double low_cutoff = (public_xtra->low_cutoff);
  double high_cutoff = (public_xtra->high_cutoff);
  int max_search_rad = (public_xtra->max_search_rad);
  int max_npts = (public_xtra->max_npts);
  int max_cpts = (public_xtra->max_cpts);
  Vector    *tmpRF = NULL;

  /* Conditioning data */
  int nc = (cdata->nc);
  double    *x = (cdata->x);
  double    *y = (cdata->y);
  double    *z = (cdata->z);
  double    *v = (cdata->v);

  /* Grid parameters */
  Grid      *grid = (instance_xtra->grid);
  Subgrid   *subgrid;
  Subvector *sub_field;
  Subvector *sub_tmpRF;

  /* Subgrid parameters */
  int nx, ny, nz;
  double dx, dy, dz;
  int nxG, nyG, nzG;

  /* Counters, indices, flags */
  int gridloop;
  int i, j, k, n, m;
  int ii, jj, kk;
  int i2, j2, k2;
  int imin, jmin, kmin;
  int rpx, rpy, rpz;
  int npts;
  int index1, index2, index3;

  /* Spatial variables */
  double    *fieldp;
  double    *tmpRFp;
  int iLx, iLy, iLz;            /* Correlation length in terms of grid points */
  int iLxp1, iLyp1, iLzp1;      /* One more than each of the above */
  int nLx, nLy, nLz;            /* Size of correlation neighborhood in grid pts. */
  int iLxyz;                    /* iLxyz = iLx*iLy*iLz */
  int nLxyz;                    /* nLxyz = nLx*nLy*nLz */
  int ix, iy, iz;
  int ref;
  int ix2, iy2, iz2;
  int i_search, j_search, k_search;
  int ci_search, cj_search, ck_search;
  double X0, Y0, Z0;

  /* Variables used in kriging  algorithm */
  double cmean, csigma;         /* Conditional mean and std. dev. from kriging */
  double A;
  double    *A_sub;             /* Sub-covariance matrix for external cond pts */
  double    *A11;               /* Submatrix; note that A11 is 1-dim */
  double    **A12, **A21, **A22;/* Submatrices for external conditioning data */
  double    **M;                /* Used as a temporary matrix */
  double    *b;                 /* Covariance vector for conditioning points */
  double    *b_tmp, *b2;
  double    *w, *w_tmp;         /* Solution vector to Aw=b */
  int       *ixx, *iyy, *izz;
  double    *value;
  int di, dj, dk;
  double uni, gau;
  double    ***cov;
  int ierr;

  /* Conditioning data variables */
  int cpts;                     /* N cond pts for a single simulated node */
  double    *cval;              /* Values for cond data for single node */

  /* Communications */
  VectorUpdateCommHandle *handle;
  int update_mode;

  /* Miscellaneous variables */
  int       **rand_path;
  char      ***marker;
  int p, r, modulus;
  double a1, a2, a3;
  double cx, cy, cz;
  double sum;

  // FIXME Shouldn't we get this from numeric_limits?
  double Tiny = 1.0e-12;

  (void)geounit;

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  tmpRF = NewVectorType(instance_xtra->grid, 1, max_search_rad, vector_cell_centered);

  /*-----------------------------------------------------------------------
   * Start sequential Gaussian simulator algorithm
   *-----------------------------------------------------------------------*/
  /* Begin timing */
  BeginTiming(public_xtra->time_index);

  /* initialize random number generators */
  SeedRand(public_xtra->seed);

  /* For now, we will assume that all subgrids have the same uniform spacing */
  subgrid = GridSubgrid(grid, 0);

  dx = SubgridDX(subgrid);
  dy = SubgridDY(subgrid);
  dz = SubgridDZ(subgrid);

  /* Size of search neighborhood through which random path must be defined */
  iLx = (int)(lambdaX / dx);
  iLy = (int)(lambdaY / dy);
  iLz = (int)(lambdaZ / dz);

  /* For computational efficiency, we'll limit the
   * size of the search neighborhood. */
  if (iLx > max_search_rad)
    iLx = max_search_rad;
  if (iLy > max_search_rad)
    iLy = max_search_rad;
  if (iLz > max_search_rad)
    iLz = max_search_rad;

  iLxp1 = iLx + 1;
  iLyp1 = iLy + 1;
  iLzp1 = iLz + 1;
  iLxyz = iLxp1 * iLyp1 * iLzp1;

  /* Define the size of a correlation neighborhood */
  nLx = 2 * iLx + 1;
  nLy = 2 * iLy + 1;
  nLz = 2 * iLz + 1;
  nLxyz = nLx * nLy * nLz;

  /*------------------------
   * Define a random path through the points in this subgrid.
   * The random path generation procedure of Srivastava and
   * Gomez has been adopted in this subroutine.  A linear
   * congruential generator of the form: r(i) = 5*r(i-1)+1 mod(2**n)
   * has a cycle length of 2**n.  By choosing the smallest power of
   * 2 that is still larger than the total number of points to be
   * simulated, the method ensures that all indices will be
   * generated once and only once.
   *------------------------*/
  rand_path = talloc(int*, iLxyz);
  for (i = 0; i < iLxyz; i++)
    rand_path[i] = talloc(int, 3);
  modulus = 2;
  while (modulus < iLxyz + 1)
    modulus *= 2;

  /* Compute a random starting node */
  p = (int)Rand();
  r = 1 + p * (iLxyz - 1);

  k = (r - 1) / (iLxp1 * iLyp1);
  j = (r - 1 - iLxp1 * iLyp1 * k) / iLxp1;
  i = (r - 1) - (k * iLyp1 + j) * iLxp1;
  rand_path[0][2] = k;
  rand_path[0][1] = j;
  rand_path[0][0] = i;

  /* Determine the next nodes */
  for (n = 1; n < iLxyz; n++)
  {
    r = (5 * r + 1) % modulus;
    while ((r < 1) || (r > iLxyz))
      r = (5 * r + 1) % modulus;

    k = ((r - 1) / (iLxp1 * iLyp1));
    j = (((r - 1) - iLxp1 * iLyp1 * k) / iLxp1);
    i = (r - 1) - (k * iLyp1 + j) * iLxp1;
    rand_path[n][0] = i;
    rand_path[n][1] = j;
    rand_path[n][2] = k;
  }

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

  /* Allocate memory for variables that will be used in kriging */
  A11 = ctalloc(double, nLxyz * nLxyz);
  A_sub = ctalloc(double, nLxyz * nLxyz);
  A12 = ctalloc(double*, nLxyz);
  A21 = ctalloc(double*, nLxyz);
  A22 = ctalloc(double*, nLxyz);
  M = ctalloc(double*, nLxyz);
  for (i = 0; i < nLxyz; i++)
  {
    A12[i] = ctalloc(double, nLxyz);
    A21[i] = ctalloc(double, nLxyz);
    A22[i] = ctalloc(double, nLxyz);
    M[i] = ctalloc(double, nLxyz);
  }

  b = ctalloc(double, nLxyz);
  b2 = ctalloc(double, nLxyz);
  b_tmp = ctalloc(double, nLxyz);
  w = ctalloc(double, nLxyz);
  w_tmp = ctalloc(double, nLxyz);
  value = ctalloc(double, nLxyz);
  cval = ctalloc(double, nLxyz);
  ixx = ctalloc(int, nLxyz);
  iyy = ctalloc(int, nLxyz);
  izz = ctalloc(int, nLxyz);

  /* Allocate space for the "marker" used to keep track of which
   * points in a representative correlation box have been simulated
   * already.
   */
  marker = talloc(char**, (3 * iLx + 1));
  marker += iLx;
  for (i = -iLx; i <= 2 * iLx; i++)
  {
    marker[i] = talloc(char*, (3 * iLy + 1));
    marker[i] += iLy;
    for (j = -iLy; j <= 2 * iLy; j++)
    {
      marker[i][j] = ctalloc(char, (3 * iLz + 1));
      marker[i][j] += iLz;
      for (k = -iLz; k <= 2 * iLz; k++)
        marker[i][j][k] = 0;
    }
  }

  /* Convert the cutoff values to a gaussian if they're lognormal on input */
  if ((dist_type == 1) || (dist_type == 3))
  {
    if (low_cutoff <= 0.0)
    {
      low_cutoff = Tiny;
    }
    else
    {
      low_cutoff = (log(low_cutoff / mean)) / sigma;
    }

    if (high_cutoff <= 0.0)
    {
      high_cutoff = DBL_MAX;
    }
    else
    {
      high_cutoff = (log(high_cutoff / mean)) / sigma;
    }
  }

  /*--------------------------------------------------------------------
   * Start pGs algorithm
   *--------------------------------------------------------------------*/
  for (gridloop = 0; gridloop < GridNumSubgrids(grid); gridloop++)
  {
    subgrid = GridSubgrid(grid, gridloop);
    sub_tmpRF = VectorSubvector(tmpRF, gridloop);
    sub_field = VectorSubvector(field, gridloop);
    tmpRFp = SubvectorData(sub_tmpRF);
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

    /* RDF: assume resolution is the same in all 3 directions */
    ref = SubgridRX(subgrid);

    /* Initialize tmpRF vector */
    GrGeomInLoop(i, j, k, gr_geounit, ref, ix, iy, iz, nx, ny, nz,
    {
      index2 = SubvectorEltIndex(sub_tmpRF, i, j, k);
      tmpRFp[index2] = 0.0;
    });

    /* Convert conditioning data to N(0,1)  distribution if
     * it's assumed to be lognormal. Then copy it into tmpRFp */
    if ((dist_type == 1) || (dist_type == 3))
    {
      for (n = 0; n < nc; n++)
      {
        i = (int)((x[n] - X0) / dx + 0.5);
        j = (int)((y[n] - Y0) / dy + 0.5);
        k = (int)((z[n] - Z0) / dz + 0.5);

        if ((ix - max_search_rad <= i && i <= ix + nx + max_search_rad) &&
            (iy - max_search_rad <= j && j <= iy + ny + max_search_rad) &&
            (iz - max_search_rad <= k && k <= iz + nz + max_search_rad))
        {
          index2 = SubvectorEltIndex(sub_tmpRF, i, j, k);
          if (v[n] <= 0.0)
            tmpRFp[index2] = Tiny;
          else
            tmpRFp[index2] = (log(v[n] / mean)) / sigma;
        }
      }
    }

    /* Otherwise, shift data to N(0,1) distribution */
    else
    {
      for (n = 0; n < nc; n++)
      {
        i = (int)((x[n] - X0) / dx + 0.5);
        j = (int)((y[n] - Y0) / dy + 0.5);
        k = (int)((z[n] - Z0) / dz + 0.5);

        if ((ix - max_search_rad <= i && i <= ix + nx + max_search_rad) &&
            (iy - max_search_rad <= j && j <= iy + ny + max_search_rad) &&
            (iz - max_search_rad <= k && k <= iz + nz + max_search_rad))
        {
          index2 = SubvectorEltIndex(sub_tmpRF, i, j, k);
          tmpRFp[index2] = (v[n] - mean) / sigma;
        }
      }
    }

    /* Set the search radii in each direction. If the maximum
     * number of points in a neighborhood is exceeded, these limits
     * will be reduced. */
    i_search = iLx;
    j_search = iLy;
    k_search = iLz;

    /* Compute values at all points using all templates */
    for (n = 0; n < iLxyz; n++)
    {
      /* Update the ghost layer before proceeding */
      if (n > 0)
      {
        /* First reset max_search_radius */
        max_search_rad = i_search;
        if (j_search > max_search_rad)
          max_search_rad = j_search;
        if (k_search > max_search_rad)
          max_search_rad = k_search;

        /* Reset the comm package based on the new max_search_radius */
        if (max_search_rad == 1)
          update_mode = VectorUpdatePGS1;
        else if (max_search_rad == 2)
          update_mode = VectorUpdatePGS2;
        else if (max_search_rad == 3)
          update_mode = VectorUpdatePGS3;
        else
          update_mode = VectorUpdatePGS4;

        handle = InitVectorUpdate(tmpRF, update_mode);
        FinalizeVectorUpdate(handle);
      }

      rpx = rand_path[n][0];
      rpy = rand_path[n][1];
      rpz = rand_path[n][2];
      ix2 = rpx;  while (ix2 < ix)
        ix2 += iLxp1;
      iy2 = rpy;  while (iy2 < iy)
        iy2 += iLyp1;
      iz2 = rpz;  while (iz2 < iz)
        iz2 += iLzp1;

      /* This if clause checks to see if there are, in fact,
       * any points at all in this subgrid, for this
       * particular region. Note that each value of n in the
       * above n-loop corresponds to a different region. */
      if ((ix2 < ix + nx) && (iy2 < iy + ny) && (iz2 < iz + nz))
      {
        /*
         * Construct the input matrix and vector for kriging,
         * solve the linear system, and compute csigma.
         * These depend only on the spatial distribution of
         * conditioning data, not on the actual values of
         * the data. Only the conditional mean (cmean) depends
         * on actual values, so it must be computed for every
         * point. Thus, it's found within the pgs_Boxloop below.
         * The size of the linear system that must be solved here
         * will be no larger than (2r+1)^3, where r=max_search_rad.
         * It is clear from this why it is necessary to limit
         * the size of the search radius.
         */

        /* Here the marker array indicates which points within
         * the search radius have been simulated already. This
         * spatial pattern of conditioning points will be the
         * same for every point in the current template. Thus,
         * this system can be solved once *outside* of the
         * GrGeomInLoop2 below. */

        npts = 9999;
        while (npts > max_npts)
        {
          m = 0;
          /* Count the number of points in search ellipse */
          for (k = rpz - k_search; k <= rpz + k_search; k++)
            for (j = rpy - j_search; j <= rpy + j_search; j++)
              for (i = rpx - i_search; i <= rpx + i_search; i++)
              {
                if (marker[i][j][k])
                {
                  ixx[m] = i;
                  iyy[m] = j;
                  izz[m++] = k;
                }
              }
          npts = m;

          /* If npts is too large, reduce the size of the
           * search ellipse one axis at a time. */
          if (npts > max_npts)
          {
            /* If i_search is the biggest, reduce it by one. */
            if ((i_search >= j_search) && (i_search >= k_search))
            {
              i_search--;
            }

            /* Or, if j_search is the biggest, reduce it by one. */
            else if ((j_search >= i_search) && (j_search >= k_search))
            {
              j_search--;
            }

            /* Otherwise, reduce k_search by one. */
            else
            {
              k_search--;
            }
          }
        }

        m = 0;
        for (j = 0; j < npts; j++)
        {
          di = abs(rpx - ixx[j]);
          dj = abs(rpy - iyy[j]);
          dk = abs(rpz - izz[j]);
          b[j] = cov[di][dj][dk];

          for (i = 0; i < npts; i++)
          {
            di = abs(ixx[i] - ixx[j]);
            dj = abs(iyy[i] - iyy[j]);
            dk = abs(izz[i] - izz[j]);
            A11[m++] = cov[di][dj][dk];
          }
        }

        /* Solve the linear system */
        for (i = 0; i < npts; i++)
          w[i] = b[i];

        if (npts > 0)
        {
          dpofa_(A11, &npts, &npts, &ierr);
          dposl_(A11, &npts, &npts, w);
        }

        /* Compute the conditional standard deviation for the RV
         * to be simulated. */
        csigma = 0.0;
        for (i = 0; i < npts; i++)
          csigma += w[i] * b[i];
        csigma = sqrt(cov[0][0][0] - csigma);

        /* The following loop hits every point in the current
         * region. That is, it skips by max_search_rad+1
         * through the subgrid. In this way, all the points
         * in this loop may simulated simultaneously; each is
         * outside the search radius of all the others. */
        nxG = (nx + ix);
        nyG = (ny + iy);
        nzG = (nz + iz);

        for (k = iz2; k < nzG; k += iLzp1)
          for (j = iy2; j < nyG; j += iLyp1)
            for (i = ix2; i < nxG; i += iLxp1)
            {
              index1 = SubvectorEltIndex(sub_field, i, j, k);
              index2 = SubvectorEltIndex(sub_tmpRF, i, j, k);

              /* Only simulate points in this geounit and that don't
               * already have a value. If a node already has a value,
               * it was assigned as external conditioning data,
               * so we don't need to simulate it.  */
              if (fabs(tmpRFp[index2]) < Tiny)
              {
                /* Condition the random variable */
                m = 0;
                cpts = 0;

                for (kk = -k_search; kk <= k_search; kk++)
                  for (jj = -j_search; jj <= j_search; jj++)
                    for (ii = -i_search; ii <= i_search; ii++)
                    {
                      value[m] = 0.0;
                      index3 = SubvectorEltIndex(sub_tmpRF, i + ii, j + jj, k + kk);

                      if (marker[ii + rpx][jj + rpy][kk + rpz])
                      {
                        value[m++] = tmpRFp[index3];
                      }

                      /* In this case, there is a value at this point,
                       * but it wasn't simulated yet (as indicated by the
                       * fact that the marker has no place for it). Thus,
                       * it must be external conditioning data.  */
                      else if (fabs(tmpRFp[index3]) > Tiny)
                      {
                        ixx[npts + cpts] = rpx + ii;
                        iyy[npts + cpts] = rpy + jj;
                        izz[npts + cpts] = rpz + kk;
                        cval[cpts++] = tmpRFp[index3];
                      }
                    }

                /* If cpts is too large, reduce the size of the
                 * search neighborhood, one axis at a time. */
                /* Define the size of the search neighborhood */
                ci_search = i_search;
                cj_search = j_search;
                ck_search = k_search;
                while (cpts > max_cpts)
                {
                  /* If ci_search is the biggest, reduce it by one. */
                  if ((ci_search >= cj_search) && (ci_search >= ck_search))
                    ci_search--;

                  /* Or, if cj_search is the biggest, reduce it by one. */
                  else if ((cj_search >= ci_search) && (cj_search >= ck_search))
                    cj_search--;

                  /* Otherwise, reduce ck_search by one. */
                  else
                    ck_search--;

                  /* Now recount the conditioning data points */
                  m = 0;
                  cpts = 0;
                  for (kk = -ck_search; kk <= ck_search; kk++)
                    for (jj = -cj_search; jj <= cj_search; jj++)
                      for (ii = -ci_search; ii <= ci_search; ii++)
                      {
                        index3 = SubvectorEltIndex(sub_tmpRF, i + ii, j + jj, k + kk);

                        if (!(marker[rpx + ii][rpy + jj][rpz + kk]) &&
                            (fabs(tmpRFp[index3]) > Tiny))
                        {
                          ixx[npts + cpts] = rpx + ii;
                          iyy[npts + cpts] = rpy + jj;
                          izz[npts + cpts] = rpz + kk;
                          cval[cpts++] = tmpRFp[index3];
                        }
                      }
                }

                for (i2 = 0; i2 < npts; i2++)
                  w_tmp[i2] = w[i2];

                /*--------------------------------------------------
                 * Conditioning to external data is done here.
                 *--------------------------------------------------*/
                if (cpts > 0)
                {
                  /* Compute the submatrices */
                  for (j2 = 0; j2 < npts + cpts; j2++)
                  {
                    di = abs(rpx - ixx[j2]);
                    dj = abs(rpy - iyy[j2]);
                    dk = abs(rpz - izz[j2]);
                    b[j2] = cov[di][dj][dk];

                    for (i2 = 0; i2 < npts + cpts; i2++)
                    {
                      di = abs(ixx[i2] - ixx[j2]);
                      dj = abs(iyy[i2] - iyy[j2]);
                      dk = abs(izz[i2] - izz[j2]);
                      A = cov[di][dj][dk];
                      if (i2 < npts && j2 >= npts)
                        A12[i2][j2 - npts] = A;
                      if (i2 >= npts && j2 < npts)
                        A21[i2 - npts][j2] = A;
                      if (i2 >= npts && j2 >= npts)
                        A22[i2 - npts][j2 - npts] = A;
                    }
                  }

                  /* Compute b2' = b2 - A21 * A11_inv * b1 and augment b1 */

                  // GCC 15.2.0 Was issuing warning about reading  at offset [-17179869184, -8] into source object of size [8, 17179869176] allocated by ‘calloc’
                  // Seems related to npts possibly being negative?, could avoid by putting an if (npts >= 0) arount this loop as well.

#if defined(__GNUC__) && !defined(__clang__)
                  /* GCC version check: major >= 15 */
  #if __GNUC__ > 15 || (__GNUC__ == 15 && __GNUC_MINOR__ >= 0)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstringop-overread"
  #endif
#endif
                  for (i2 = 0; i2 < cpts; i2++)
                    b2[i2] = b[i2 + npts];


#if defined(__GNUC__) && !defined(__clang__)
  #if __GNUC__ > 15 || (__GNUC__ == 15 && __GNUC_MINOR__ >= 0)
    #pragma GCC diagnostic pop
  #endif
#endif

                  for (i2 = 0; i2 < npts; i2++)
                    b_tmp[i2] = b[i2];
                  dposl_(A11, &npts, &npts, b_tmp);

                  for (i2 = 0; i2 < cpts; i2++)
                  {
                    sum = 0.0;
                    for (j2 = 0; j2 < npts; j2++)
                    {
                      sum += A21[i2][j2] * b_tmp[j2];
                    }
                    b2[i2] -= sum;
                  }
                  for (i2 = 0; i2 < cpts; i2++)
                    b[i2 + npts] = b2[i2];

                  /* Compute A22' = A22 - A21 * A11_inv * A12 */
                  for (j2 = 0; j2 < cpts; j2++)
                    for (i2 = 0; i2 < npts; i2++)
                      M[j2][i2] = A12[i2][j2];

                  if (npts > 0)
                  {
                    for (i2 = 0; i2 < cpts; i2++)
                      dposl_(A11, &npts, &npts, M[i2]);
                  }

                  for (j2 = 0; j2 < cpts; j2++)
                    for (i2 = 0; i2 < cpts; i2++)
                    {
                      sum = 0.0;
                      for (k2 = 0; k2 < npts; k2++)
                        sum += A21[i2][k2] * M[j2][k2];
                      A22[i2][j2] -= sum;
                    }

                  m = 0;
                  for (j2 = 0; j2 < cpts; j2++)
                    for (i2 = 0; i2 < cpts; i2++)
                      A_sub[m++] = A22[i2][j2];

                  /* Compute x2 where A22*x2 = b2' */
                  dpofa_(A_sub, &cpts, &cpts, &ierr);
                  dposl_(A_sub, &cpts, &cpts, b2);

                  /* Compute w_tmp where A11*w_tmp = (b1 - A12*b2) */
                  if (npts > 0)
                  {
                    for (i2 = 0; i2 < npts; i2++)
                    {
                      sum = 0.0;
                      for (k2 = 0; k2 < cpts; k2++)
                        sum += A12[i2][k2] * b2[k2];
                      w_tmp[i2] = b[i2] - sum;
                    }
                    dposl_(A11, &npts, &npts, w_tmp);
                  }

                  /* Fill in the rest of w_tmp with b2 */
                  for (i2 = npts; i2 < npts + cpts; i2++)
                  {
                    w_tmp[i2] = b2[i2];
                    value[i2] = cval[i2 - npts];
                  }

                  /* Recompute csigma */
                  csigma = 0.0;
                  for (i2 = 0; i2 < npts + cpts; i2++)
                    csigma += w_tmp[i2] * b[i2];
                  csigma = sqrt(cov[0][0][0] - csigma);
                }
                /*--------------------------------------------------
                 * End of external conditioning
                 *--------------------------------------------------*/
                cmean = 0.0;
                for (m = 0; m < npts + cpts; m++)
                  cmean += w_tmp[m] * value[m];

                /* uni = fieldp[index1]; */
                uni = Rand();
                gauinv_(&uni, &gau, &ierr);
                tmpRFp[index2] = csigma * gau + cmean;

                /* Cutoff tail values if required */
                if (dist_type > 1)
                {
                  if (tmpRFp[index2] < low_cutoff)
                    tmpRFp[index2] = low_cutoff;
                  if (tmpRFp[index2] > high_cutoff)
                    tmpRFp[index2] = high_cutoff;
                }
              }        /* if( abs(tmpRFp[index2]) < Tiny )  */
            }
        /* end of triple for-loops over i,j,k  */

        /* Update the marker vector */
        imin = rpx - iLxp1; if (imin < -iLx)
          imin += iLxp1;
        jmin = rpy - iLyp1; if (jmin < -iLy)
          jmin += iLyp1;
        kmin = rpz - iLzp1; if (kmin < -iLz)
          kmin += iLzp1;

        for (kk = kmin; kk <= 2 * iLz; kk += iLzp1)
          for (jj = jmin; jj <= 2 * iLy; jj += iLyp1)
            for (ii = imin; ii <= 2 * iLx; ii += iLxp1)
            {
              marker[ii][jj][kk] = 1;
            }
      }     /* if(...) */
    }   /* n loop */

    /* Make log-normal if requested. Note that low
     * and high cutoffs are already accomplished. */
    if ((dist_type == 1) || (dist_type == 3))
    {
      GrGeomInLoop(i, j, k, gr_geounit, ref, ix, iy, iz, nx, ny, nz,
      {
        index1 = SubvectorEltIndex(sub_field, i, j, k);
        index2 = SubvectorEltIndex(sub_tmpRF, i, j, k);
        fieldp[index1] = mean * exp((sigma) * tmpRFp[index2]);
      });
    }

    /* Shift the Gaussian distribution */
    else if ((dist_type == 0) || (dist_type == 2))
    {
      GrGeomInLoop(i, j, k, gr_geounit, ref, ix, iy, iz, nx, ny, nz,
      {
        index1 = SubvectorEltIndex(sub_field, i, j, k);
        index2 = SubvectorEltIndex(sub_tmpRF, i, j, k);
        fieldp[index1] = mean + sigma * tmpRFp[index2];
      });
    }
  }  /* gridloop */

  /*-----------------------------------------------------------------------
   * END grid loop
   *-----------------------------------------------------------------------*/
  /* Free up local arrays */
  for (i = 0; i < iLxyz; i++)
    tfree(rand_path[i]);
  tfree(rand_path);

  for (i = 0; i < nLx; i++)
  {
    for (j = 0; j < nLy; j++)
      tfree(cov[i][j]);
    tfree(cov[i]);
  }
  tfree(cov);

  for (i = 0; i < nLxyz; i++)
  {
    tfree(A12[i]);
    tfree(A21[i]);
    tfree(A22[i]);
    tfree(M[i]);
  }
  tfree(A11);
  tfree(A_sub);
  tfree(A12);
  tfree(A21);
  tfree(A22);
  tfree(M);
  tfree(b);
  tfree(b2);
  tfree(b_tmp);
  tfree(w);
  tfree(w_tmp);
  tfree(value);
  tfree(cval);
  tfree(ixx);
  tfree(iyy);
  tfree(izz);

  for (i = -iLx; i <= 2 * iLx; i++)
  {
    for (j = -iLy; j <= 2 * iLy; j++)
    {
      tfree(marker[i][j] - iLz);
    }
    tfree(marker[i] - iLy);
  }
  tfree(marker - iLx);

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(tmpRF);

  /* End timing */
  EndTiming(public_xtra->time_index);
}


/*--------------------------------------------------------------------------
 * PGSRFInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PGSRFInitInstanceXtra(
                                 Grid *  grid,
                                 double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  (void)temp_data;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/
  if (grid != NULL)
  {
    /* set new data */
    (instance_xtra->grid) = grid;
  }


  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PGSRFFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PGSRFFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PGSRFNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PGSRFNewPublicXtra(char *geom_name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  char key[IDB_MAX_KEY_LEN];
  char *tmp;

  NameArray strat_type_na;
  NameArray log_normal_na;

  public_xtra = ctalloc(PublicXtra, 1);

  sprintf(key, "Geom.%s.Perm.LambdaX", geom_name);
  public_xtra->lambdaX = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LambdaY", geom_name);
  public_xtra->lambdaY = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.LambdaZ", geom_name);
  public_xtra->lambdaZ = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.GeomMean", geom_name);
  public_xtra->mean = GetDouble(key);
  sprintf(key, "Geom.%s.Perm.Sigma", geom_name);
  public_xtra->sigma = GetDouble(key);

  sprintf(key, "Geom.%s.Perm.Seed", geom_name);
  public_xtra->seed = GetInt(key);

  sprintf(key, "Geom.%s.Perm.MaxNPts", geom_name);
  public_xtra->max_npts = GetInt(key);

  sprintf(key, "Geom.%s.Perm.MaxCpts", geom_name);
  public_xtra->max_cpts = GetInt(key);

  log_normal_na = NA_NewNameArray("Normal Log NormalTruncated LogTruncated");
  sprintf(key, "Geom.%s.Perm.LogNormal", geom_name);
  tmp = GetString(key);
  public_xtra->dist_type = NA_NameToIndexExitOnError(log_normal_na, tmp, key);
  NA_FreeNameArray(log_normal_na);

  strat_type_na = NA_NewNameArray("Horizontal Bottom Top");
  sprintf(key, "Geom.%s.Perm.StratType", geom_name);
  tmp = GetString(key);
  public_xtra->strat_type = NA_NameToIndexExitOnError(strat_type_na, tmp, key);
  NA_FreeNameArray(strat_type_na);


  if (public_xtra->dist_type > 1)
  {
    sprintf(key, "Geom.%s.Perm.LowCutoff", geom_name);
    public_xtra->low_cutoff = GetDouble(key);

    sprintf(key, "Geom.%s.Perm.HighCutoff", geom_name);
    public_xtra->high_cutoff = GetDouble(key);
  }

  /* The maximum search radius is currently limited
   * by the PGSRF routine. It is set here to allow
   * possible user control in the future. */
  /*(public_xtra -> max_search_rad) = 4;
   * added max search rad as a key to improve correlation structure of
   * RF in testing
   */
  sprintf(key, "Geom.%s.Perm.MaxSearchRad", geom_name);
  public_xtra->max_search_rad = GetInt(key);

  (public_xtra->time_index) = RegisterTiming("PGS RF");

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PGSRFFreePublicXtra
 *--------------------------------------------------------------------------*/

void  PGSRFFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * PGSRFSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PGSRFSizeOfTempData()
{
  int size = 0;

  return(size);
}

