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

#include "parflow.h"

#include <string.h>
#include <float.h>
#include <assert.h>
#include <math.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int type;     /* input type */
  void  *data;  /* pointer to Type structure */

  NameArray regions;
} PublicXtra;

typedef struct {
  Grid    *grid;

  double  *temp_data;
} InstanceXtra;

typedef struct {
  int num_regions;
  int    *region_indices;
  double *values;
} Type0;

typedef struct {
  double min_pressure_head;
  double h_s;
  int num_sample_points;

  double *x;
  double *a;          /* S(h) values */
  double *d;          /* spline slopes for S */
  double *a_der;      /* dS/dh values */
  double *d_der;      /* spline slopes for dS/dh */

  /* used by linear interpolation method */
  double *slope;
  double *slope_der;

  int interpolation_method;

  double interval;

  double s_res;
  double s_sat;
} SatTable;

typedef struct {
  int num_regions;
  int    *region_indices;
  int data_from_file;
  char   *alpha_file;
  char   *n_file;
  char   *s_sat_file;
  char   *s_res_file;
  double *alphas;
  double *ns;
  double *s_ress;
  double *s_difs;
  double *h_s_values;           /* per-region Ippisch air-entry head */
  SatTable **lookup_tables;     /* per-region saturation lookup tables */
  Vector *alpha_values;
  Vector *n_values;
  Vector *s_res_values;
  Vector *s_sat_values;
} Type1;                      /* Van Genuchten Saturation Curve */

typedef struct {
  int num_regions;
  int    *region_indices;
  double *alphas;
  double *betas;
  double *s_ress;
  double *s_difs;
} Type2;                      /* Haverkamp et.al. Saturation Curve */

typedef struct {
  int num_regions;
  int    *region_indices;
} Type3;                      /* Data points for Saturation Curve */

typedef struct {
  int num_regions;
  int     *region_indices;
  int     *degrees;
  double **coefficients;
} Type4;                      /* Polynomial function for Saturation Curve */

typedef struct {
  char    *filename;

  Vector  *satRF;
} Type5;                      /* Spatially varying field over entire domain
                               * read from a file */

/*--------------------------------------------------------------------------
 * SatComputeTable:
 *    Builds a saturation lookup table mirroring VanGComputeTable pattern.
 *--------------------------------------------------------------------------*/

SatTable *SatComputeTable(
                          int    interpolation_method,
                          int    num_sample_points,
                          double min_pressure_head,
                          double alpha,
                          double n,
                          double h_s,
                          double s_res,
                          double s_sat)
{
  double *x, *a, *d, *a_der, *d_der;

  SatTable *new_table = ctalloc(SatTable, 1);

  new_table->interpolation_method = interpolation_method;
  new_table->num_sample_points = num_sample_points;
  new_table->min_pressure_head = min_pressure_head;
  new_table->h_s = h_s;
  new_table->s_res = s_res;
  new_table->s_sat = s_sat;

  new_table->x = ctalloc(double, num_sample_points + 1);
  new_table->a = ctalloc(double, num_sample_points + 1);
  new_table->d = ctalloc(double, num_sample_points + 1);
  new_table->a_der = ctalloc(double, num_sample_points + 1);
  new_table->d_der = ctalloc(double, num_sample_points + 1);

  if (interpolation_method == 1)
  {
    new_table->slope = ctalloc(double, num_sample_points + 1);
    new_table->slope_der = ctalloc(double, num_sample_points + 1);
  }

  x = new_table->x;
  a = new_table->a;
  d = new_table->d;
  a_der = new_table->a_der;
  d_der = new_table->d_der;

  /* Heap-allocated workspace (avoids ~MB-sized VLA at large num_sample_points) */
  double *h = ctalloc(double, num_sample_points + 1);
  double *f = ctalloc(double, num_sample_points + 1);
  double *del = ctalloc(double, num_sample_points + 1);
  double *f_der = ctalloc(double, num_sample_points + 1);
  double *del_der = ctalloc(double, num_sample_points + 1);
  double alph, beta, magn;
  int index;
  double interval, m;

  double s_dif = s_sat - s_res;
  m = 1.0e0 - (1.0e0 / n);

  /* Ippisch: precompute Scs (1.0 when h_s = 0) */
  double Scs = 1.0;
  if (h_s > 0.0)
  {
    Scs = pow(1.0 + pow(alpha * h_s, n), -m);
  }

  /* Table domain: h_s to fabs(min_pressure_head), evenly spaced over
   * num_sample_points segments using num_sample_points + 1 sample points
   * (indices 0..num_sample_points). Use range/N (not range/(N-1)) so that
   * x[num_sample_points] lands exactly on |min_pressure_head|. */
  double table_range = fabs(min_pressure_head) - h_s;
  interval = table_range / (double)num_sample_points;
  new_table->interval = interval;

  for (index = 0; index <= num_sample_points; index++)
  {
    x[index] = h_s + index * interval;

    double Sc = pow(1.0 + pow(alpha * x[index], n), -m);
    /* S(h) with Ippisch normalization */
    a[index] = s_dif * (Sc / Scs) + s_res;

    /* dS/dh (positive magnitude, matching existing convention) */
    a_der[index] = (m * n * alpha * pow(alpha * x[index], (n - 1))) * s_dif
                   / (Scs * pow(1.0 + pow(alpha * x[index], n), m + 1));
  }

  /* Fill in slope for linear interpolation */
  if (interpolation_method == 1)
  {
    for (index = 0; index < num_sample_points; index++)
    {
      new_table->slope[index] = (a[index + 1] - a[index]) /
                                new_table->interval;
      new_table->slope_der[index] = (a_der[index + 1] - a_der[index]) /
                                    new_table->interval;
    }
  }

  /* Monotonic Hermite spline (Fritsch & Carlson 1980) */
  memset(del, 0, (num_sample_points + 1) * sizeof(double));
  for (index = 0; index < num_sample_points; index++)
  {
    h[index] = x[index + 1] - x[index];
    f[index] = a[index + 1] - a[index];
    del[index] = f[index] / h[index];
    f_der[index] = a_der[index + 1] - a_der[index];
    del_der[index] = f_der[index] / h[index];
  }
  d[0] = del[0];
  d[num_sample_points] = del[num_sample_points - 1];
  d_der[0] = del_der[0];
  d_der[num_sample_points] = del_der[num_sample_points - 1];

  for (index = 1; index < num_sample_points; index++)
  {
    d[index] = (del[index - 1] + del[index]) / 2;
    d_der[index] = (del_der[index - 1] + del_der[index]) / 2;
  }

  for (index = 0; index < num_sample_points; index++)
  {
    if (del[index] == 0.0)
    {
      d[index] = 0;
      d[index + 1] = 0;
    }
    else
    {
      alph = d[index] / del[index];
      beta = d[index + 1] / del[index];
      magn = pow(alph, 2) + pow(beta, 2);
      if (magn > 9.0)
      {
        d[index] = 3 * alph * del[index] / magn;
        d[index + 1] = 3 * beta * del[index] / magn;
      }
    }

    if (del_der[index] == 0.0)
    {
      d_der[index] = 0;
      d_der[index + 1] = 0;
    }
    else
    {
      alph = d_der[index] / del_der[index];
      beta = d_der[index + 1] / del_der[index];
      magn = pow(alph, 2) + pow(beta, 2);
      if (magn > 9.0)
      {
        d_der[index] = 3 * alph * del_der[index] / magn;
        d_der[index + 1] = 3 * beta * del_der[index] / magn;
      }
    }
  }

  tfree(h);
  tfree(f);
  tfree(del);
  tfree(f_der);
  tfree(del_der);

  return new_table;
}

__host__ __device__
static inline double SatLookupSpline(
                                     double    pressure_head,
                                     SatTable *lookup_table,
                                     int       fcn)
{
  double sat, t;
  int pt = 0;
  int num_sample_points = lookup_table->num_sample_points;
  double min_pressure_head = lookup_table->min_pressure_head;
  double h_s = lookup_table->h_s;

  assert(pressure_head >= 0);

  /* Air-entry zone: saturated */
  if (pressure_head <= h_s)
  {
    if (fcn == CALCFCN)
      return lookup_table->s_sat;
    else
      return 0.0;
  }

  /* Beyond dry end of table */
  if (pressure_head >= fabs(min_pressure_head))
  {
    if (fcn == CALCFCN)
      return lookup_table->s_res;
    else
      return 0.0;
  }

  double interval = lookup_table->interval;
  pt = (int)floor((pressure_head - h_s) / interval);
  /* Clamp to last valid spline interval [pt, pt+1]: pt in [0, num_sample_points-1] */
  if (pt >= num_sample_points)
  {
    pt = num_sample_points - 1;
  }

  double x = lookup_table->x[pt];
  double a = lookup_table->a[pt];
  double d = lookup_table->d[pt];
  double a_der = lookup_table->a_der[pt];
  double d_der = lookup_table->d_der[pt];

  if (fcn == CALCFCN)
  {
    t = (pressure_head - x) / (lookup_table->x[pt + 1] - x);
    sat = (2.0 * pow(t, 3) - 3.0 * pow(t, 2) + 1.0) * a
          + (pow(t, 3) - 2.0 * pow(t, 2) + t)
          * (lookup_table->x[pt + 1] - x) * d + (-2.0 * pow(t, 3)
                                                 + 3.0 * pow(t, 2)) * (lookup_table->a[pt + 1])
          + (pow(t, 3) - pow(t, 2)) * (lookup_table->x[pt + 1] - x)
          * (lookup_table->d[pt + 1]);
  }
  else
  {
    t = (pressure_head - x) / (lookup_table->x[pt + 1] - x);
    sat = (2.0 * pow(t, 3) - 3.0 * pow(t, 2) + 1.0) * a_der
          + (pow(t, 3) - 2.0 * pow(t, 2) + t)
          * (lookup_table->x[pt + 1] - x) * d_der + (-2.0 * pow(t, 3)
                                                     + 3.0 * pow(t, 2)) * (lookup_table->a_der[pt + 1])
          + (pow(t, 3) - pow(t, 2)) * (lookup_table->x[pt + 1] - x)
          * (lookup_table->d_der[pt + 1]);
  }

  return sat;
}

__host__ __device__
static inline double SatLookupLinear(
                                     double    pressure_head,
                                     SatTable *lookup_table,
                                     int       fcn)
{
  int pt = 0;
  int num_sample_points = lookup_table->num_sample_points;
  double min_pressure_head = lookup_table->min_pressure_head;
  double h_s = lookup_table->h_s;

  assert(pressure_head >= 0);

  /* Air-entry zone: saturated */
  if (pressure_head <= h_s)
  {
    if (fcn == CALCFCN)
      return lookup_table->s_sat;
    else
      return 0.0;
  }

  /* Beyond dry end of table */
  if (pressure_head >= fabs(min_pressure_head))
  {
    if (fcn == CALCFCN)
      return lookup_table->s_res;
    else
      return 0.0;
  }

  double interval = lookup_table->interval;
  pt = (int)floor((pressure_head - h_s) / interval);
  /* Clamp to last valid linear segment [pt, pt+1]: pt in [0, num_sample_points-1] */
  if (pt >= num_sample_points)
  {
    pt = num_sample_points - 1;
  }

  double x = lookup_table->x[pt];

  if (fcn == CALCFCN)
  {
    return lookup_table->a[pt] + lookup_table->slope[pt] * (pressure_head - x);
  }
  else
  {
    return lookup_table->a_der[pt] + lookup_table->slope_der[pt] * (pressure_head - x);
  }
}

/*--------------------------------------------------------------------------
 * Saturation:
 *    This routine returns a Vector of saturations based on pressures.
 *--------------------------------------------------------------------------*/

void     Saturation(
                    Vector *     phase_saturation, /* Vector of return saturations */
                    Vector *     phase_pressure, /* Vector of pressures */
                    Vector *     phase_density, /* Vector of densities */
                    double       gravity, /* Magnitude of gravity in neg. z direction */
                    ProblemData *problem_data, /* Contains geometry info. for the problem */
                    int          fcn) /* Flag determining what to calculate
                                       * fcn = CALCFCN => calculate the function
                                       *                  value
                                       * fcn = CALCDER => calculate the function
                                       *                  derivative */
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0         *dummy0;
  Type1         *dummy1;
  Type2         *dummy2;
  Type3         *dummy3;
  Type4         *dummy4;
  Type5         *dummy5;

  Grid          *grid = VectorGrid(phase_saturation);

  GrGeomSolid   *gr_solid, *gr_domain;

  Subvector     *ps_sub;
  Subvector     *pp_sub;
  Subvector     *pd_sub;
  Subvector     *satRF_sub;
  Subvector     *n_values_sub;
  Subvector     *alpha_values_sub;
  Subvector     *s_res_values_sub;
  Subvector     *s_sat_values_sub;

  double        *psdat, *ppdat, *pddat, *satRFdat;
  double        *n_values_dat, *alpha_values_dat;
  double        *s_res_values_dat, *s_sat_values_dat;

  SubgridArray  *subgrids = GridSubgrids(grid);

  Subgrid       *subgrid;

  int sg;

  int ix, iy, iz, r;
  int nx, ny, nz;

  int i, j, k;

  int            *region_indices, num_regions, ir;

  /* Initialize saturations */

// SGS FIXME why is this needed?
#undef max
  InitVectorAll(phase_saturation, -FLT_MAX);

  switch ((public_xtra->type))
  {
    case 0: /* Constant saturation */
    {
      double  *values;
      int ir;

      dummy0 = (Type0*)(public_xtra->data);

      num_regions = (dummy0->num_regions);
      region_indices = (dummy0->region_indices);
      values = (dummy0->values);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

        ForSubgridI(sg, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, sg);
          ps_sub = VectorSubvector(phase_saturation, sg);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          psdat = SubvectorData(ps_sub);

          if (fcn == CALCFCN)
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              psdat[ips] = values[ir];
            });
          }
          else   /* fcn = CALCDER */
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              psdat[ips] = 0.0;
            });
          }     /* End else clause */
        }       /* End subgrid loop */
      }         /* End reion loop */
      break;
    }        /* End case 0 */

    case 1: /* Van Genuchten saturation curve */
    {
      int data_from_file;
      double *alphas, *ns, *s_ress, *s_difs, *h_s_values;

      Vector *n_values, *alpha_values, *s_res_values, *s_sat_values;

      dummy1 = (Type1*)(public_xtra->data);

      num_regions = (dummy1->num_regions);
      region_indices = (dummy1->region_indices);
      alphas = (dummy1->alphas);
      ns = (dummy1->ns);
      s_ress = (dummy1->s_ress);
      s_difs = (dummy1->s_difs);
      h_s_values = (dummy1->h_s_values);
      data_from_file = (dummy1->data_from_file);

      if (data_from_file == 0) /* Soil parameters given by region */
      {
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(sg, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, sg);
            ps_sub = VectorSubvector(phase_saturation, sg);
            pp_sub = VectorSubvector(phase_pressure, sg);
            pd_sub = VectorSubvector(phase_density, sg);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            r = SubgridRX(subgrid);

            psdat = SubvectorData(ps_sub);
            ppdat = SubvectorData(pp_sub);
            pddat = SubvectorData(pd_sub);

            if (dummy1->lookup_tables && dummy1->lookup_tables[ir])
            {
              /* Saturation lookup table */
              SatTable *sat_table = dummy1->lookup_tables[ir];

              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                int ips = SubvectorEltIndex(ps_sub, i, j, k);
                int ipp = SubvectorEltIndex(pp_sub, i, j, k);
                int ipd = SubvectorEltIndex(pd_sub, i, j, k);

                if (ppdat[ipp] >= 0.0)
                  psdat[ips] = (fcn == CALCFCN) ? sat_table->s_sat : 0.0;
                else
                {
                  double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                  if (sat_table->interpolation_method == 1)
                    psdat[ips] = SatLookupLinear(head, sat_table, fcn);
                  else
                    psdat[ips] = SatLookupSpline(head, sat_table, fcn);
                }
              });
            }
            else
            {
              /* Direct evaluation (with Ippisch modification if h_s > 0) */
              double h_s_val = h_s_values ? h_s_values[ir] : 0.0;
              double Scs = 1.0;
              if (h_s_val > 0.0)
              {
                double alpha_r = alphas[ir];
                double n_r = ns[ir];
                double m_r = 1.0e0 - (1.0e0 / n_r);
                Scs = pow(1.0 + pow(alpha_r * h_s_val, n_r), -m_r);
              }

              if (fcn == CALCFCN)
              {
                GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
                {
                  int ips = SubvectorEltIndex(ps_sub, i, j, k);
                  int ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  int ipd = SubvectorEltIndex(pd_sub, i, j, k);

                  double alpha = alphas[ir];
                  double n = ns[ir];
                  double m = 1.0e0 - (1.0e0 / n);
                  double s_res = s_ress[ir];
                  double s_dif = s_difs[ir];

                  if (ppdat[ipp] >= 0.0)
                    psdat[ips] = s_dif + s_res;
                  else
                  {
                    double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                    if (head <= h_s_val)
                      psdat[ips] = s_dif + s_res;
                    else
                    {
                      double Sc = pow(1.0 + pow(alpha * head, n), -m);
                      psdat[ips] = s_dif * (Sc / Scs) + s_res;
                    }
                  }
                });
              }    /* End if clause */
              else /* fcn = CALCDER */
              {
                GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
                {
                  int ips = SubvectorEltIndex(ps_sub, i, j, k);
                  int ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  int ipd = SubvectorEltIndex(pd_sub, i, j, k);

                  double alpha = alphas[ir];
                  double n = ns[ir];
                  double m = 1.0e0 - (1.0e0 / n);
                  double s_dif = s_difs[ir];

                  if (ppdat[ipp] >= 0.0)
                    psdat[ips] = 0.0;
                  else
                  {
                    double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                    if (head <= h_s_val)
                      psdat[ips] = 0.0;
                    else
                      psdat[ips] = (m * n * alpha * pow(alpha * head, (n - 1))) * s_dif
                                   / (Scs * pow(1.0 + pow(alpha * head, n), m + 1));
                  }
                });
              }   /* End else clause */
            }     /* End table/direct-eval branch */
          }     /* End subgrid loop */
        }       /* End loop over regions */
      }         /* End if data not from file */
      else
      {
        gr_solid = ProblemDataGrDomain(problem_data);
        n_values = dummy1->n_values;
        alpha_values = dummy1->alpha_values;
        s_res_values = dummy1->s_res_values;
        s_sat_values = dummy1->s_sat_values;

        ForSubgridI(sg, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, sg);
          ps_sub = VectorSubvector(phase_saturation, sg);
          pp_sub = VectorSubvector(phase_pressure, sg);
          pd_sub = VectorSubvector(phase_density, sg);

          n_values_sub = VectorSubvector(n_values, sg);
          alpha_values_sub = VectorSubvector(alpha_values, sg);
          s_res_values_sub = VectorSubvector(s_res_values, sg);
          s_sat_values_sub = VectorSubvector(s_sat_values, sg);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          psdat = SubvectorData(ps_sub);
          ppdat = SubvectorData(pp_sub);
          pddat = SubvectorData(pd_sub);

          n_values_dat = SubvectorData(n_values_sub);
          alpha_values_dat = SubvectorData(alpha_values_sub);
          s_res_values_dat = SubvectorData(s_res_values_sub);
          s_sat_values_dat = SubvectorData(s_sat_values_sub);

          if (fcn == CALCFCN)
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              int ipp = SubvectorEltIndex(pp_sub, i, j, k);
              int ipd = SubvectorEltIndex(pd_sub, i, j, k);

              int n_index = SubvectorEltIndex(n_values_sub, i, j, k);
              int alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);
              int s_res_index = SubvectorEltIndex(s_res_values_sub, i, j, k);
              int s_sat_index = SubvectorEltIndex(s_sat_values_sub, i, j, k);

              double alpha = alpha_values_dat[alpha_index];
              double n = n_values_dat[n_index];
              double m = 1.0e0 - (1.0e0 / n);
              double s_res = s_res_values_dat[s_res_index];
              double s_sat = s_sat_values_dat[s_sat_index];

              if (ppdat[ipp] >= 0.0)
                psdat[ips] = s_sat;
              else
              {
                double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                psdat[ips] = (s_sat - s_res) /
                             pow(1.0 + pow((alpha * head), n), m)
                             + s_res;
              }
            });
          }      /* End if clause */
          else   /* fcn = CALCDER */
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              int ipp = SubvectorEltIndex(pp_sub, i, j, k);
              int ipd = SubvectorEltIndex(pd_sub, i, j, k);

              int n_index = SubvectorEltIndex(n_values_sub, i, j, k);
              int alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);
              int s_res_index = SubvectorEltIndex(s_res_values_sub, i, j, k);
              int s_sat_index = SubvectorEltIndex(s_sat_values_sub, i, j, k);

              double alpha = alpha_values_dat[alpha_index];
              double n = n_values_dat[n_index];
              double m = 1.0e0 - (1.0e0 / n);
              double s_res = s_res_values_dat[s_res_index];
              double s_sat = s_sat_values_dat[s_sat_index];
              double s_dif = s_sat - s_res;

              if (ppdat[ipp] >= 0.0)
                psdat[ips] = 0.0;
              else
              {
                double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                psdat[ips] = (m * n * alpha * pow(alpha * head, (n - 1))) * s_dif
                             / (pow(1.0 + pow(alpha * head, n), m + 1));
              }
            });
          }     /* End else clause */
        }       /* End subgrid loop */
      }         /* End if data_from_file */
      break;
    }        /* End case 1 */

    case 2: /* Haverkamp et.al. saturation curve */
    {
      double *alphas, *betas, *s_ress, *s_difs;

      dummy2 = (Type2*)(public_xtra->data);

      num_regions = (dummy2->num_regions);
      region_indices = (dummy2->region_indices);
      alphas = (dummy2->alphas);
      betas = (dummy2->betas);
      s_ress = (dummy2->s_ress);
      s_difs = (dummy2->s_difs);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

        ForSubgridI(sg, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, sg);
          ps_sub = VectorSubvector(phase_saturation, sg);
          pp_sub = VectorSubvector(phase_pressure, sg);
          pd_sub = VectorSubvector(phase_density, sg);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          psdat = SubvectorData(ps_sub);
          ppdat = SubvectorData(pp_sub);
          pddat = SubvectorData(pd_sub);

          if (fcn == CALCFCN)
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              int ipp = SubvectorEltIndex(pp_sub, i, j, k);
              int ipd = SubvectorEltIndex(pd_sub, i, j, k);

              double alpha = alphas[ir];
              double beta = betas[ir];
              double s_res = s_ress[ir];
              double s_dif = s_difs[ir];

              if (ppdat[ipp] >= 0.0)
                psdat[ips] = s_dif + s_res;
              else
              {
                double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                psdat[ips] = alpha * s_dif / (alpha + pow(head, beta))
                             + s_res;
              }
            });
          }      /* End if clause */
          else   /* fcn = CALCDER */
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              int ipp = SubvectorEltIndex(pp_sub, i, j, k);
              int ipd = SubvectorEltIndex(pd_sub, i, j, k);

              double alpha = alphas[ir];
              double beta = betas[ir];
              double s_dif = s_difs[ir];

              if (ppdat[ipp] >= 0.0)
                psdat[ips] = 0.0;
              else
              {
                double head = fabs(ppdat[ipp]) / (pddat[ipd] * gravity);
                psdat[ips] = alpha * s_dif * beta * pow(head, beta - 1)
                             / pow((alpha + pow(head, beta)), 2);
              }
            });
          }     /* End else clause */
        }       /* End subgrid loop */
      }         /* End loop over regions */
      break;
    }        /* End case 2 */

    case 3: /* Data points for saturation curve */
    {
      dummy3 = (Type3*)(public_xtra->data);

      num_regions = (dummy3->num_regions);
      region_indices = (dummy3->region_indices);

      if (!amps_Rank(amps_CommWorld))
        printf("Data curves for sats not yet supported.\n");

      break;
    }        /* End case 3 */

    case 4: /* Polynomial function of pressure saturation curve */
    {
      int     *degrees;
      double **coefficients, *region_coeffs;

      dummy4 = (Type4*)(public_xtra->data);

      num_regions = (dummy4->num_regions);
      region_indices = (dummy4->region_indices);
      degrees = (dummy4->degrees);
      coefficients = (dummy4->coefficients);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
        region_coeffs = coefficients[ir];

        ForSubgridI(sg, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, sg);

          ps_sub = VectorSubvector(phase_saturation, sg);
          pp_sub = VectorSubvector(phase_pressure, sg);


          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          psdat = SubvectorData(ps_sub);
          ppdat = SubvectorData(pp_sub);

          if (fcn == CALCFCN)
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              int ipp = SubvectorEltIndex(pp_sub, i, j, k);

              if (ppdat[ipp] == 0.0)
                psdat[ips] = region_coeffs[0];
              else
              {
                psdat[ips] = 0.0;
                for (int dg = 0; dg < degrees[ir] + 1; dg++)
                {
                  psdat[ips] += region_coeffs[dg] * pow(ppdat[ipp], dg);
                }
              }
            });
          }      /* End if clause */
          else   /* fcn = CALCDER */
          {
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              int ipp = SubvectorEltIndex(pp_sub, i, j, k);

              if (ppdat[ipp] == 0.0)
                psdat[ips] = 0.0;
              else
              {
                psdat[ips] = 0.0;
                for (int dg = 0; dg < degrees[ir] + 1; dg++)
                {
                  psdat[ips] += region_coeffs[dg] * dg
                                * pow(ppdat[ipp], (dg - 1));
                }
              }
            });
          }     /* End else clause */
        }       /* End subgrid loop */
      }         /* End loop over regions */
      break;
    }        /* End case 4 */

    case 5: /* ParFlow binary file with spatially varying saturation values */
    {
      Vector *satRF;

      dummy5 = (Type5*)(public_xtra->data);

      satRF = dummy5->satRF;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);
        ps_sub = VectorSubvector(phase_saturation, sg);
        satRF_sub = VectorSubvector(satRF, sg);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        /* RDF: assume resolution is the same in all 3 directions */
        r = SubgridRX(subgrid);

        psdat = SubvectorData(ps_sub);
        satRFdat = SubvectorData(satRF_sub);

        if (fcn == CALCFCN)
        {
          GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
          {
            int ips = SubvectorEltIndex(ps_sub, i, j, k);
            int ipRF = SubvectorEltIndex(satRF_sub, i, j, k);

            psdat[ips] = satRFdat[ipRF];
          });
        }     /* End if clause */
        else  /* fcn = CALCDER */
        {
          GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
          {
            int ips = SubvectorEltIndex(ps_sub, i, j, k);

            psdat[ips] = 0.0;
          });
        }    /* End else clause */
      }      /* End subgrid loop */
      break;
    }        /* End case 5 */
  }          /* End switch */
}

/*--------------------------------------------------------------------------
 * SaturationInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SaturationInitInstanceXtra(
                                      Grid *  grid,
                                      double *temp_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  Type1         *dummy1;
  Type5         *dummy5;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      if (public_xtra->type == 1)
      {
        dummy1 = (Type1*)(public_xtra->data);
        if ((dummy1->data_from_file) == 1)
        {
          FreeVector(dummy1->n_values);
          FreeVector(dummy1->alpha_values);
          FreeVector(dummy1->s_res_values);
          FreeVector(dummy1->s_sat_values);

          dummy1->n_values = NULL;
          dummy1->alpha_values = NULL;
          dummy1->s_res_values = NULL;
          dummy1->s_sat_values = NULL;
        }
      }
      if (public_xtra->type == 5)
      {
        dummy5 = (Type5*)(public_xtra->data);
        FreeVector(dummy5->satRF);
      }
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    /* Uses a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);
      if ((dummy1->data_from_file) == 1)
      {
        dummy1->n_values = NewVectorType(grid, 1, 1, vector_cell_centered);
        dummy1->alpha_values = NewVectorType(grid, 1, 1, vector_cell_centered);
        dummy1->s_res_values = NewVectorType(grid, 1, 1, vector_cell_centered);
        dummy1->s_sat_values = NewVectorType(grid, 1, 1, vector_cell_centered);
      }
    }
    if (public_xtra->type == 5)
    {
      dummy5 = (Type5*)(public_xtra->data);
      (dummy5->satRF) = NewVectorType(grid, 1, 1, vector_cell_centered);
    }
  }


  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;

    /* Uses a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);
      if ((dummy1->data_from_file) == 1)
      {
        ReadPFBinary((dummy1->alpha_file),
                     (dummy1->alpha_values));
        ReadPFBinary((dummy1->n_file),
                     (dummy1->n_values));
        ReadPFBinary((dummy1->s_res_file),
                     (dummy1->s_res_values));
        ReadPFBinary((dummy1->s_sat_file),
                     (dummy1->s_sat_values));
      }
    }
    if (public_xtra->type == 5)
    {
      dummy5 = (Type5*)(public_xtra->data);

      ReadPFBinary((dummy5->filename),
                   (dummy5->satRF));
    }
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * SaturationFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  SaturationFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (instance_xtra)
  {
    if (public_xtra->type == 1)
    {
      Type1* dummy1 = (Type1*)(public_xtra->data);
      if ((dummy1->data_from_file) == 1)
      {
        /* Data will be shared by all instances */
        if (dummy1->n_values)
        {
          FreeVector(dummy1->n_values);
          FreeVector(dummy1->alpha_values);
          FreeVector(dummy1->s_res_values);
          FreeVector(dummy1->s_sat_values);

          dummy1->n_values = NULL;
          dummy1->alpha_values = NULL;
          dummy1->s_res_values = NULL;
          dummy1->s_sat_values = NULL;
        }
      }
    }
    if (public_xtra->type == 5)
    {
      Type5* dummy5 = (Type5*)(public_xtra->data);
      FreeVector(dummy5->satRF);
    }

    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SaturationNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SaturationNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;
  Type2         *dummy2;
  Type3         *dummy3;
  Type4         *dummy4;
  Type5         *dummy5;

  int num_regions, ir, ic;

  char *switch_name;
  char *region;

  char key[IDB_MAX_KEY_LEN];

  NameArray type_na;

  type_na = NA_NewNameArray("Constant VanGenuchten Haverkamp Data Polynomial PFBFile");

  public_xtra = ctalloc(PublicXtra, 1);

  switch_name = GetString("Phase.Saturation.Type");
  public_xtra->type = NA_NameToIndexExitOnError(type_na, switch_name, "Phase.Saturation.Type");


  switch_name = GetString("Phase.Saturation.GeomNames");
  public_xtra->regions = NA_NewNameArray(switch_name);

  num_regions = NA_Sizeof(public_xtra->regions);

  switch ((public_xtra->type))
  {
    case 0:
    {
      dummy0 = ctalloc(Type0, 1);

      dummy0->num_regions = num_regions;

      dummy0->region_indices = ctalloc(int, num_regions);
      dummy0->values = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy0->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        if (dummy0->region_indices[ir] < 0)
        {
          InputError("Error: invalid geometry name <%s> for key <%s>\n",
                     region, "Phase.Saturation.GeomNames");
        }

        sprintf(key, "Geom.%s.Saturation.Value", region);
        dummy0->values[ir] = GetDouble(key);
      }

      (public_xtra->data) = (void*)dummy0;

      break;
    }

    case 1:
    {
      double s_sat;

      dummy1 = ctalloc(Type1, 1);

      sprintf(key, "Phase.Saturation.VanGenuchten.File");
      dummy1->data_from_file = GetIntDefault(key, 0);

      if ((dummy1->data_from_file) == 0)
      {
        dummy1->num_regions = num_regions;

        (dummy1->region_indices) = ctalloc(int, num_regions);
        (dummy1->alphas) = ctalloc(double, num_regions);
        (dummy1->ns) = ctalloc(double, num_regions);
        (dummy1->s_ress) = ctalloc(double, num_regions);
        (dummy1->s_difs) = ctalloc(double, num_regions);
        (dummy1->h_s_values) = ctalloc(double, num_regions);
        (dummy1->lookup_tables) = ctalloc(SatTable*, num_regions);

        /* Read Ippisch air-entry mode */
        int air_entry_mode = 0;  /* 0=None, 1=Constant, 2=InverseAlpha, 3=PerRegion */
        double global_h_s = 0.0;
        {
          NameArray air_entry_na = NA_NewNameArray("None Constant InverseAlpha PerRegion");
          char *air_entry_name = GetStringDefault("Phase.Saturation.VanGenuchten.AirEntryMode", "None");
          air_entry_mode = NA_NameToIndexExitOnError(air_entry_na, air_entry_name,
                                                     "Phase.Saturation.VanGenuchten.AirEntryMode");
          NA_FreeNameArray(air_entry_na);

          if (air_entry_mode == 1)  /* Constant */
          {
            global_h_s = GetDouble("Phase.Saturation.VanGenuchten.AirEntryHead");
          }
        }

        for (ir = 0; ir < num_regions; ir++)
        {
          region = NA_IndexToName(public_xtra->regions, ir);

          dummy1->region_indices[ir] =
            NA_NameToIndex(GlobalsGeomNames, region);

          if (dummy1->region_indices[ir] < 0)
          {
            InputError("Error: invalid geometry name <%s> for key <%s>\n",
                       region, "Phase.Saturation.GeomNames");
          }


          sprintf(key, "Geom.%s.Saturation.Alpha", region);
          dummy1->alphas[ir] = GetDouble(key);

          sprintf(key, "Geom.%s.Saturation.N", region);
          dummy1->ns[ir] = GetDouble(key);

          sprintf(key, "Geom.%s.Saturation.SRes", region);
          dummy1->s_ress[ir] = GetDouble(key);

          sprintf(key, "Geom.%s.Saturation.SSat", region);
          s_sat = GetDouble(key);

          (dummy1->s_difs[ir]) = s_sat - (dummy1->s_ress[ir]);

          /* Compute h_s for this region based on AirEntryMode */
          double h_s = 0.0;
          switch (air_entry_mode)
          {
            case 0:  /* None */
              h_s = 0.0;
              break;

            case 1:  /* Constant */
              h_s = global_h_s;
              break;

            case 2:  /* InverseAlpha */
              if (dummy1->alphas[ir] <= 0.0)
              {
                InputError("Error: AirEntryMode InverseAlpha requires alpha > 0 for region <%s>: %s\n",
                           region, "alpha must be positive");
              }
              h_s = 1.0 / dummy1->alphas[ir];
              break;

            case 3:  /* PerRegion */
              sprintf(key, "Geom.%s.Saturation.AirEntryHead", region);
              h_s = GetDouble(key);
              break;
          }
          dummy1->h_s_values[ir] = h_s;

          /* Build saturation lookup table if requested */
          sprintf(key, "Geom.%s.Saturation.NumSamplePoints", region);
          int num_sample_points = GetIntDefault(key, 0);

          if (num_sample_points)
          {
            if (num_sample_points < 2)
            {
              InputError("Error: Saturation.NumSamplePoints must be >= 2 for region <%s>: %s\n",
                         region, "table interpolation requires at least two points");
            }

            sprintf(key, "Geom.%s.Saturation.MinPressureHead", region);
            double min_pressure_head = GetDouble(key);

            if (fabs(min_pressure_head) <= h_s)
            {
              InputError("Error: |Saturation.MinPressureHead| must exceed AirEntryHead h_s for region <%s>: %s\n",
                         region, "saturation table has zero or negative range");
            }

            NameArray interp_na = NA_NewNameArray("Spline Linear");
            sprintf(key, "Geom.%s.Saturation.InterpolationMethod", region);
            char *interp_name = GetStringDefault(key, "Spline");
            int interpolation_method = NA_NameToIndexExitOnError(interp_na, interp_name, key);
            NA_FreeNameArray(interp_na);

            dummy1->lookup_tables[ir] = SatComputeTable(
                                                        interpolation_method,
                                                        num_sample_points,
                                                        min_pressure_head,
                                                        dummy1->alphas[ir],
                                                        dummy1->ns[ir],
                                                        h_s,
                                                        dummy1->s_ress[ir],
                                                        s_sat);
          }
          else
          {
            dummy1->lookup_tables[ir] = NULL;
          }
        }

        dummy1->alpha_file = NULL;
        dummy1->n_file = NULL;
        dummy1->s_res_file = NULL;
        dummy1->s_sat_file = NULL;
        dummy1->alpha_values = NULL;
        dummy1->n_values = NULL;
        dummy1->s_res_values = NULL;
        dummy1->s_sat_values = NULL;
      }
      else
      {
        sprintf(key, "Geom.%s.Saturation.Alpha.Filename", "domain");
        dummy1->alpha_file = GetString(key);
        sprintf(key, "Geom.%s.Saturation.N.Filename", "domain");
        dummy1->n_file = GetString(key);
        sprintf(key, "Geom.%s.Saturation.SRes.Filename", "domain");
        dummy1->s_res_file = GetString(key);
        sprintf(key, "Geom.%s.Saturation.SSat.Filename", "domain");
        dummy1->s_sat_file = GetString(key);

        dummy1->num_regions = 0;
        dummy1->region_indices = NULL;
        dummy1->alphas = NULL;
        dummy1->ns = NULL;
        dummy1->s_ress = NULL;
        dummy1->s_difs = NULL;
        dummy1->h_s_values = NULL;
        dummy1->lookup_tables = NULL;
      }

      (public_xtra->data) = (void*)dummy1;

      break;
    }

    case 2:
    {
      double s_sat;

      dummy2 = ctalloc(Type2, 1);

      dummy2->num_regions = num_regions;

      (dummy2->region_indices) = ctalloc(int, num_regions);
      (dummy2->alphas) = ctalloc(double, num_regions);
      (dummy2->betas) = ctalloc(double, num_regions);
      (dummy2->s_ress) = ctalloc(double, num_regions);
      (dummy2->s_difs) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy2->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);


        sprintf(key, "Geom.%s.Saturation.A", region);
        dummy2->alphas[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.Saturation.gamma", region);
        dummy2->betas[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.Saturation.SRes", region);
        dummy2->s_ress[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.Saturation.SSat", region);
        s_sat = GetDouble(key);

        (dummy2->s_difs[ir]) = s_sat - (dummy2->s_ress[ir]);
      }
      (public_xtra->data) = (void*)dummy2;

      break;
    }

    case 3:
    {
      dummy3 = ctalloc(Type3, 1);

      dummy3->num_regions = num_regions;

      (dummy3->region_indices) = ctalloc(int, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy3->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);
      }
      (public_xtra->data) = (void*)dummy3;

      break;
    }

    case 4:
    {
      int degree;

      dummy4 = ctalloc(Type4, 1);

      dummy4->num_regions = num_regions;

      (dummy4->region_indices) = ctalloc(int, num_regions);
      (dummy4->degrees) = ctalloc(int, num_regions);
      (dummy4->coefficients) = ctalloc(double*, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy4->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        sprintf(key, "Geom.%s.Saturation.Degree", region);
        dummy4->degrees[ir] = GetInt(key);

        degree = (dummy4->degrees[ir]);
        dummy4->coefficients[ir] = ctalloc(double, degree + 1);

        for (ic = 0; ic < degree + 1; ic++)
        {
          sprintf(key, "Geom.%s.Saturation.Coeff.%d", region, ic);
          dummy4->coefficients[ir][ic] = GetDouble(key);
        }
      }
      public_xtra->data = (void*)dummy4;

      break;
    }

    case 5:
    {
      dummy5 = ctalloc(Type5, 1);

      sprintf(key, "Geom.%s.Saturation.FileName", "domain");
      dummy5->filename = GetString(key);

      public_xtra->data = (void*)dummy5;

      break;
    }

    default:
    {
      InputError("Error: invalid type <%s> for key <%s>\n",
                 switch_name, key);
    }
  }      /* End switch */

  NA_FreeNameArray(type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * SaturationFreePublicXtra
 *--------------------------------------------------------------------------*/

void  SaturationFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;
  Type1       *dummy1;
  Type2       *dummy2;
  Type3       *dummy3;
  Type4       *dummy4;
  Type5       *dummy5;

  int num_regions, ir;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->regions);

    switch ((public_xtra->type))
    {
      case 0:
      {
        dummy0 = (Type0*)(public_xtra->data);

        tfree(dummy0->region_indices);
        tfree(dummy0->values);
        tfree(dummy0);

        break;
      }

      case 1:
      {
        dummy1 = (Type1*)(public_xtra->data);

        if (dummy1->data_from_file == 0)
        {
          tfree(dummy1->region_indices);
          tfree(dummy1->alphas);
          tfree(dummy1->ns);
          tfree(dummy1->s_ress);
          tfree(dummy1->s_difs);
          tfree(dummy1->h_s_values);

          if (dummy1->lookup_tables)
          {
            int num_regions = dummy1->num_regions;
            for (int ir = 0; ir < num_regions; ir++)
            {
              if (dummy1->lookup_tables[ir])
              {
                tfree(dummy1->lookup_tables[ir]->x);
                tfree(dummy1->lookup_tables[ir]->a);
                tfree(dummy1->lookup_tables[ir]->d);
                tfree(dummy1->lookup_tables[ir]->a_der);
                tfree(dummy1->lookup_tables[ir]->d_der);
                tfree(dummy1->lookup_tables[ir]->slope);
                tfree(dummy1->lookup_tables[ir]->slope_der);
                tfree(dummy1->lookup_tables[ir]);
              }
            }
            tfree(dummy1->lookup_tables);
          }
        }

        tfree(dummy1);

        break;
      }

      case 2:
      {
        dummy2 = (Type2*)(public_xtra->data);

        tfree(dummy2->region_indices);
        tfree(dummy2->alphas);
        tfree(dummy2->betas);
        tfree(dummy2->s_ress);
        tfree(dummy2->s_difs);
        tfree(dummy2);

        break;
      }

      case 3:
      {
        dummy3 = (Type3*)(public_xtra->data);

        tfree(dummy3->region_indices);
        tfree(dummy3);

        break;
      }

      case 4:
      {
        dummy4 = (Type4*)(public_xtra->data);

        num_regions = (dummy4->num_regions);

        for (ir = 0; ir < num_regions; ir++)
        {
          tfree(dummy4->coefficients[ir]);
        }

        tfree(dummy4->region_indices);
        tfree(dummy4->degrees);
        tfree(dummy4->coefficients);
        tfree(dummy4);

        break;
      }

      case 5:
      {
        dummy5 = (Type5*)(public_xtra->data);

        tfree(dummy5);

        break;
      }
    }    /* End of case statement */

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * SaturationSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SaturationSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type1         *dummy1;

  int sz = 0;

  if (public_xtra->type == 1)
  {
    dummy1 = (Type1*)(public_xtra->data);
    if ((dummy1->data_from_file) == 1)
    {
      /* add local TempData size to `sz' */
      sz += SizeOfVector(dummy1->n_values);
      sz += SizeOfVector(dummy1->alpha_values);
      sz += SizeOfVector(dummy1->s_res_values);
      sz += SizeOfVector(dummy1->s_sat_values);
    }
  }

  return sz;
}

void  SaturationOutput()
{
  //PFModule    *this_module = ThisPFModule;
  //PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  printf("SaturationOutput does nothing\n");
}


void  SaturationOutputStatic(
                             char *       file_prefix,
                             ProblemData *problem_data /* Contains geometry info. for the problem */
                             )
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type1         *dummy1;

  Grid          *grid = VectorGrid(ProblemDataSpecificStorage(problem_data));

  GrGeomSolid   *gr_solid;

  Subvector     *n_values_sub;
  Subvector     *alpha_values_sub;
  Subvector     *s_res_values_sub;
  Subvector     *s_sat_values_sub;

  double        *n_values_dat, *alpha_values_dat;
  double        *s_res_values_dat, *s_sat_values_dat;

  SubgridArray  *subgrids = GridSubgrids(grid);

  Subgrid       *subgrid;

  int sg;

  int ix, iy, iz, r;
  int nx, ny, nz;

  int i, j, k;

  int            *region_indices, num_regions, ir;

  Subvector      *pd_alpha_sub;         //BB
  Subvector      *pd_n_sub;         //BB
  Subvector      *pd_sres_sub;         //BB
  Subvector      *pd_ssat_sub;         //BB
  double *pd_alpha_dat, *pd_n_dat, *pd_sres_dat, *pd_ssat_dat;    //BB

  Vector *pd_alpha = NewVectorType(grid, 1, 1, vector_cell_centered);
  Vector *pd_n = NewVectorType(grid, 1, 1, vector_cell_centered);
  Vector *pd_sres = NewVectorType(grid, 1, 1, vector_cell_centered);
  Vector *pd_ssat = NewVectorType(grid, 1, 1, vector_cell_centered);

  /* Initialize saturations */
  InitVector(pd_alpha, 0.0);
  InitVector(pd_n, 0.0);
  InitVector(pd_sres, 0.0);
  InitVector(pd_ssat, 0.0);

  switch ((public_xtra->type))
  {
    case 1: /* Van Genuchten saturation curve */
    {
      int data_from_file;
      double *alphas, *ns, *s_ress;

      Vector *n_values, *alpha_values, *s_res_values, *s_sat_values;

      dummy1 = (Type1*)(public_xtra->data);

      num_regions = (dummy1->num_regions);
      region_indices = (dummy1->region_indices);
      alphas = (dummy1->alphas);
      ns = (dummy1->ns);
      s_ress = (dummy1->s_ress);
      double* s_difs = (dummy1->s_difs);
      data_from_file = (dummy1->data_from_file);

      if (data_from_file == 0) /* Soil parameters given by region */
      {
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(sg, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, sg);

            pd_alpha_sub = VectorSubvector(pd_alpha, sg);   //BB
            pd_n_sub = VectorSubvector(pd_n, sg);           //BB
            pd_sres_sub = VectorSubvector(pd_sres, sg);     //BB
            pd_ssat_sub = VectorSubvector(pd_ssat, sg);     //BB

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            r = SubgridRX(subgrid);

            pd_alpha_dat = SubvectorData(pd_alpha_sub);   //BB
            pd_n_dat = SubvectorData(pd_n_sub);           //BB
            pd_sres_dat = SubvectorData(pd_sres_sub);     //BB
            pd_ssat_dat = SubvectorData(pd_ssat_sub);     //BB

            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(pd_alpha_sub, i, j, k);

              double alpha = alphas[ir];
              double n = ns[ir];
              double s_res = s_ress[ir];
              double s_dif = s_difs[ir];

              pd_alpha_dat[ips] = alpha;  //BB
              pd_n_dat[ips] = n;  //BB
              pd_sres_dat[ips] = s_res;  //BB  // no ssat???
              // Storing s_dif in the structure, convert back to s_sat for output
              pd_ssat_dat[ips] = s_dif + s_res;
            });
          }     /* End subgrid loop */
        }       /* End loop over regions */
      }         /* End if data not from file */
      else
      {
        gr_solid = ProblemDataGrDomain(problem_data);
        n_values = dummy1->n_values;
        alpha_values = dummy1->alpha_values;
        s_res_values = dummy1->s_res_values;
        s_sat_values = dummy1->s_sat_values;

        ForSubgridI(sg, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, sg);

          n_values_sub = VectorSubvector(n_values, sg);
          alpha_values_sub = VectorSubvector(alpha_values, sg);
          s_res_values_sub = VectorSubvector(s_res_values, sg);
          s_sat_values_sub = VectorSubvector(s_sat_values, sg);

          pd_alpha_sub = VectorSubvector(pd_alpha, sg);   //BB
          pd_n_sub = VectorSubvector(pd_n, sg);           //BB
          pd_sres_sub = VectorSubvector(pd_sres, sg);     //BB
          pd_ssat_sub = VectorSubvector(pd_ssat, sg);     //BB

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          n_values_dat = SubvectorData(n_values_sub);
          alpha_values_dat = SubvectorData(alpha_values_sub);
          s_res_values_dat = SubvectorData(s_res_values_sub);
          s_sat_values_dat = SubvectorData(s_sat_values_sub);

          pd_alpha_dat = SubvectorData(pd_alpha_sub);   //BB
          pd_n_dat = SubvectorData(pd_n_sub);           //BB
          pd_sres_dat = SubvectorData(pd_sres_sub);     //BB
          pd_ssat_dat = SubvectorData(pd_ssat_sub);     //BB

          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            int ips = SubvectorEltIndex(pd_alpha_sub, i, j, k);

            int n_index = SubvectorEltIndex(n_values_sub, i, j, k);
            int alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);
            int s_res_index = SubvectorEltIndex(s_res_values_sub, i, j, k);
            int s_sat_index = SubvectorEltIndex(s_sat_values_sub, i, j, k);

            double alpha = alpha_values_dat[alpha_index];
            double n = n_values_dat[n_index];
            double s_res = s_res_values_dat[s_res_index];
            double s_sat = s_sat_values_dat[s_sat_index];

            pd_alpha_dat[ips] = alpha;  //BB
            pd_n_dat[ips] = n;  //BB
            pd_sres_dat[ips] = s_res;  //BB
            pd_ssat_dat[ips] = s_sat;  //BB
          });
        }       /* End subgrid loop */
      }         /* End if data_from_file */
      break;
    }        /* End case 1 */
  }          /* End switch */

  char file_postfix[2048];

  strcpy(file_postfix, "alpha");
  WritePFBinary(file_prefix, file_postfix,
                pd_alpha);

  strcpy(file_postfix, "n");
  WritePFBinary(file_prefix, file_postfix,
                pd_n);

  strcpy(file_postfix, "sres");
  WritePFBinary(file_prefix, file_postfix,
                pd_sres);

  strcpy(file_postfix, "ssat");
  WritePFBinary(file_prefix, file_postfix,
                pd_ssat);

  FreeVector(pd_alpha);
  FreeVector(pd_n);
  FreeVector(pd_sres);
  FreeVector(pd_ssat);
}

