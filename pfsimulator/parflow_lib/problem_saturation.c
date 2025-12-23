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
      double *alphas, *ns, *s_ress, *s_difs;

      Vector *n_values, *alpha_values, *s_res_values, *s_sat_values;

      dummy1 = (Type1*)(public_xtra->data);

      num_regions = (dummy1->num_regions);
      region_indices = (dummy1->region_indices);
      alphas = (dummy1->alphas);
      ns = (dummy1->ns);
      s_ress = (dummy1->s_ress);
      s_difs = (dummy1->s_difs);
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
                  psdat[ips] = s_dif / pow(1.0 + pow((alpha * head), n), m)
                               + s_res;
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
                  psdat[ips] = (m * n * alpha * pow(alpha * head, (n - 1))) * s_dif
                               / (pow(1.0 + pow(alpha * head, n), m + 1));
                }
              });
            }   /* End else clause */
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

