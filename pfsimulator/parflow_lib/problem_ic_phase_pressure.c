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
#include "parflow_netcdf.h"
#include <string.h>

#include <float.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  NameArray regions;  /* "NameArray" is defined in "input_database.h" and contains info given below
                       *                  typedef struct NameArray__
                       * {
                       *   int num;
                       *   char **names;
                       *   char *tok_string;
                       *   char *string;
                       *
                       * } NameArrayStruct; */

  int type;
  void    *data;
} PublicXtra;

typedef struct {
  PFModule *phase_density;

  Grid     *grid;

  double   *temp_data;
} InstanceXtra;

typedef struct {
  int num_regions;
  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
  int num_regions;
  int     *region_indices;
  double  *reference_elevations;
  double  *pressure_values;
} Type1;                       /* hydrostatic regions with a single
                               * reference depth for each region */

typedef struct {
  int num_regions;
  int     *region_indices;

  /* for the reference patch */
  int     *geom_indices;
  int     *patch_indices;

  double  *pressure_values;
} Type2;                       /* hydrostatic regions with a
                                * reference patch for each region */

typedef struct {
  char    *filename;
  Vector  *ic_values;
} Type3;                      /* Spatially varying field over entire domain
                               * read from a file */
typedef struct {
  char    *filename;
  int timestep;
  Vector  *ic_values;
} Type4;                      /* Spatially varying field over entire domain
                               * read from a file */


/*--------------------------------------------------------------------------
 * ICPhasePressure:
 *    This routine returns a Vector of pressures at the initial time.
 *--------------------------------------------------------------------------*/

void         ICPhasePressure(

                             Vector *     ic_pressure, /* Return values of initial condition */
                             Vector *     mask, /* Mask of active cells needed by the LSM */
                             ProblemData *problem_data, /* Contains geometry information for the problem */
                             Problem *    problem) /* General problem information */

{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  PFModule      *phase_density = (instance_xtra->phase_density);

  Grid          *grid = VectorGrid(ic_pressure);

  GrGeomSolid   *gr_solid, *gr_domain;

  SubgridArray  *subgrids = GridSubgrids(grid);

  Subgrid       *subgrid;

  Vector        *temp_new_density = NULL;
  Vector        *temp_new_density_der = NULL;

  Vector        *temp_fcn = NULL;

  Subvector     *ps_sub;
  Subvector     *tf_sub;
  Subvector     *tnd_sub;
  Subvector     *tndd_sub;
  Subvector     *ic_values_sub;
  Subvector     *m_sub;

  Vector      *rsz = ProblemDataRealSpaceZ(problem_data);
  Subvector   *rsz_sub;
  double      *rsz_dat;

  double        *data;
  double        *fcn_data;
  double        *new_density_data;
  double        *new_density_der_data;
  double        *psdat, *ic_values_dat, *m_dat;

  double gravity = -ProblemGravity(problem);

  int num_regions;
  int           *region_indices;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  int is, i, j, k, ips, iel, ipicv, ival;

  amps_Invoice result_invoice;

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/

  temp_new_density = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  temp_new_density_der = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  temp_fcn = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /*-----------------------------------------------------------------------
   * Initial pressure conditions
   *-----------------------------------------------------------------------*/

  /* SGS Some of this initialization is being done multiple places. Why is that?*/
  // SGS where is this macro coming from?
#undef max
  InitVector(ic_pressure, -FLT_MAX);
  InitVector(temp_new_density, 0.0);
  InitVector(temp_new_density_der, 0.0);
  InitVector(temp_fcn, 0.0);

  switch ((public_xtra->type))
  {
    case 0: /* Assign constant values within regions. */
    {
      double  *values;
      int ir;

      Type0* dummy0 = (Type0*)(public_xtra->data);

      num_regions = (dummy0->num_regions);
      region_indices = (dummy0->region_indices);
      values = (dummy0->values);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          ps_sub = VectorSubvector(ic_pressure, is);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          data = SubvectorData(ps_sub);
          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ips = SubvectorEltIndex(ps_sub, i, j, k);    /*#define SubvectorEltIndex(subvector, x, y, z) \
                                                          *                                                     (((x) - SubvectorIX(subvector)) + \
                                                          *                                                     (((y) - SubvectorIY(subvector)) + \
                                                          *                                                     (((z) - SubvectorIZ(subvector))) * \
                                                          *                                                             SubvectorNY(subvector)) * \
                                                          *                                                             SubvectorNX(subvector))*/

            data[ips] = values[ir];
          });
        }      /* End of subgrid loop */
      }        /* End of region loop */
      break;
    }          /* End of case 0 */

    case 1: /* Hydrostatic regions relative to a single reference depth. */
    {
      /* Calculate hydrostatic conditions within region for
       * elevations different from reference elevation.
       * Hydrostatic condition is:
       *       grad p - rho g grad z = 0 */

      int max_its = 10;
      int iterations;

      double  *reference_elevations;
      double  *pressure_values;
      double  *ref_den;
      double dtmp;
      double nonlin_resid;
      int ir;

      Type1* dummy1 = (Type1*)(public_xtra->data);

      num_regions = (dummy1->num_regions);
      region_indices = (dummy1->region_indices);
      reference_elevations = (dummy1->reference_elevations);
      pressure_values = (dummy1->pressure_values);

      ref_den = ctalloc(double, num_regions);

      result_invoice = amps_NewInvoice("%d", &nonlin_resid);

      /* Solve a nonlinear problem for hydrostatic pressure
       * at points in region given pressure at reference elevation.
       * Note that the problem is only nonlinear if
       * density depends on pressure.
       *
       * The nonlinear problem to solve is:
       * F(p) = 0
       * F(p) = P - P_ref - 0.5*(rho(P) + rho(P_ref))*gravity*(z - z_ref)
       *
       * Newton's method is used to find a solution. */

      nonlin_resid = 1.0;
      iterations = -1;
      while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
      {
        /* Get derivative of density at new pressures. */
        if (iterations > -1)
        {
          PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, ic_pressure,
                                                                 temp_new_density_der, &dtmp,
                                                                 &dtmp, CALCDER));
        }

        /* Get mask values. */
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);
            m_sub = VectorSubvector(mask, is);
            m_dat = SubvectorData(m_sub);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            /* RDF: assume resolution is the same in all 3 directions */
            r = SubgridRX(subgrid);

            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(m_sub, i, j, k);
              m_dat[ips] = region_indices[ir] + 1;
            });
          }       /* End of subgrid loop */
        }         /* End of region loop */

        /* Get new pressure values. */
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);
            ps_sub = VectorSubvector(ic_pressure, is);
            data = SubvectorData(ps_sub);
            tf_sub = VectorSubvector(temp_fcn, is);
            fcn_data = SubvectorData(tf_sub);
            tndd_sub = VectorSubvector(temp_new_density_der, is);
            new_density_der_data = SubvectorData(tndd_sub);

            rsz_sub = VectorSubvector(rsz, is);
            rsz_dat = SubvectorData(rsz_sub);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            /* RDF: assume resolution is the same in all 3 directions */
            r = SubgridRX(subgrid);

            if (iterations > -1)       /* Update pressure values */
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(ps_sub, i, j, k);
                ival = SubvectorEltIndex(rsz_sub, i, j, k);
                dtmp = 1.0 - 0.5 * new_density_der_data[ips] * gravity
                       * (rsz_dat[ival] - reference_elevations[ir]);
                data[ips] = data[ips] - fcn_data[ips] / dtmp;
              });
            }
            else    /* Initialize stuff */
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(ps_sub, i, j, k);
                data[ips] = pressure_values[ir];
              });
            }
          }       /* End of subgrid loop */
        }         /* End of region loop */

        /* Get density values at new pressure values. */
        PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, ic_pressure,
                                                               temp_new_density, &dtmp,
                                                               &dtmp, CALCFCN));

        /* Calculate nonlinear residual and value of the nonlinear function
         * at current pressures. */
        nonlin_resid = 0.0;
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);
            tf_sub = VectorSubvector(temp_fcn, is);
            fcn_data = SubvectorData(tf_sub);
            tnd_sub = VectorSubvector(temp_new_density, is);
            new_density_data = SubvectorData(tnd_sub);
            ps_sub = VectorSubvector(ic_pressure, is);
            data = SubvectorData(ps_sub);

            rsz_sub = VectorSubvector(rsz, is);
            rsz_dat = SubvectorData(rsz_sub);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            /* RDF: assume resolution is the same in all 3 directions */
            r = SubgridRX(subgrid);

            if (iterations > -1)       /* Determine new nonlinear residual */
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(tf_sub, i, j, k);
                ival = SubvectorEltIndex(rsz_sub, i, j, k);
                fcn_data[ips] = data[ips] - pressure_values[ir]
                                - 0.5 * (new_density_data[ips] + ref_den[ir])
                                * gravity * (rsz_dat[ival] - reference_elevations[ir]);
                nonlin_resid += fcn_data[ips] * fcn_data[ips];
              });
            }
            else     /* Determine initial residual */
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(tf_sub, i, j, k);
                ival = SubvectorEltIndex(rsz_sub, i, j, k);
                ref_den[ir] = new_density_data[ips];
                fcn_data[ips] = -ref_den[ir] * gravity
                                * (rsz_dat[ival] - reference_elevations[ir]);
                nonlin_resid += fcn_data[ips] * fcn_data[ips];
              });
            }
          }       /* End of subgrid loop */
        }         /* End of region loop */

        amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
        nonlin_resid = sqrt(nonlin_resid);

        iterations++;
      }        /* End of while loop */

      amps_FreeInvoice(result_invoice);
      tfree(ref_den);

      break;
    }          /* End of case 1 */

    case 2: /* Hydrostatic regions with a reference surface for each region */
    {
      /* Calculate hydrostatic conditions within region for
       * elevations at reference surface.
       * Hydrostatic condition is:
       *       grad p - rho g grad z = 0 */

      GeomSolid *ref_solid;

      int max_its = 10;
      int iterations;

      int      *patch_indices;
      int      *geom_indices;
      double   *pressure_values;
      double   *ref_den;
      double dtmp;
      double nonlin_resid;
      double  ***elevations;
      int ir;

      Type2* dummy2 = (Type2*)(public_xtra->data);

      num_regions = (dummy2->num_regions);
      region_indices = (dummy2->region_indices);
      geom_indices = (dummy2->geom_indices);
      patch_indices = (dummy2->patch_indices);
      pressure_values = (dummy2->pressure_values);

      ref_den = ctalloc(double, num_regions);
      elevations = ctalloc(double**, num_regions);

      result_invoice = amps_NewInvoice("%d", &nonlin_resid);

      /* Get mask values. */
      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          m_sub = VectorSubvector(mask, is);
          m_dat = SubvectorData(m_sub);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          /* RDF: assume resolution is the same in all 3 directions */
          r = SubgridRX(subgrid);

          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
          {
            ips = SubvectorEltIndex(m_sub, i, j, k);
            m_dat[ips] = region_indices[ir] + 1;
          });
        }      /* End of subgrid loop */
      }        /* End of region loop */


      /* Calculate array of elevations on reference surface. */
      for (ir = 0; ir < num_regions; ir++)
      {
        ref_solid = ProblemDataSolid(problem_data, geom_indices[ir]);
        elevations[ir] = CalcElevations(ref_solid, patch_indices[ir],
                                        subgrids, problem_data);
      }        /* End of region loop */

      /* Solve a nonlinear problem for hydrostatic pressure
       * at points in region given pressure at reference elevation.
       * Note that the problem is only nonlinear if
       * density depends on pressure.
       *
       * The nonlinear problem to solve is:
       * F(p) = 0
       * F(p) = P - P_ref - 0.5*(rho(P) + rho(P_ref))*gravity*(z - z_ref)
       *
       * Newton's method is used to find a solution. */

      nonlin_resid = 1.0;
      iterations = -1;
      while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
      {
        if (iterations > -1)
        {
          /* Get derivative of density at new pressures. */
          PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, ic_pressure,
                                                                 temp_new_density_der, &dtmp,
                                                                 &dtmp, CALCDER));
        }

        /* Get new pressure values. */
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);
            ps_sub = VectorSubvector(ic_pressure, is);
            data = SubvectorData(ps_sub);
            tf_sub = VectorSubvector(temp_fcn, is);
            fcn_data = SubvectorData(tf_sub);
            tndd_sub = VectorSubvector(temp_new_density_der, is);
            new_density_der_data = SubvectorData(tndd_sub);

            rsz_sub = VectorSubvector(rsz, is);
            rsz_dat = SubvectorData(rsz_sub);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            /* RDF: assume resolution is the same in all 3 directions */
            r = SubgridRX(subgrid);

            if (iterations > -1)       /* Update pressure values */
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(ps_sub, i, j, k);
                iel = (i - ix) + (j - iy) * nx;
                ival = SubvectorEltIndex(rsz_sub, i, j, k);
                dtmp = 1.0 - 0.5 * new_density_der_data[ips] * gravity
                       * (rsz_dat[ival] - elevations[ir][is][iel]);
                data[ips] = data[ips] - fcn_data[ips] / dtmp;
              });
            }
            else    /* Initialize stuff */
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(ps_sub, i, j, k);
                data[ips] = pressure_values[ir];
              });
            }
          }       /* End of subgrid loop */
        }         /* End of region loop */

        /* Get density values at new pressure values. */
        PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, ic_pressure,
                                                               temp_new_density, &dtmp,
                                                               &dtmp, CALCFCN));
        /* Calculate nonlinear residual and value of the nonlinear function
         * at current pressures. */
        nonlin_resid = 0.0;
        for (ir = 0; ir < num_regions; ir++)
        {
          gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);
            tf_sub = VectorSubvector(temp_fcn, is);
            fcn_data = SubvectorData(tf_sub);
            tnd_sub = VectorSubvector(temp_new_density, is);
            new_density_data = SubvectorData(tnd_sub);
            ps_sub = VectorSubvector(ic_pressure, is);
            data = SubvectorData(ps_sub);

            rsz_sub = VectorSubvector(rsz, is);
            rsz_dat = SubvectorData(rsz_sub);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            /* RDF: assume resolution is the same in all 3 directions */
            r = SubgridRX(subgrid);

            if (iterations > -1)
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(tf_sub, i, j, k);
                iel = (i - ix) + (j - iy) * nx;
                ival = SubvectorEltIndex(rsz_sub, i, j, k);
                fcn_data[ips] = data[ips] - pressure_values[ir]
                                - 0.5 * (new_density_data[ips] + ref_den[ir])
                                * gravity * (rsz_dat[ival] - elevations[ir][is][iel]);
                nonlin_resid += fcn_data[ips] * fcn_data[ips];
              });
            }
            else
            {
              GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
              {
                ips = SubvectorEltIndex(tf_sub, i, j, k);
                iel = (i - ix) + (j - iy) * nx;
                ival = SubvectorEltIndex(rsz_sub, i, j, k);
                ref_den[ir] = new_density_data[ips];
                fcn_data[ips] = -ref_den[ir] * gravity
                                * (rsz_dat[ival] - elevations[ir][is][iel]);
                nonlin_resid += fcn_data[ips] * fcn_data[ips];
              });
            }
          }       /* End of subgrid loop */
        }         /* End of region loop */

        amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
        nonlin_resid = sqrt(nonlin_resid);

        iterations++;
      }        /* End of while loop */

      amps_FreeInvoice(result_invoice);
      tfree(ref_den);

      for (ir = 0; ir < num_regions; ir++)
      {
        ForSubgridI(is, subgrids)
        {
          tfree(elevations[ir][is]);
        }
        tfree(elevations[ir]);
      }
      tfree(elevations);

      break;
    }          /* End of case 2 */

    case 3: /* ParFlow binary file with spatially varying pressure values */
    {
      Type3* dummy3 = (Type3*)(public_xtra->data);

      Vector *ic_values = dummy3->ic_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        m_sub = VectorSubvector(mask, is);
        m_dat = SubvectorData(m_sub);

        ps_sub = VectorSubvector(ic_pressure, is);
        ic_values_sub = VectorSubvector(ic_values, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        /* RDF: assume resolution is the same in all 3 directions */
        r = SubgridRX(subgrid);

        psdat = SubvectorData(ps_sub);
        ic_values_dat = SubvectorData(ic_values_sub);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(ps_sub, i, j, k);
          ipicv = SubvectorEltIndex(ic_values_sub, i, j, k);

          psdat[ips] = ic_values_dat[ipicv];
          // SGS fixthis
          m_dat[ips] = 99999;
          // m_dat[ips] = 1.0;
        });
      }        /* End subgrid loop */

      break;
    }          /* End case 3 */

    case 4: /* ParFlow NetCDF file with spatially varying pressure values */
    {
      Type4         *dummy4 = (Type4*)(public_xtra->data);

      Vector *ic_values = dummy4->ic_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        m_sub = VectorSubvector(mask, is);
        m_dat = SubvectorData(m_sub);

        ps_sub = VectorSubvector(ic_pressure, is);
        ic_values_sub = VectorSubvector(ic_values, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        /* RDF: assume resolution is the same in all 3 directions */
        r = SubgridRX(subgrid);

        psdat = SubvectorData(ps_sub);
        ic_values_dat = SubvectorData(ic_values_sub);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          ips = SubvectorEltIndex(ps_sub, i, j, k);
          ipicv = SubvectorEltIndex(ic_values_sub, i, j, k);

          psdat[ips] = ic_values_dat[ipicv];
          // SGS fixthis
          m_dat[ips] = 99999;
          // m_dat[ips] = 1.0;
        });
      }        /* End subgrid loop */

      break;
    }          /* End case 4 */
  }            /* End of switch statement */


  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(temp_new_density);
  FreeVector(temp_new_density_der);
  FreeVector(temp_fcn);
}


/*--------------------------------------------------------------------------
 * ICPhasePressureInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *ICPhasePressureInitInstanceXtra(
                                           Problem *problem,
                                           Grid *   grid,
                                           double * temp_data)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
    }

    /* set new data */
    (instance_xtra->grid) = grid;


    /* Uses a spatially varying field */
    if (public_xtra->type == 3)
    {
      Type3* dummy3 = (Type3*)(public_xtra->data);

      /* Allocate temp vector */
      dummy3->ic_values = NewVectorType(grid, 1, 1, vector_cell_centered);
      ReadPFBinary((dummy3->filename), (dummy3->ic_values));
    }
    else if (public_xtra->type == 4)
    {
      Type4* dummy4 = (Type4*)(public_xtra->data);

      /* Allocate temp vector */
      dummy4->ic_values = NewVectorType(grid, 1, 1, vector_cell_centered);
      ReadPFNC(dummy4->filename, dummy4->ic_values, "pressure", dummy4->timestep, 3);
    }
  }

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->phase_density) =
      PFModuleNewInstanceType(NewDefault, ProblemPhaseDensity(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->phase_density), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * ICPhasePressureFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  ICPhasePressureFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    /* Uses a spatially varying field */
    if (public_xtra->type == 3)
    {
      Type3* dummy3 = (Type3*)(public_xtra->data);

      FreeVector(dummy3->ic_values);
    }
    else if (public_xtra->type == 4)
    {
      Type4* dummy4 = (Type4*)(public_xtra->data);
      FreeVector(dummy4->ic_values);
    }

    PFModuleFreeInstance(instance_xtra->phase_density);

    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * ICPhasePressureNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ICPhasePressureNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  int num_regions;
  int ir;

  char *switch_name;
  char *region;

  char key[IDB_MAX_KEY_LEN];
  char ncKey[IDB_MAX_KEY_LEN];

  NameArray type_na;

  type_na = NA_NewNameArray(
                            "Constant HydroStaticDepth HydroStaticPatch PFBFile NCFile");

  public_xtra = ctalloc(PublicXtra, 1);

  switch_name = GetString("ICPressure.Type");

  public_xtra->type = NA_NameToIndexExitOnError(type_na, switch_name, "ICPressure.Type");

  switch_name = GetString("ICPressure.GeomNames");
  public_xtra->regions = NA_NewNameArray(switch_name);

  num_regions = NA_Sizeof(public_xtra->regions);

  switch ((public_xtra->type))
  {
    case 0:
    {
      Type0* dummy0 = ctalloc(Type0, 1);

      dummy0->num_regions = num_regions;

      (dummy0->region_indices) = ctalloc(int, num_regions);
      (dummy0->values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy0->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        sprintf(key, "Geom.%s.ICPressure.Value", region);
        dummy0->values[ir] = GetDouble(key);
      }

      (public_xtra->data) = (void*)dummy0;
      break;
    }

    case 1:
    {
      Type1* dummy1 = ctalloc(Type1, 1);

      dummy1->num_regions = num_regions;

      (dummy1->region_indices) = ctalloc(int, num_regions);
      (dummy1->reference_elevations) = ctalloc(double, num_regions);
      (dummy1->pressure_values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy1->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);

        sprintf(key, "Geom.%s.ICPressure.RefElevation", region);
        dummy1->reference_elevations[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.ICPressure.Value", region);
        dummy1->pressure_values[ir] = GetDouble(key);
      }

      (public_xtra->data) = (void*)dummy1;

      break;
    }

    case 2:
    {
      Type2* dummy2 = ctalloc(Type2, 1);

      dummy2->num_regions = num_regions;

      (dummy2->region_indices) = ctalloc(int, num_regions);
      (dummy2->geom_indices) = ctalloc(int, num_regions);
      (dummy2->patch_indices) = ctalloc(int, num_regions);
      (dummy2->pressure_values) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++)
      {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy2->region_indices[ir] =
          NA_NameToIndex(GlobalsGeomNames, region);
        sprintf(key, "Geom.%s.ICPressure.Value", region);
        dummy2->pressure_values[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.ICPressure.RefGeom", region);
        switch_name = GetString(key);
        dummy2->geom_indices[ir] = NA_NameToIndexExitOnError(GlobalsGeomNames,
                                                             switch_name, key);

        sprintf(key, "Geom.%s.ICPressure.RefPatch", region);
        switch_name = GetString(key);
        dummy2->patch_indices[ir] =
          NA_NameToIndexExitOnError(GeomSolidPatches(
                                                     GlobalsGeometries[dummy2->geom_indices[ir]]),
                                    switch_name, key);
      }

      (public_xtra->data) = (void*)dummy2;

      break;
    }

    case 3:
    {
      Type3* dummy3 = ctalloc(Type3, 1);

      sprintf(key, "Geom.%s.ICPressure.FileName", "domain");
      dummy3->filename = GetString(key);
      public_xtra->data = (void*)dummy3;

      break;
    }

    case 4:
    {
      Type4* dummy4 = ctalloc(Type4, 1);

      sprintf(ncKey, "Geom.%s.ICPressure.FileName", "domain");
      dummy4->filename = GetString(ncKey);

      sprintf(ncKey, "Geom.%s.ICPressure.TimeStep", "domain");
      dummy4->timestep = GetIntDefault(ncKey, 0);

      public_xtra->data = (void*)dummy4;

      break;
    }

    default:
    {
      InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
    }
  }

  NA_FreeNameArray(type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * ICPhasePressureFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ICPhasePressureFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->regions);

    switch ((public_xtra->type))
    {
      case 0:
      {
        Type0* dummy0 = (Type0*)(public_xtra->data);

        tfree(dummy0->region_indices);
        tfree(dummy0->values);
        tfree(dummy0);
        break;
      }

      case 1:
      {
        Type1* dummy1 = (Type1*)(public_xtra->data);

        tfree(dummy1->region_indices);
        tfree(dummy1->reference_elevations);
        tfree(dummy1->pressure_values);
        tfree(dummy1);
        break;
      }

      case 2:
      {
        Type2* dummy2 = (Type2*)(public_xtra->data);

        tfree(dummy2->region_indices);
        tfree(dummy2->patch_indices);
        tfree(dummy2->geom_indices);
        tfree(dummy2->pressure_values);
        tfree(dummy2);
        break;
      }

      case 3:
      {
        Type3* dummy3 = (Type3*)(public_xtra->data);
        tfree(dummy3);
        break;
      }

      case 4:
      {
        Type4* dummy4 = (Type4*)(public_xtra->data);
        tfree(dummy4);
        break;
      }
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * ICPhasePressureSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ICPhasePressureSizeOfTempData()
{
  int sz = 0;

  return sz;
}


