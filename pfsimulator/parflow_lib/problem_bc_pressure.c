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

#include <string.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;
  //int     iflag;   //@RMM
} PublicXtra;

typedef struct {
  Problem  *problem;
  PFModule *phase_density;

  double     ***elevations;
  ProblemData  *problem_data;
  Grid         *grid;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * BCPressure:
 *   This routine returns a BCStruct structure which describes where
 *   and what the boundary conditions are.
 *--------------------------------------------------------------------------*/

BCStruct    *BCPressure(

                        ProblemData *problem_data, /* Contains BC info transferred by the
                                                    * BCPressurePackage function */
                        Grid *       grid, /* Grid data */
                        GrGeomSolid *gr_domain, /* Gridded domain solid */
                        double       time) /* Current time - needed to determine where on
                                            * the boundary time cycle we are */
{
  PFModule       *this_module = ThisPFModule;
  PublicXtra     *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra   *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PFModule       *phase_density = (instance_xtra->phase_density);

  BCPressureData *bc_pressure_data = ProblemDataBCPressureData(problem_data);

  TimeCycleData  *time_cycle_data;

  int num_phases = (public_xtra->num_phases);

  Problem        *problem = (instance_xtra->problem);

  SubgridArray   *subgrids = GridSubgrids(grid);

  Subgrid        *subgrid;

  Vector      *z_mult = ProblemDataZmult(problem_data);
  Vector      *rsz = ProblemDataRealSpaceZ(problem_data);
  Subvector   *z_mult_sub;
  Subvector   *rsz_sub;
  double      *z_mult_dat;
  double      *rsz_dat;

  BCStruct       *bc_struct;
  double       ***values;

  double         *patch_values;
  int patch_values_size;

  int num_patches;
  int ipatch, is, i, j, k, ival, phase;
  int cycle_number, interval_number;


  //       if (time == 10000.0) {
  //      printf("time: %f \n", time);
  bc_struct = NULL;
  num_patches = BCPressureDataNumPatches(bc_pressure_data);

  if (num_patches > 0)
  {
    time_cycle_data = BCPressureDataTimeCycleData(bc_pressure_data);

    /*---------------------------------------------------------------------
     * Set up bc_struct with NULL values component
     *---------------------------------------------------------------------*/

    bc_struct = NewBCStruct(subgrids, gr_domain,
                            num_patches,
                            BCPressureDataPatchIndexes(bc_pressure_data),
                            BCPressureDataBCTypes(bc_pressure_data),
                            NULL);

    /*---------------------------------------------------------------------
     * Set up values component of bc_struct
     *---------------------------------------------------------------------*/

    values = talloc(double **, num_patches);
    memset(values, 0, num_patches * sizeof(double **));
    BCStructValues(bc_struct) = values;

    for (ipatch = 0; ipatch < num_patches; ipatch++)
    {
      values[ipatch] = talloc(double *, SubgridArraySize(subgrids));
      memset(values[ipatch], 0, SubgridArraySize(subgrids) * sizeof(double *));

      cycle_number = BCPressureDataCycleNumber(bc_pressure_data, ipatch);
      interval_number = TimeCycleDataComputeIntervalNumber(
                                                           problem, time, time_cycle_data, cycle_number);

      switch (BCPressureDataType(bc_pressure_data, ipatch))
      {
        case DirEquilRefPatch:
        {
          /* Constant pressure value specified on a reference patch.
           * Calculate hydrostatic conditions along boundary patch for
           * elevations different from reference patch elevations.
           * Hydrostatic condition is:
           * grad p - rho g grad z = 0 */

          GeomSolid       *ref_solid;
          double dz2;
          double interface_den;
          double gravity = -ProblemGravity(problem);

          int ref_patch;
          int max_its = 10;
          int ix, iy, nx;

          double         **elevations;

          GetBCPressureTypeStruct(DirEquilRefPatch, interval_data, bc_pressure_data,
                                  ipatch, interval_number);

          if (instance_xtra->elevations == NULL)
          {
            instance_xtra->elevations = talloc(double **, num_patches);
            memset(instance_xtra->elevations, 0, num_patches * sizeof(double **));
            instance_xtra->problem_data = problem_data;
            instance_xtra->grid = grid;
          }

          if (instance_xtra->elevations[ipatch] == NULL)
          {
            ref_solid = ProblemDataSolid(problem_data,
                                         DirEquilRefPatchRefSolid(interval_data));
            ref_patch = DirEquilRefPatchRefPatch(interval_data);

            /* Calculate elevations at (x,y) points on reference patch. */
            instance_xtra->elevations[ipatch] = CalcElevations(ref_solid, ref_patch, subgrids, problem_data);
          }

          elevations = instance_xtra->elevations[ipatch];

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            z_mult_sub = VectorSubvector(z_mult, is);
            rsz_sub = VectorSubvector(rsz, is);
            z_mult_dat = SubvectorData(z_mult_sub);
            rsz_dat = SubvectorData(rsz_sub);

            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });
            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);

            nx = SubgridNX(subgrid);

            dz2 = SubgridDZ(subgrid) * 0.5;

            ForPatchCellsPerFace(BC_ALL,
                                 BeforeAllCells(DoNothing),
                                 LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                 Locals(int ips, iel, iterations, phase;
                                        double ref_press, ref_den, fcn_val, nonlin_resid;
                                        double z, dtmp, density, density_der;
                                        double interface_press, offset, height; ),
                                 CellSetup({
              ref_press = DirEquilRefPatchValue(interval_data);
              PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                 (0, NULL, NULL, &ref_press, &ref_den,
                                  CALCFCN));
              ips = SubvectorEltIndex(z_mult_sub, i, j, k);
              iel = (i - ix) + (j - iy) * nx;
              fcn_val = 0.0;
              nonlin_resid = 1.0;
              iterations = -1;
            }),
                                 FACE(LeftFace, { z = rsz_dat[ips]; }),
                                 FACE(RightFace, { z = rsz_dat[ips]; }),
                                 FACE(DownFace, { z = rsz_dat[ips]; }),
                                 FACE(UpFace, { z = rsz_dat[ips]; }),
                                 FACE(BackFace, { z = rsz_dat[ips] - dz2 * z_mult_dat[ips]; }),
                                 FACE(FrontFace, { z = rsz_dat[ips] + dz2 * z_mult_dat[ips]; }),
                                 CellFinalize(
            {
              /* Solve a nonlinear problem for hydrostatic pressure
               * at points on boundary patch given pressure on reference
               * patch.  Note that the problem is only nonlinear if
               * density depends on pressure.
               *
               * The nonlinear problem to solve is:
               *   F(p) = 0
               *   F(p) = P - P_ref
               *          - 0.5*(rho(P) + rho(P_ref))*gravity*(z - z_ref)
               *
               * Newton's method is used to find a solution. */

              while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
              {
                if (iterations > -1)
                {
                  PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                     (0, NULL, NULL, &patch_values[ival],
                                      &density_der, CALCDER));
                  dtmp = 1.0 - 0.5 * density_der * gravity
                         * (z - elevations[is][iel]);
                  patch_values[ival] = patch_values[ival] - fcn_val / dtmp;
                }
                else
                {
                  patch_values[ival] = ref_press;
                }
                PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                   (0, NULL, NULL, &patch_values[ival],
                                    &density, CALCFCN));

                fcn_val = patch_values[ival] - ref_press
                          - 0.5 * (density + ref_den) * gravity
                          * (z - elevations[is][iel]);
                nonlin_resid = fabs(fcn_val);

                iterations++;
              }                                 /* End of while loop */


              /* Iterate over the phases and reset pressures according to
               * hydrostatic conditions with appropriate densities.
               * At each interface, we have hydrostatic conditions, so
               *
               * z_inter = (P_inter - P_ref) /
               *            (0.5*(rho(P_inter)+rho(P_ref))*gravity
               + z_ref
               +
               + Thus, the interface height and pressure are known
               + and hydrostatic conditions can be determined for
               + new phase.
               +
               + NOTE:  This only works for Pc = 0. */

              for (phase = 1; phase < num_phases; phase++)
              {
                interface_press = DirEquilRefPatchValueAtInterface(
                                                                   interval_data, phase);
                PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                   (phase - 1, NULL, NULL, &interface_press,
                                    &interface_den, CALCFCN));
                offset = (interface_press - ref_press)
                         / (0.5 * (interface_den + ref_den) * gravity);
                ref_press = interface_press;
                PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                   (phase, NULL, NULL, &ref_press, &ref_den,
                                    CALCFCN));

                /* Only reset pressure value if in another phase.
                 * The following "if" test determines whether this point
                 * is in another phase by checking if the computed
                 * pressure is less than the interface value.  This
                 * test ONLY works if the phases are distributed such
                 * that the lighter phases are above the heavier ones. */

                if (patch_values[ival] < interface_press)
                {
                  height = elevations[is][iel];
                  nonlin_resid = 1.0;
                  iterations = -1;
                  while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
                  {
                    if (iterations > -1)
                    {
                      PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                         (phase, NULL, NULL, &patch_values[ival],
                                          &density_der, CALCDER));

                      dtmp = 1.0 - 0.5 * density_der * gravity
                             * (z - height);
                      patch_values[ival] = patch_values[ival]
                                           - fcn_val / dtmp;
                    }
                    else
                    {
                      height = height + offset;
                      patch_values[ival] = ref_press;
                    }

                    PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                       (phase, NULL, NULL,
                                        &patch_values[ival], &density,
                                        CALCFCN));

                    fcn_val = patch_values[ival] - ref_press
                              - 0.5 * (density + ref_den)
                              * gravity * (z - height);
                    nonlin_resid = fabs(fcn_val);

                    iterations++;
                  }                                   /* End of while loop */
                }                                     /* End if above interface */
              }                                       /* End phase loop */
            }),
                                 AfterAllCells(DoNothing)
                                 ); /* End ForPatchCellsPerFace loop */
          }                      /* End subgrid loop */


          break;
        } /* End DirEquilRefPatch */

        case DirEquilPLinear:
        {
          /* Piecewise linear pressure value specified on reference
           * patch.
           * Calculate hydrostatic conditions along patch for
           * elevations different from reference patch elevations.
           * Hydrostatic condition is:
           *             grad p - rho g grad z = 0 */

          int num_points;
          int ip;

          double dx2, dy2, dz2;
          double unitx, unity, line_min, line_length, xy, slope;

          double dtmp, offset, interface_press, interface_den;
          double ref_den, ref_press;
          double density_der, density;
          double height;
          double gravity = -ProblemGravity(problem);

          int max_its = 10;

          GetBCPressureTypeStruct(DirEquilPLinear, interval_data, bc_pressure_data,
                                  ipatch, interval_number);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            z_mult_sub = VectorSubvector(z_mult, is);
            rsz_sub = VectorSubvector(rsz, is);
            z_mult_dat = SubvectorData(z_mult_sub);
            rsz_dat = SubvectorData(rsz_sub);

            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            dx2 = SubgridDX(subgrid) / 2.0;
            dy2 = SubgridDY(subgrid) / 2.0;
            dz2 = SubgridDZ(subgrid) / 2.0;

            /* compute unit direction vector for piecewise linear line */
            unitx = DirEquilPLinearXUpper(interval_data)
                    - DirEquilPLinearXLower(interval_data);
            unity = DirEquilPLinearYUpper(interval_data)
                    - DirEquilPLinearYLower(interval_data);
            line_length = sqrt(unitx * unitx + unity * unity);
            unitx /= line_length;
            unity /= line_length;
            line_min = DirEquilPLinearXLower(interval_data) * unitx
                       + DirEquilPLinearYLower(interval_data) * unity;

            ForPatchCellsPerFace(BC_ALL,
                                 BeforeAllCells(DoNothing),
                                 LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                 Locals(int ips, iterations;
                                        double x, y, z, fcn_val, nonlin_resid; ),
                                 CellSetup({
              ips = SubvectorEltIndex(z_mult_sub, i, j, k);
              x = RealSpaceX(i, SubgridRX(subgrid));
              y = RealSpaceY(j, SubgridRY(subgrid));
              z = rsz_dat[ips];
              fcn_val = 0.0;
              nonlin_resid = 1.0;
              iterations = -1;
            }),
                                 FACE(LeftFace, { x = x - dx2; }),
                                 FACE(RightFace, { x = x + dx2; }),
                                 FACE(DownFace, { y = y - dy2; }),
                                 FACE(UpFace, { y = y + dy2; }),
                                 FACE(BackFace, { z = z - dz2 * z_mult_dat[ips]; }),
                                 FACE(FrontFace, { z = z + dz2 * z_mult_dat[ips]; }),
                                 CellFinalize(
            {
              /* project center of BC face onto piecewise line */
              xy = (x * unitx + y * unity - line_min) / line_length;

              /* find two neighboring points */
              ip = 1;
              num_points = DirEquilPLinearNumPoints(interval_data);
              for (; ip < (num_points - 1); ip++)
              {
                if (xy < DirEquilPLinearPoint(interval_data, ip))
                  break;
              }

              /* compute the slope */
              slope = ((DirEquilPLinearValue(interval_data, ip)
                        - DirEquilPLinearValue(interval_data, (ip - 1)))
                       / (DirEquilPLinearPoint(interval_data, ip)
                          - DirEquilPLinearPoint(interval_data, (ip - 1))));

              ref_press = DirEquilPLinearValue(interval_data, ip - 1)
                          + slope * (xy - DirEquilPLinearPoint(interval_data, ip - 1));
              PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                 (0, NULL, NULL, &ref_press, &ref_den,
                                  CALCFCN));

              /* Solve a nonlinear problem for hydrostatic pressure
               * at points on boundary patch given reference pressure.
               * Note that the problem is only nonlinear if
               * density depends on pressure.
               *
               * The nonlinear problem to solve is:
               *   F(p) = 0
               *   F(p) = P - P_ref
               *          - 0.5*(rho(P) + rho(P_ref))*gravity*z
               *
               * Newton's method is used to find a solution. */

              while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
              {
                if (iterations > -1)
                {
                  PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                     (0, NULL, NULL, &patch_values[ival],
                                      &density_der, CALCDER));
                  dtmp = 1.0 - 0.5 * density_der * gravity * z;
                  patch_values[ival] = patch_values[ival] - fcn_val / dtmp;
                }
                else
                {
                  patch_values[ival] = ref_press;
                }
                PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                   (0, NULL, NULL, &patch_values[ival],
                                    &density, CALCFCN));

                fcn_val = patch_values[ival] - ref_press
                          - 0.5 * (density + ref_den) * gravity * z;
                nonlin_resid = fabs(fcn_val);

                iterations++;
              }                                 /* End of while loop */

              /* Iterate over the phases and reset pressures according to
               * hydrostatic conditions with appropriate densities.
               * At each interface, we have hydrostatic conditions, so
               *
               * z_inter = (P_inter - P_ref) /
               *            (0.5*(rho(P_inter)+rho(P_ref))*gravity
               + z_ref
               +
               + Thus, the interface height and pressure are known
               + and hydrostatic conditions can be determined for
               + new phase.
               +
               + NOTE:  This only works for Pc = 0. */

              for (phase = 1; phase < num_phases; phase++)
              {
                interface_press = DirEquilPLinearValueAtInterface(
                                                                  interval_data, phase);
                PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                   (phase - 1, NULL, NULL, &interface_press,
                                    &interface_den, CALCFCN));
                offset = (interface_press - ref_press)
                         / (0.5 * (interface_den + ref_den) * gravity);
                ref_press = interface_press;
                PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                   (phase, NULL, NULL, &ref_press, &ref_den,
                                    CALCFCN));

                /* Only reset pressure value if in another phase.
                 * The following "if" test determines whether this point
                 * is in another phase by checking if the computed
                 * pressure is less than the interface value.  This
                 * test ONLY works if the phases are distributed such
                 * that the lighter phases are above the heavier ones. */

                if (patch_values[ival] < interface_press)
                {
                  height = 0.0;
                  nonlin_resid = 1.0;
                  iterations = -1;
                  while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
                  {
                    if (iterations > -1)
                    {
                      PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                         (phase, NULL, NULL, &patch_values[ival],
                                          &density_der, CALCDER));

                      dtmp = 1.0 - 0.5 * density_der * gravity * (z - height);
                      patch_values[ival] = patch_values[ival]
                                           - fcn_val / dtmp;
                    }
                    else
                    {
                      height = height + offset;
                      patch_values[ival] = ref_press;
                    }

                    PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                                       (phase, NULL, NULL,
                                        &patch_values[ival], &density,
                                        CALCFCN));

                    fcn_val = patch_values[ival] - ref_press
                              - 0.5 * (density + ref_den) * gravity
                              * (z - height);
                    nonlin_resid = fabs(fcn_val);

                    iterations++;
                  }                                   /* End of while loop */
                }                                     /* End if above interface */
              }                                       /* End phase loop */
            }),
                                 AfterAllCells(DoNothing)
                                 ); /* End ForPatchCellsPerFace */
          }
          break;
        } /* End DirEquilPLinear */

        case FluxConst:
        {
          /* Constant flux rate value on patch */
          double flux;

          GetBCPressureTypeStruct(FluxConst, interval_data, bc_pressure_data,
                                  ipatch, interval_number);

          flux = FluxConstValue(interval_data);
          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values[ival] = flux;
            });
          }       /* End subgrid loop */
          break;
        } /* End FluxConst */

        case FluxVolumetric:
        {
          /* Constant volumetric flux value on patch */
          double dx, dy, dz;
          double area, volumetric_flux;

          GetBCPressureTypeStruct(FluxVolumetric, interval_data, bc_pressure_data,
                                  ipatch, interval_number);


          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            z_mult_sub = VectorSubvector(z_mult, is);
            z_mult_dat = SubvectorData(z_mult_sub);

            dx = SubgridDX(subgrid);
            dy = SubgridDY(subgrid);
            dz = SubgridDZ(subgrid);

            area = 0.0;
            patch_values_size = 0;
            ForPatchCellsPerFace(BC_ALL,
                                 BeforeAllCells(DoNothing),
                                 LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                 Locals(int ips; ),
                                 CellSetup({
              patch_values_size++;
              ips = SubvectorEltIndex(z_mult_sub, i, j, k);
            }),
                                 FACE(LeftFace, { area += dy * dz * z_mult_dat[ips]; }),
                                 FACE(RightFace, { area += dy * dz * z_mult_dat[ips]; }),
                                 FACE(DownFace, { area += dx * dz * z_mult_dat[ips]; }),
                                 FACE(UpFace, { area += dx * dz * z_mult_dat[ips]; }),
                                 FACE(BackFace, { area += dx * dy; }),
                                 FACE(FrontFace, { area += dx * dy; }),
                                 CellFinalize(DoNothing),
                                 AfterAllCells(DoNothing)
                                 );

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            if (area > 0.0)
            {
              volumetric_flux = FluxVolumetricValue(interval_data)
                                / area;
              ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
              {
                patch_values[ival] = volumetric_flux;
              });
            }
          }           /* End subgrid loop */
          break;
        } /* End FluxVolumetric */

        case PressureFile:
        {
          /* Read input pressures from file (temporary).
           * This case assumes hydraulic head input conditions and
           * a constant density.  */
          Vector          *tmp_vector;
          Subvector       *subvector;
          char            *filename;
          double          *tmpp;
          int itmp;
          double density, dtmp;

          /* Calculate density using dtmp as dummy argument. */
          dtmp = 0.0;
          PFModuleInvokeType(PhaseDensityInvoke, phase_density,
                             (0, NULL, NULL, &dtmp, &density, CALCFCN));

          GetBCPressureTypeStruct(PressureFile, interval_data, bc_pressure_data,
                                  ipatch, interval_number);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            z_mult_sub = VectorSubvector(z_mult, is);
            rsz_sub = VectorSubvector(rsz, is);
            z_mult_dat = SubvectorData(z_mult_sub);
            rsz_dat = SubvectorData(rsz_sub);

            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            tmp_vector = NewVectorType(grid, 1, 0, vector_cell_centered);

            filename = PressureFileName(interval_data);
            ReadPFBinary(filename, tmp_vector);

            subvector = VectorSubvector(tmp_vector, is);

            tmpp = SubvectorData(subvector);
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              /*int ips = SubvectorEltIndex(z_mult_sub, i, j, k);*/
              itmp = SubvectorEltIndex(subvector, i, j, k);

              patch_values[ival] = tmpp[itmp];     /*- density*gravity*z;*/
              /*last part taken out, very likely to be a bug)*/
            });

            FreeVector(tmp_vector);
          }             /* End subgrid loop */
          break;
        } /* End PressureFile */

        case FluxFile:
        {
          /* Read input fluxes from file (temporary) */
          Vector          *tmp_vector;
          Subvector       *subvector;
          char            *filename;
          double          *tmpp;
          int itmp;

          GetBCPressureTypeStruct(FluxFile, interval_data, bc_pressure_data,
                                  ipatch, interval_number);


          ForSubgridI(is, subgrids)
          {
            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            tmp_vector = NewVectorType(grid, 1, 0, vector_cell_centered);

            filename = FluxFileName(interval_data);
            ReadPFBinary(filename, tmp_vector);

            subvector = VectorSubvector(tmp_vector, is);

            tmpp = SubvectorData(subvector);
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              itmp = SubvectorEltIndex(subvector, i, j, k);

              patch_values[ival] = tmpp[itmp];
            });

            FreeVector(tmp_vector);
          }         /* End subgrid loop */
          break;
        } /* End FluxFile */

        case ExactSolution:
        {
          /* Calculate pressure based on pre-defined functions */
          double dx2, dy2, dz2;
          int fcn_type;

          GetBCPressureTypeStruct(ExactSolution, interval_data, bc_pressure_data,
                                  ipatch, interval_number);

          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            z_mult_sub = VectorSubvector(z_mult, is);
            rsz_sub = VectorSubvector(rsz, is);
            z_mult_dat = SubvectorData(z_mult_sub);
            rsz_dat = SubvectorData(rsz_sub);

            /* compute patch_values_size */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            dx2 = SubgridDX(subgrid) / 2.0;
            dy2 = SubgridDY(subgrid) / 2.0;
            dz2 = SubgridDZ(subgrid) / 2.0;

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            fcn_type = ExactSolutionFunctionType(interval_data);

            switch (fcn_type)
            {
              case 1:  /* p = x */
              {
                ForPatchCellsPerFace(BC_ALL,
                                     BeforeAllCells(DoNothing),
                                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                     Locals(double x; ),
                                     CellSetup({ x = RealSpaceX(i, SubgridRX(subgrid)); }),
                                     FACE(LeftFace, { x = x - dx2; }),
                                     FACE(RightFace, { x = x + dx2; }),
                                     FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                                     FACE(BackFace, DoNothing), FACE(FrontFace, DoNothing),
                                     CellFinalize({ patch_values[ival] = x; }),
                                     AfterAllCells(DoNothing)
                                     );
                break;
              }     /* End case 1 */

              case 2:  /* p = x + y + z */
              {
                ForPatchCellsPerFace(BC_ALL,
                                     BeforeAllCells(DoNothing),
                                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                     Locals(int ips;
                                            double x, y, z; ),
                                     CellSetup({
                  ips = SubvectorEltIndex(z_mult_sub, i, j, k);
                  x = RealSpaceX(i, SubgridRX(subgrid));
                  y = RealSpaceY(j, SubgridRY(subgrid));
                  z = rsz_dat[ips];
                }),
                                     FACE(LeftFace, { x = x - dx2; }),
                                     FACE(RightFace, { x = x + dx2; }),
                                     FACE(DownFace, { y = y - dy2; }),
                                     FACE(UpFace, { y = y + dy2; }),
                                     FACE(BackFace, { z = z - dz2 * z_mult_dat[ips]; }),
                                     FACE(FrontFace, { z = z + dz2 * z_mult_dat[ips]; }),
                                     CellFinalize({ patch_values[ival] = x + y + z; }),
                                     AfterAllCells(DoNothing)
                                     );
                break;
              }     /* End case 2 */

              case 3:  /* p = x^3y^2 + sinxy + 1*/
              {
                ForPatchCellsPerFace(BC_ALL,
                                     BeforeAllCells(DoNothing),
                                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                     Locals(double x, y; ),
                                     CellSetup({
                  x = RealSpaceX(i, SubgridRX(subgrid));
                  y = RealSpaceY(j, SubgridRY(subgrid));
                }),
                                     FACE(LeftFace, { x = x - dx2; }),
                                     FACE(RightFace, { x = x + dx2; }),
                                     FACE(DownFace, { y = y - dy2; }),
                                     FACE(UpFace, { y = y + dy2; }),
                                     FACE(BackFace, DoNothing), FACE(FrontFace, DoNothing),
                                     CellFinalize({
                  patch_values[ival] = x * x * x * y * y + sin(x * y) + 1;
                }),
                                     AfterAllCells(DoNothing)
                                     );
                break;
              }     /* End case 3 */

              case 4:  /* p = x^3 y^4 + x^2 + sinxy cosy + 1 */
              {
                ForPatchCellsPerFace(BC_ALL,
                                     BeforeAllCells(DoNothing),
                                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                     Locals(double x, y; ),
                                     CellSetup({
                  x = RealSpaceX(i, SubgridRX(subgrid));
                  y = RealSpaceY(j, SubgridRY(subgrid));
                }),
                                     FACE(LeftFace, { x = x - dx2; }),
                                     FACE(RightFace, { x = x + dx2; }),
                                     FACE(DownFace, { y = y - dy2; }),
                                     FACE(UpFace, { y = y + dy2; }),
                                     FACE(BackFace, DoNothing), FACE(FrontFace, DoNothing),
                                     CellFinalize({
                  patch_values[ival] = pow(x, 3) * pow(y, 4) + x * x + sin(x * y) * cos(y) + 1;
                }),
                                     AfterAllCells(DoNothing)
                                     );
                break;
              }     /* End case 4 */

              case 5:  /* p = xyzt +1 */
              {
                ForPatchCellsPerFace(BC_ALL,
                                     BeforeAllCells(DoNothing),
                                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                     Locals(int ips; double x, y, z; ),
                                     CellSetup({
                  ips = SubvectorEltIndex(z_mult_sub, i, j, k);
                  x = RealSpaceX(i, SubgridRX(subgrid));
                  y = RealSpaceY(j, SubgridRY(subgrid));
                  z = rsz_dat[ips];
                }),
                                     FACE(LeftFace, { x = x - dx2; }),
                                     FACE(RightFace, { x = x + dx2; }),
                                     FACE(DownFace, { y = y - dy2; }),
                                     FACE(UpFace, { y = y + dy2; }),
                                     FACE(BackFace, { z = z - dz2 * z_mult_dat[ips]; }),
                                     FACE(FrontFace, { z = z + dz2 * z_mult_dat[ips]; }),
                                     CellFinalize({ patch_values[ival] = x * y * z * time + 1; }),
                                     AfterAllCells(DoNothing)
                                     );
                break;
              }     /* End case 5 */

              case 6:  /* p = xyzt +1 */
              {
                ForPatchCellsPerFace(BC_ALL,
                                     BeforeAllCells(DoNothing),
                                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                                     Locals(int ips; double x, y, z; ),
                                     CellSetup({
                  ips = SubvectorEltIndex(z_mult_sub, i, j, k);
                  x = RealSpaceX(i, SubgridRX(subgrid));
                  y = RealSpaceY(j, SubgridRY(subgrid));
                  z = rsz_dat[ips];
                }),
                                     FACE(LeftFace, { x = x - dx2; }),
                                     FACE(RightFace, { x = x + dx2; }),
                                     FACE(DownFace, { y = y - dy2; }),
                                     FACE(UpFace, { y = y + dy2; }),
                                     FACE(BackFace, { z = z - dz2 * z_mult_dat[ips]; }),
                                     FACE(FrontFace, { z = z + dz2 * z_mult_dat[ips]; }),
                                     CellFinalize({ patch_values[ival] = x * y * z * time + 1; }),
                                     AfterAllCells(DoNothing)
                                     );
                break;
              }     /* End case 5 */
            }       /* End switch */
          }         /* End subgrid loop */
          break;
        } /* End ExactSolution */

        case OverlandFlow:
        {
          /* Constant "rainfall" rate value on patch */
          double flux;

          GetBCPressureTypeStruct(FluxConst, interval_data, bc_pressure_data,
                                  ipatch, interval_number);


          flux = OverlandFlowValue(interval_data);
          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values[ival] = flux;
            });
          }       /* End subgrid loop */
          break;
        } /* End OverlandFlow */

        case OverlandFlowPFB:
        {
          /* Read input fluxes from file (overland) */
          Vector          *tmp_vector;
          Subvector       *subvector;
          //double          *data;
          char            *filename;
          double          *tmpp;
          int itmp;

          GetBCPressureTypeStruct(OverlandFlowPFB, interval_data, bc_pressure_data,
                                  ipatch, interval_number);

          ForSubgridI(is, subgrids)
          {
            /* compute patch_values_size (this isn't really needed yet) */
            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            tmp_vector = NewVectorType(grid, 1, 0, vector_cell_centered);
            //data = ctalloc(double, SizeOfVector(tmp_vector));
            //SetTempVectorData(tmp_vector, data);

            printf("reading overland file \n");
            filename = OverlandFlowPFBFileName(interval_data);
            ReadPFBinary(filename, tmp_vector);

            subvector = VectorSubvector(tmp_vector, is);

            tmpp = SubvectorData(subvector);
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              itmp = SubvectorEltIndex(subvector, i, j, k);

              patch_values[ival] = tmpp[itmp];
            });

            //tfree(VectorData(tmp_vector));
            FreeVector(tmp_vector);
          }              /* End subgrid loop */
          break;
        } /* End OverlandFlowPFB */

        case SeepageFace:
        {
          GetBCPressureTypeStruct(SeepageFace, interval_data, bc_pressure_data,
                                  ipatch, interval_number);
          double flux;

          flux = SeepageFaceValue(interval_data);
          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values[ival] = flux;
            });
          }

          break;
        } /* End SeepageFace */

        case OverlandKinematic:
        {
          GetBCPressureTypeStruct(OverlandKinematic, interval_data, bc_pressure_data,
                                  ipatch, interval_number);
          double flux;

          flux = OverlandKinematicValue(interval_data);
          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values[ival] = flux;
            });
          }

          break;
        } /* End OverlandKinematic */

        case OverlandDiffusive:
        {
          GetBCPressureTypeStruct(OverlandDiffusive, interval_data, bc_pressure_data,
                                  ipatch, interval_number);
          double flux;

          flux = OverlandDiffusiveValue(interval_data);
          ForSubgridI(is, subgrids)
          {
            subgrid = SubgridArraySubgrid(subgrids, is);

            patch_values_size = 0;
            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values_size++;
            });

            patch_values = talloc(double, patch_values_size);
            memset(patch_values, 0, patch_values_size * sizeof(double));
            values[ipatch][is] = patch_values;

            ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is,
            {
              patch_values[ival] = flux;
            });
          }

          break;
        } /* End OverlandDiffusive */
      }
    }
  }
  //}
  return bc_struct;
}


/*--------------------------------------------------------------------------
 * BCPressureInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *BCPressureInitInstanceXtra(Problem *problem)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    instance_xtra = talloc(InstanceXtra, 1);
    memset(instance_xtra, 0, sizeof(InstanceXtra));
  }
  else
  {
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `problem'
   *-----------------------------------------------------------------------*/

  if (problem != NULL)
  {
    instance_xtra->problem = problem;
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->phase_density) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->phase_density), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * BCPressureFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void BCPressureFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    if (instance_xtra->elevations)
    {
      ProblemData *problem_data = instance_xtra->problem_data;
      BCPressureData *bc_pressure_data = ProblemDataBCPressureData(problem_data);
      int num_patches;
      SubgridArray   *subgrids = GridSubgrids(instance_xtra->grid);
      int ipatch;
      int is;

      num_patches = BCPressureDataNumPatches(bc_pressure_data);

      for (ipatch = 0; ipatch < num_patches; ipatch++)
      {
        if (instance_xtra->elevations[ipatch])
        {
          ForSubgridI(is, subgrids)
          {
            if (instance_xtra->elevations[ipatch][is])
            {
              tfree(instance_xtra->elevations[ipatch][is]);
            }
          }

          tfree(instance_xtra->elevations[ipatch]);
        }
      }

      tfree(instance_xtra->elevations);
    }
    PFModuleFreeInstance(instance_xtra->phase_density);
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * BCPressureNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *BCPressureNewPublicXtra(
                                   int num_phases)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  /* allocate space for the public_xtra structure */
  public_xtra = talloc(PublicXtra, 1);
  memset(public_xtra, 0, sizeof(PublicXtra));

  (public_xtra->num_phases) = num_phases;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * BCPressureFreePublicXtra
 *--------------------------------------------------------------------------*/

void  BCPressureFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * BCPressureSizeOfTempData
 *--------------------------------------------------------------------------*/

int  BCPressureSizeOfTempData()
{
  return 0;
}
