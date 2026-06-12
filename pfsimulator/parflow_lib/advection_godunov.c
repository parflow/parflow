/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
* Advection box to perform the godunov step.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int time_index;
  int high_order;
  int transverse;
  int enforce_minmax;
  int dim[3];
} PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Problem *problem;
  Grid    *grid;
  double  *temp_data;

  /* instance data */
  int max_nx;
  int max_ny;
  int max_nz;
  double *fx;
  double *fy;
  double *fz;
  double *vx;
  double *wx;
  double *uy;
  double *wy;
  double *uz;
  double *vz;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * Godunov
 *--------------------------------------------------------------------------*/

void     Godunov(
                 ProblemData *problem_data,
                 int          phase,
                 int          concentration,
                 Vector *     old_concentration,
                 Vector *     new_concentration,
                 Vector *     x_velocity,
                 Vector *     y_velocity,
                 Vector *     z_velocity,
                 Vector *     old_porsat,
                 Vector *     new_porsat_inv,
                 double       time,
                 double       deltat)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Problem    *problem = (instance_xtra->problem);
  Grid       *grid = (instance_xtra->grid);

  Vector     *scale = NULL;
  Vector     *right_hand_side = NULL;
  Vector     *min_concen = NULL;
  Vector     *max_concen = NULL;

  double *fx = (instance_xtra->fx);
  double *fy = (instance_xtra->fy);
  double *fz = (instance_xtra->fz);
  double *vx = (instance_xtra->vx);
  double *wx = (instance_xtra->wx);
  double *uy = (instance_xtra->uy);
  double *wy = (instance_xtra->wy);
  double *uz = (instance_xtra->uz);
  double *vz = (instance_xtra->vz);


  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;
  WellDataStat     *well_data_stat;

  TimeCycleData    *time_cycle_data;

  Vector           *perm_x = ProblemDataPermeabilityX(problem_data);
  Vector           *perm_y = ProblemDataPermeabilityY(problem_data);
  Vector           *perm_z = ProblemDataPermeabilityZ(problem_data);

  VectorUpdateCommHandle       *handle = NULL;

  SubgridArray     *subgrids;
  Subgrid          *subgrid,
                   *well_subgrid,
                   *tmp_subgrid;
  Subvector        *subvector,
                   *subvector_scal,
                   *subvector_rhs,
                   *subvector_xvel,
                   *subvector_yvel,
                   *subvector_zvel,
                   *subvector_npsi,
                   *concen_sub,
                   *min_sub,
                   *max_sub,
                   *px_sub,
                   *py_sub,
                   *pz_sub;

  int sg, well;
  int ix, iy, iz;
  int nx, ny, nz;
  double dx, dy, dz;
  int nx_c, ny_c, nz_c,
      nx_m, ny_m, nz_m,
      nx_p, ny_p, nz_p,
      nx_w, ny_w, nz_w,
      nx_xv, ny_xv, nz_xv,
      nx_yv, ny_yv, nz_yv,
      nx_zv, ny_zv, nz_zv;

  int i, j, k, ci, mi, pi, wi, xi, yi, zi;
  int nx_cells, ny_cells, nz_cells, index, flopest;
  double lambda, decay_factor;

  double *c, *cn;
  double *rhs, *scal, *ops, *npsi;
  double *px, *py, *pz;
  double *xvel_u, *xvel_l, *yvel_u, *yvel_l, *zvel_u, *zvel_l;
  double *c_xl, *c_xu, *c_yl, *c_yu, *c_zl, *c_zu;
  double *uedge, *vedge, *wedge;
  double *min, *max;

  int cycle_number, interval_number;
  int dlo[3], dhi[3];
  double hx[3];
  double dt;
  double cell_volume, field_sum, total_volume, cell_change,
         well_stat, contaminant_stat;
  double well_value, input_c, volume, flux, scaled_flux, weight = 0;
  double avg_x, avg_y, avg_z, area_x, area_y, area_z, area_sum;

  amps_Invoice result_invoice;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  BeginTiming(public_xtra->time_index);

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/

  if (WellDataNumWells(well_data) > 0)
  {
    scale = NewVectorType(instance_xtra->grid, 1, 2, vector_cell_centered);
    right_hand_side = NewVectorType(instance_xtra->grid, 1, 2, vector_cell_centered);
  }


  if (public_xtra->enforce_minmax)
  {
    min_concen = NewVectorType(instance_xtra->grid, 1, 0, vector_cell_centered);
    max_concen = NewVectorType(instance_xtra->grid, 1, 0, vector_cell_centered);
  }

  /*-----------------------------------------------------------------------
   * Initialize some data
   *-----------------------------------------------------------------------*/
  dt = deltat;

  subgrids = GridSubgrids(grid);

  nx_cells = IndexSpaceNX(0);
  ny_cells = IndexSpaceNY(0);
  nz_cells = IndexSpaceNZ(0);

  /*-----------------------------------------------------------------------
   * Advect on all the subgrids
   *-----------------------------------------------------------------------*/

  ForSubgridI(sg, subgrids)
  {
    subgrid = GridSubgrid(grid, sg);

    /**** Get locations for subvector data of vectors passed in ****/
    c = SubvectorData(VectorSubvector(old_concentration, sg));
    cn = SubvectorData(VectorSubvector(new_concentration, sg));
    ops = SubvectorData(VectorSubvector(old_porsat, sg));
    npsi = SubvectorData(VectorSubvector(new_porsat_inv, sg));
    uedge = SubvectorData(VectorSubvector(x_velocity, sg));
    vedge = SubvectorData(VectorSubvector(y_velocity, sg));
    wedge = SubvectorData(VectorSubvector(z_velocity, sg));

    /***** Compute extents of data *****/
    dlo[0] = SubgridIX(subgrid);
    dlo[1] = SubgridIY(subgrid);
    dlo[2] = SubgridIZ(subgrid);
    dhi[0] = SubgridIX(subgrid) + (SubgridNX(subgrid) - 1);
    dhi[1] = SubgridIY(subgrid) + (SubgridNY(subgrid) - 1);
    dhi[2] = SubgridIZ(subgrid) + (SubgridNZ(subgrid) - 1);

    /***** Compute the grid spacing *****/
    hx[0] = SubgridDX(subgrid);
    hx[1] = SubgridDY(subgrid);
    hx[2] = SubgridDZ(subgrid);

    /***** Make the call to the low-order advection routine *****/
    CALL_ADVECT_UPWIND(c, cn, uedge, vedge, wedge, ops, npsi,
                       fx, fy, fz, dlo, dhi, hx, public_xtra->dim, dt);
  }


  if (public_xtra->enforce_minmax)
  {
    handle = InitVectorUpdate(new_concentration, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    ForSubgridI(sg, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, sg);

      concen_sub = VectorSubvector(new_concentration, sg);
      min_sub = VectorSubvector(min_concen, sg);
      max_sub = VectorSubvector(max_concen, sg);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_c = SubvectorNX(concen_sub);
      ny_c = SubvectorNY(concen_sub);
      nz_c = SubvectorNZ(concen_sub);

      nx_m = SubvectorNX(min_sub);
      ny_m = SubvectorNY(min_sub);
      nz_m = SubvectorNZ(min_sub);

      min = SubvectorElt(min_sub, ix, iy, iz);
      max = SubvectorElt(max_sub, ix, iy, iz);

      c = SubvectorElt(concen_sub, ix, iy, iz);
      c_xl = SubvectorElt(concen_sub, ix - 1, iy, iz);
      c_xu = SubvectorElt(concen_sub, ix + 1, iy, iz);
      c_yl = SubvectorElt(concen_sub, ix, iy - 1, iz);
      c_yu = SubvectorElt(concen_sub, ix, iy + 1, iz);
      c_zl = SubvectorElt(concen_sub, ix, iy, iz - 1);
      c_zu = SubvectorElt(concen_sub, ix, iy, iz + 1);

      ci = 0;
      mi = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                mi, nx_m, ny_m, nz_m, 1, 1, 1,
                ci, nx_c, ny_c, nz_c, 1, 1, 1,
      {
        min[mi] = pfmin(c[ci], pfmin(c_xl[ci], pfmin(c_xu[ci], pfmin(c_yl[ci], pfmin(c_yu[ci], pfmin(c_zl[ci], c_zu[ci]))))));
        max[mi] = pfmax(c[ci], pfmax(c_xl[ci], pfmax(c_xu[ci], pfmax(c_yl[ci], pfmax(c_yu[ci], pfmax(c_zl[ci], c_zu[ci]))))));
      });
    }
  }

  /*-----------------------------------------------------------------------
   * Adjust for degradation
   *-----------------------------------------------------------------------*/

  lambda = ProblemContaminantDegradation(problem, concentration);

  contaminant_stat = 0.0;
  if (lambda != 0.0)
  {
    flopest = 3 * nx_cells * ny_cells * nz_cells;
    decay_factor = exp(-1.0 * lambda * dt);

    ForSubgridI(sg, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, sg);

      subvector = VectorSubvector(new_concentration, sg);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);

      nx_c = SubvectorNX(subvector);
      ny_c = SubvectorNY(subvector);
      nz_c = SubvectorNZ(subvector);

      cell_volume = dx * dy * dz;

      cn = SubvectorElt(subvector, ix, iy, iz);

      ci = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                ci, nx_c, ny_c, nz_c, 1, 1, 1,
      {
        cn[ci] = cn[ci] * decay_factor;

        cell_change = cn[ci] * ((decay_factor - 1.0) / decay_factor);
        contaminant_stat += cell_change * cell_volume;
      });
    }

    IncFLOPCount(flopest);
  }


  /*-----------------------------------------------------------------------
   * Set up well terms, right hand side and scaling term
   *-----------------------------------------------------------------------*/

  index = phase * WellDataNumContaminants(well_data) + concentration;

  if (WellDataNumWells(well_data) > 0)
  {
    InitVectorAll(scale, 0.0);
    InitVectorAll(right_hand_side, 0.0);

    time_cycle_data = WellDataTimeCycleData(well_data);
    for (well = 0; well < WellDataNumPressWells(well_data); well++)
    {
      well_data_physical = WellDataPressWellPhysical(well_data, well);
      cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
      interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);
      well_data_value = WellDataPressWellIntervalValue(well_data, well, interval_number);
      well_data_stat = WellDataPressWellStat(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      volume = WellDataPhysicalSize(well_data_physical);

      total_volume = 0.0;

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);
        subvector_npsi = VectorSubvector(new_porsat_inv, sg);
        subvector_xvel = VectorSubvector(x_velocity, sg);
        subvector_yvel = VectorSubvector(y_velocity, sg);
        subvector_zvel = VectorSubvector(z_velocity, sg);

        nx_w = SubvectorNX(subvector_scal);      /*scal, rhs & npsi share nx_w */
        ny_w = SubvectorNY(subvector_scal);      /*scal, rhs & npsi share ny_w */
        nz_w = SubvectorNZ(subvector_scal);      /*scal, rhs & npsi share nz_w */

        nx_xv = SubvectorNX(subvector_xvel);
        ny_xv = SubvectorNY(subvector_xvel);
        nz_xv = SubvectorNZ(subvector_xvel);

        nx_yv = SubvectorNX(subvector_yvel);
        ny_yv = SubvectorNY(subvector_yvel);
        nz_yv = SubvectorNZ(subvector_yvel);

        nx_zv = SubvectorNX(subvector_zvel);
        ny_zv = SubvectorNY(subvector_zvel);
        nz_zv = SubvectorNZ(subvector_zvel);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          cell_volume = dx * dy * dz;

          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);
          npsi = SubvectorElt(subvector_npsi, ix, iy, iz);

          xvel_l = SubvectorElt(subvector_xvel, ix, iy, iz);
          xvel_u = SubvectorElt(subvector_xvel, ix + 1, iy, iz);

          yvel_l = SubvectorElt(subvector_yvel, ix, iy, iz);
          yvel_u = SubvectorElt(subvector_yvel, ix, iy + 1, iz);

          zvel_l = SubvectorElt(subvector_zvel, ix, iy, iz);
          zvel_u = SubvectorElt(subvector_zvel, ix, iy, iz + 1);

          if (WellDataPhysicalAction(well_data_physical)
              == INJECTION_WELL)
          {
            if (WellDataValueDeltaContaminantPtrs(well_data_value))
            {
              input_c =
                WellDataValueContaminantFraction(well_data_value,
                                                 index)
                * fabs(WellDataValueDeltaContaminantPtr(well_data_value,
                                                        index))
                / volume;
            }
            else
            {
              input_c = WellDataValueContaminantValue(well_data_value,
                                                      index);
            }

            if (public_xtra->enforce_minmax)
            {
              min_sub = VectorSubvector(min_concen, sg);
              max_sub = VectorSubvector(max_concen, sg);

              nx_m = SubvectorNX(min_sub);
              ny_m = SubvectorNY(min_sub);
              nz_m = SubvectorNZ(min_sub);

              min = SubvectorElt(min_sub, ix, iy, iz);
              max = SubvectorElt(max_sub, ix, iy, iz);

              mi = 0;
              BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                        mi, nx_m, ny_m, nz_m, 1, 1, 1,
              {
                min[mi] = pfmin(input_c, min[mi]);
                max[mi] = pfmax(input_c, max[mi]);
              });
            }

            if (input_c > 0.0)
            {
              xi = 0; yi = 0; zi = 0; wi = 0;
              BoxLoopI4(i, j, k,
                        ix, iy, iz, nx, ny, nz,
                        xi, nx_xv, ny_xv, nz_xv,
                        yi, nx_yv, ny_yv, nz_yv,
                        zi, nx_zv, ny_zv, nz_zv,
                        wi, nx_w, ny_w, nz_w,
              {
                flux = (xvel_u[xi] - xvel_l[xi]) / dx
                       + (yvel_u[yi] - yvel_l[yi]) / dy
                       + (zvel_u[zi] - zvel_l[zi]) / dz;

                scaled_flux = flux * npsi[wi];

                rhs[wi] = -dt * scaled_flux * input_c;

                total_volume += dt * flux * cell_volume;
              });
            }
          }
          else if (WellDataPhysicalAction(well_data_physical)
                   == EXTRACTION_WELL)
          {
            xi = 0; yi = 0; zi = 0; wi = 0;
            BoxLoopI4(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      xi, nx_xv, ny_xv, nz_xv,
                      yi, nx_yv, ny_yv, nz_yv,
                      zi, nx_zv, ny_zv, nz_zv,
                      wi, nx_w, ny_w, nz_w,
            {
              /*   compute flux for each cell and store it   */
              flux = (xvel_u[xi] - xvel_l[xi]) / dx
                     + (yvel_u[yi] - yvel_l[yi]) / dy
                     + (zvel_u[zi] - zvel_l[zi]) / dz;

              scaled_flux = flux * npsi[wi];

              scal[wi] = dt * scaled_flux;

              total_volume += dt * flux * cell_volume;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }

      if (concentration == 0)
      {
        result_invoice = amps_NewInvoice("%d", &total_volume);
        amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
        amps_FreeInvoice(result_invoice);

        WellDataStatDeltaPhase(well_data_stat, phase) = total_volume;
        WellDataStatPhaseStat(well_data_stat, phase) += total_volume;
      }
    }

    for (well = 0; well < WellDataNumFluxWells(well_data); well++)
    {
      well_data_physical = WellDataFluxWellPhysical(well_data, well);
      cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
      interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);
      well_data_value = WellDataFluxWellIntervalValue(well_data, well, interval_number);
      well_data_stat = WellDataFluxWellStat(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      well_value = 0.0;
      if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
      {
        well_value = WellDataValuePhaseValue(well_data_value, phase);
      }
      else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
      {
        well_value = -WellDataValuePhaseValue(well_data_value, phase);
      }

      volume = WellDataPhysicalSize(well_data_physical);
      flux = well_value / volume;

      total_volume = 0.0;

      avg_x = WellDataPhysicalAveragePermeabilityX(well_data_physical);
      avg_y = WellDataPhysicalAveragePermeabilityY(well_data_physical);
      avg_z = WellDataPhysicalAveragePermeabilityZ(well_data_physical);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);
        subvector_npsi = VectorSubvector(new_porsat_inv, sg);

        px_sub = VectorSubvector(perm_x, sg);
        py_sub = VectorSubvector(perm_y, sg);
        pz_sub = VectorSubvector(perm_z, sg);

        nx_w = SubvectorNX(subvector_scal);     /*scal, rhs & npsi share nx_w */
        ny_w = SubvectorNY(subvector_scal);     /*scal, rhs & npsi share ny_w */
        nz_w = SubvectorNZ(subvector_scal);     /*scal, rhs & npsi share nz_w */

        nx_p = SubvectorNX(px_sub);
        ny_p = SubvectorNY(px_sub);
        nz_p = SubvectorNZ(px_sub);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          cell_volume = dx * dy * dz;
          area_x = dy * dz;
          area_y = dx * dz;
          area_z = dx * dy;
          area_sum = area_x + area_y + area_z;

          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);
          npsi = SubvectorElt(subvector_npsi, ix, iy, iz);

          px = SubvectorElt(px_sub, ix, iy, iz);
          py = SubvectorElt(py_sub, ix, iy, iz);
          pz = SubvectorElt(pz_sub, ix, iy, iz);

          if (WellDataPhysicalAction(well_data_physical)
              == INJECTION_WELL)
          {
            if (WellDataValueDeltaContaminantPtrs(well_data_value))
            {
              input_c =
                WellDataValueContaminantFraction(well_data_value,
                                                 index)
                * fabs(WellDataValueDeltaContaminantPtr(well_data_value,
                                                        index))
                / volume;
            }
            else
            {
              input_c = WellDataValueContaminantValue(well_data_value,
                                                      index);
            }

            if (public_xtra->enforce_minmax)
            {
              min_sub = VectorSubvector(min_concen, sg);
              max_sub = VectorSubvector(max_concen, sg);

              nx_m = SubvectorNX(min_sub);
              ny_m = SubvectorNY(min_sub);
              nz_m = SubvectorNZ(min_sub);

              min = SubvectorElt(min_sub, ix, iy, iz);
              max = SubvectorElt(max_sub, ix, iy, iz);

              mi = 0;
              BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                        mi, nx_m, ny_m, nz_m, 1, 1, 1,
              {
                min[mi] = pfmin(input_c, min[mi]);
                max[mi] = pfmax(input_c, max[mi]);
              });
            }

            if (input_c > 0.0)
            {
              wi = 0; pi = 0;
              BoxLoopI2(i, j, k,
                        ix, iy, iz, nx, ny, nz,
                        pi, nx_p, ny_p, nz_p, 1, 1, 1,
                        wi, nx_w, ny_w, nz_w, 1, 1, 1,
              {
                scaled_flux = flux * npsi[wi];

                if (WellDataPhysicalMethod(well_data_physical)
                    == FLUX_STANDARD)
                {
                  weight = 1.0;
                }
                else if (WellDataPhysicalMethod(well_data_physical)
                         == FLUX_WEIGHTED)
                {
                  weight = (px[pi] / avg_x) * (area_x / area_sum)
                           + (py[pi] / avg_y) * (area_y / area_sum)
                           + (pz[pi] / avg_z) * (area_z / area_sum);
                }
                else if (WellDataPhysicalMethod(well_data_physical)
                         == FLUX_PATTERNED)
                {
                  weight = 0.0;
                }
                else
                {
                  weight = 0.0;
                }

                scal[wi] += dt * weight * scaled_flux; // should this be assigned? scal != 0.0 leads to incorrect well node concen in injection wells
                rhs[wi] -= dt * weight * scaled_flux * input_c;

                total_volume += dt * weight * flux * cell_volume;
              });
            }
          }
          else if (WellDataPhysicalAction(well_data_physical)
                   == EXTRACTION_WELL)
          {
            wi = 0; pi = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      pi, nx_p, ny_p, nz_p, 1, 1, 1,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
            {
              scaled_flux = flux * npsi[wi];

              if (WellDataPhysicalMethod(well_data_physical)
                  == FLUX_STANDARD)
              {
                weight = 1.0;
              }
              else if (WellDataPhysicalMethod(well_data_physical)
                       == FLUX_WEIGHTED)
              {
                weight = (px[pi] / avg_x) * (area_x / area_sum)
                         + (py[pi] / avg_y) * (area_y / area_sum)
                         + (pz[pi] / avg_z) * (area_z / area_sum);
              }
              else if (WellDataPhysicalMethod(well_data_physical)
                       == FLUX_PATTERNED)
              {
                weight = 0.0;
              }

              scal[wi] -= dt * weight * scaled_flux;

              total_volume += dt * weight * flux * cell_volume;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }

      if (concentration == 0)
      {
        result_invoice = amps_NewInvoice("%d", &total_volume);
        amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
        amps_FreeInvoice(result_invoice);

        WellDataStatDeltaPhase(well_data_stat, phase) = total_volume;
        WellDataStatPhaseStat(well_data_stat, phase) += total_volume;
      }
    }
  }

  /*-----------------------------------------------------------------------
   * Compute contributions due to well terms.
   *-----------------------------------------------------------------------*/
  if (WellDataNumWells(well_data) > 0)
  {
    for (well = 0; well < WellDataNumPressWells(well_data); well++)
    {
      flopest = 5 * nx_cells * ny_cells * nz_cells;
      well_data_physical = WellDataFluxWellPhysical(well_data, well);
      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(new_concentration, sg);
        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          nx_c = SubvectorNX(subvector);
          ny_c = SubvectorNY(subvector);
          nz_c = SubvectorNZ(subvector);

          nx_w = SubvectorNX(subvector_scal);   /*scal & rhs share nx_w */
          ny_w = SubvectorNY(subvector_scal);   /*scal & rhs share ny_w */
          nz_w = SubvectorNZ(subvector_scal);   /*scal & rhs share nz_w */

          cn = SubvectorElt(subvector, ix, iy, iz);
          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          ci = 0; wi = 0;
          BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                    wi, nx_w, ny_w, nz_w, 1, 1, 1,
                    ci, nx_c, ny_c, nz_c, 1, 1, 1,
          {
            cn[ci] = (cn[ci] - rhs[wi]) / (1.0 + scal[wi]);
          });
        }
        FreeSubgrid(tmp_subgrid);      /* done with temporary subgrid */
      }
      IncFLOPCount(flopest);
    }
  }


  /*-----------------------------------------------------------------------
   * Compute changes in well stats where needed.
   *-----------------------------------------------------------------------*/

  if (WellDataNumWells(well_data) > 0)
  {
    time_cycle_data = WellDataTimeCycleData(well_data);
    for (well = 0; well < WellDataNumPressWells(well_data); well++)
    {
      well_data_physical = WellDataPressWellPhysical(well_data, well);
      cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
      interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);
      well_data_value = WellDataPressWellIntervalValue(well_data, well, interval_number);
      well_data_stat = WellDataPressWellStat(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      well_stat = 0.0;
      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(new_concentration, sg);
        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);

        nx_c = SubvectorNX(subvector);
        ny_c = SubvectorNY(subvector);
        nz_c = SubvectorNZ(subvector);

        nx_w = SubvectorNX(subvector_scal);         /* scal & rhs share nx_w */
        ny_w = SubvectorNY(subvector_scal);         /* scal & rhs share ny_w */
        nz_w = SubvectorNZ(subvector_scal);         /* scal & rhs share nz_w */

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          cell_volume = dx * dy * dz;

          cn = SubvectorElt(subvector, ix, iy, iz);
          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
          {
            wi = 0; ci = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      ci, nx_c, ny_c, nz_c, 1, 1, 1,
            {
              cell_change = -(scal[wi] * cn[ci] + rhs[wi]);
              well_stat += cell_change * cell_volume;
            });
          }
          else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
          {
            wi = 0; ci = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      ci, nx_c, ny_c, nz_c, 1, 1, 1,
            {
              cell_change = -(scal[wi] * cn[ci]);
              well_stat += cell_change * cell_volume;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }

      result_invoice = amps_NewInvoice("%d", &well_stat);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      WellDataStatDeltaContaminant(well_data_stat, index) = well_stat;
      WellDataStatContaminantStat(well_data_stat, index) += WellDataValueContaminantFraction(well_data_value, index) * well_stat;
    }

    for (well = 0; well < WellDataNumFluxWells(well_data); well++)
    {
      well_data_physical = WellDataFluxWellPhysical(well_data, well);
      cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
      interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);
      well_data_value = WellDataFluxWellIntervalValue(well_data, well, interval_number);
      well_data_stat = WellDataFluxWellStat(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      well_stat = 0.0;
      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(new_concentration, sg);
        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);

        nx_c = SubvectorNX(subvector);
        ny_c = SubvectorNY(subvector);
        nz_c = SubvectorNZ(subvector);

        nx_w = SubvectorNX(subvector_scal);         /* scal & rhs share nx_w */
        ny_w = SubvectorNY(subvector_scal);         /* scal & rhs share ny_w */
        nz_w = SubvectorNZ(subvector_scal);         /* scal & rhs share nz_w */

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          cell_volume = dx * dy * dz;

          cn = SubvectorElt(subvector, ix, iy, iz);
          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
          {
            wi = 0; ci = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      ci, nx_c, ny_c, nz_c, 1, 1, 1,
            {
              cell_change = -(scal[wi] * cn[ci] + rhs[wi]);
              well_stat += cell_change * cell_volume;
            });
          }
          else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
          {
            wi = 0; ci = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      ci, nx_c, ny_c, nz_c, 1, 1, 1,
            {
              cell_change = -(scal[wi] * cn[ci]);
              well_stat += cell_change * cell_volume;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }

      result_invoice = amps_NewInvoice("%d", &well_stat);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      WellDataStatDeltaContaminant(well_data_stat, index) = well_stat;
      WellDataStatContaminantStat(well_data_stat, index) += WellDataValueContaminantFraction(well_data_value, index) * well_stat;
    }
  }


  if (public_xtra->high_order)
  {
    handle = InitVectorUpdate(new_concentration, VectorUpdateGodunov);
    FinalizeVectorUpdate(handle);

    ForSubgridI(sg, subgrids)
    {
      subgrid = GridSubgrid(grid, sg);

      /**** Get locations for subvector data of vectors passed in ****/
      c = SubvectorData(VectorSubvector(old_concentration, sg));
      cn = SubvectorData(VectorSubvector(new_concentration, sg));
      npsi = SubvectorData(VectorSubvector(new_porsat_inv, sg));
      uedge = SubvectorData(VectorSubvector(x_velocity, sg));
      vedge = SubvectorData(VectorSubvector(y_velocity, sg));
      wedge = SubvectorData(VectorSubvector(z_velocity, sg));

      /***** Compute extents of data *****/
      dlo[0] = SubgridIX(subgrid);
      dlo[1] = SubgridIY(subgrid);
      dlo[2] = SubgridIZ(subgrid);
      dhi[0] = SubgridIX(subgrid) + (SubgridNX(subgrid) - 1);
      dhi[1] = SubgridIY(subgrid) + (SubgridNY(subgrid) - 1);
      dhi[2] = SubgridIZ(subgrid) + (SubgridNZ(subgrid) - 1);

      /***** Compute the grid spacing *****/
      hx[0] = SubgridDX(subgrid);
      hx[1] = SubgridDY(subgrid);
      hx[2] = SubgridDZ(subgrid);

      /*compute anti-diffusive fluxes */
      CALL_ADVECT_HIGHORDER(c, uedge, vedge, wedge, npsi,
                            fx, fy, fz, dlo, dhi, hx, public_xtra->dim, dt);

      if (public_xtra->transverse)
      {
        /*compute transverse flux contributions */
        CALL_ADVECT_TRANSVERSE(c, uedge, vedge, wedge,
                               dlo, dhi, hx, dt, vx, wx, uy, wy, uz, vz, fx, fy, fz);
      }

      /*multi-dimensional limiter */
      CALL_ADVECT_LIMIT(cn, fx, fy, fz, dlo, dhi, hx, public_xtra->dim, dt,
                        npsi, vx, wx, uy, wy, uz, vz);

      /*add fluxes to  new concentration, account for transient saturation*/
      CALL_ADVECT_COMPUTECONCEN(cn, fx, fy, fz, npsi,
                                dlo, dhi, hx, dt);
    }
  }

  if (public_xtra->enforce_minmax)
  {
    ForSubgridI(sg, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, sg);

      concen_sub = VectorSubvector(new_concentration, sg);
      min_sub = VectorSubvector(min_concen, sg);
      max_sub = VectorSubvector(max_concen, sg);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_c = SubvectorNX(concen_sub);
      ny_c = SubvectorNY(concen_sub);
      nz_c = SubvectorNZ(concen_sub);

      nx_m = SubvectorNX(min_sub);
      ny_m = SubvectorNY(min_sub);
      nz_m = SubvectorNZ(min_sub);

      min = SubvectorElt(min_sub, ix, iy, iz);
      max = SubvectorElt(max_sub, ix, iy, iz);

      cn = SubvectorElt(concen_sub, ix, iy, iz);

      ci = 0;
      mi = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                mi, nx_m, ny_m, nz_m, 1, 1, 1,
                ci, nx_c, ny_c, nz_c, 1, 1, 1,
      {
        cn[ci] = pfmin(max[mi], cn[ci]);
        cn[ci] = pfmax(min[mi], cn[ci]);
      });
    }
  }


  /*-----------------------------------------------------------------------
   * Informational computation and printing.
   *-----------------------------------------------------------------------*/
  if (!(GlobalsChemistryFlag))
  {
    field_sum = ComputeTotalConcen(ProblemDataGrDomain(problem_data),
                                   grid, new_concentration, new_porsat_inv);


    if (!amps_Rank(amps_CommWorld))
    {
      amps_Printf("Concentration volume for phase %1d, component %2d at time %f = %f\n", phase, concentration, time, field_sum);
    }
  }

  IncFLOPCount(VectorSize(new_concentration));

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/

  if (WellDataNumWells(well_data) > 0)
  {
    FreeVector(right_hand_side);
    FreeVector(scale);
  }

  if (public_xtra->enforce_minmax)
  {
    FreeVector(min_concen);
    FreeVector(max_concen);
  }

  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  /* well_stat / contaminant_stat accumulate advection mass changes but are
   * not yet reported; they feed the Phase 4b-2 mass-conservation check.
   * Referenced here to satisfy -Wunused-but-set-variable under -Werror. */
  (void)well_stat;
  (void)contaminant_stat;

  EndTiming(public_xtra->time_index);
}


/*--------------------------------------------------------------------------
 * GodunovInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *GodunovInitInstanceXtra(
                                   Problem *problem,
                                   Grid *   grid,
                                   double * temp_data)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra;

  SubgridArray *subgrids;

  Subgrid      *subgrid;

  int max_nx, max_ny, max_nz;
  int sg;


  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `problem'
   *-----------------------------------------------------------------------*/

  if (problem != NULL)
    (instance_xtra->problem) = problem;

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    /*** Find the maximum extents for subgrids in each direction ***/
    max_nx = 0;
    max_ny = 0;
    max_nz = 0;
    subgrids = GridSubgrids(grid);
    ForSubgridI(sg, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, sg);

      if (max_nx < SubgridNX(subgrid))
      {
        max_nx = SubgridNX(subgrid);
      }

      if (max_ny < SubgridNY(subgrid))
      {
        max_ny = SubgridNY(subgrid);
      }

      if (max_nz < SubgridNZ(subgrid))
      {
        max_nz = SubgridNZ(subgrid);
      }
    }

    (instance_xtra->max_nx) = max_nx;
    (instance_xtra->max_ny) = max_ny;
    (instance_xtra->max_nz) = max_nz;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;

    max_nx = (instance_xtra->max_nx);
    max_ny = (instance_xtra->max_ny);
    max_nz = (instance_xtra->max_nz);

    /*** set temp data pointers ***/
    (instance_xtra->fx) = temp_data;
    temp_data += (max_nx + 2 + 3) * (max_ny + 2 + 3) * (max_nz + 2 + 3);
    (instance_xtra->fy) = temp_data;
    temp_data += (max_nx + 2 + 3) * (max_ny + 2 + 3) * (max_nz + 2 + 3);
    (instance_xtra->fz) = temp_data;
    temp_data += (max_nx + 2 + 3) * (max_ny + 2 + 3) * (max_nz + 2 + 3);
    (instance_xtra->vx) = temp_data;
    temp_data += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2);
    (instance_xtra->wx) = temp_data;
    temp_data += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2);
    (instance_xtra->uy) = temp_data;
    temp_data += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2);
    (instance_xtra->wy) = temp_data;
    temp_data += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2);
    (instance_xtra->uz) = temp_data;
    temp_data += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2);
    (instance_xtra->vz) = temp_data;
    temp_data += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2);
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * GodunovFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  GodunovFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GodunovNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *GodunovNewPublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra;
  char key[IDB_MAX_KEY_LEN];
  int order;
  NameArray switch_na;
  char *switch_name;

  switch_na = NA_NewNameArray("False True");
  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->time_index) = RegisterTiming("Godunov Advection");

  sprintf(key, "Solver.AdvectOrder");
  order = GetIntDefault(key, 1);
  if (order == 2)
  {
    public_xtra->high_order = 1;
  }
  else
  {
    public_xtra->high_order = 0;
  }

  sprintf(key, "Solver.AdvectTransverse");
  switch_name = GetStringDefault(key, "False");
  public_xtra->transverse = NA_NameToIndexExitOnError(switch_na, switch_name, key);


  sprintf(key, "Solver.AdvectEnforceMinMax");
  switch_name = GetStringDefault(key, "False");
  public_xtra->enforce_minmax = NA_NameToIndexExitOnError(switch_na, switch_name, key);

  /*what is the dimensionality of the problem?*/
  public_xtra->dim[0] = (BackgroundNX(GlobalsBackground) == 1) ? 0 : 1;
  public_xtra->dim[1] = (BackgroundNY(GlobalsBackground) == 1) ? 0 : 1;
  public_xtra->dim[2] = (BackgroundNZ(GlobalsBackground) == 1) ? 0 : 1;

  NA_FreeNameArray(switch_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * GodunovFreePublicXtra
 *--------------------------------------------------------------------------*/

void GodunovFreePublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * GodunovSizeOfTempData
 *--------------------------------------------------------------------------*/

int  GodunovSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int max_nx = (instance_xtra->max_nx);
  int max_ny = (instance_xtra->max_ny);
  int max_nz = (instance_xtra->max_nz);

  int sz = 0;

  /* add local TempData size to `sz' */
  sz += (max_nx + 2 + 3) * (max_ny + 2 + 3) * (max_nz + 2 + 3) * 3;
  sz += (max_nx + 1 + 2) * (max_ny + 1 + 2) * (max_nz + 1 + 2) * 6;

  return sz;
}
