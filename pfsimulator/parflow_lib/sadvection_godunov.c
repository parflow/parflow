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
} PublicXtra;


typedef struct {
  /* InitInstanceXtra arguments */
  Problem  *problem;
  Grid     *grid;
  double   *temp_data;

  /* instance data */
  int max_nx;
  int max_ny;
  int max_nz;

  double   *slx;
  double   *sly;
  double   *slz;
  double   *sbot;
  double   *stop;
  double   *sbotp;
  double   *sfrt;
  double   *sbck;
  double   *sleft;
  double   *sright;
  double   *sfluxz;
  double   *dxscr;
  double   *dyscr;
  double   *dzscr;
  double   *dzfrm;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * SatGodunov
 *--------------------------------------------------------------------------*/

void     SatGodunov(
                    ProblemData *problem_data,
                    int          phase,
                    Vector *     old_saturation,
                    Vector *     new_saturation,
                    Vector *     x_velocity,
                    Vector *     y_velocity,
                    Vector *     z_velocity,
                    Vector *     z_permeability,
                    Vector *     solid_mass_factor,
                    double *     viscosity,
                    double *     density,
                    double       gravity,
                    double       time,
                    double       deltat,
                    int          order)
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Problem   *problem = (instance_xtra->problem);
  Grid      *grid = (instance_xtra->grid);

  Vector    *scale = NULL;
  Vector    *right_hand_side = NULL;

  double    *slx = (instance_xtra->slx);
  double    *sly = (instance_xtra->sly);
  double    *slz = (instance_xtra->slz);
  double    *sbot = (instance_xtra->sbot);
  double    *stop = (instance_xtra->stop);
  double    *sbotp = (instance_xtra->sbotp);
  double    *sfrt = (instance_xtra->sfrt);
  double    *sbck = (instance_xtra->sbck);
  double    *sleft = (instance_xtra->sleft);
  double    *sright = (instance_xtra->sright);
  double    *sfluxz = (instance_xtra->sfluxz);
  double    *dxscr = (instance_xtra->dxscr);
  double    *dyscr = (instance_xtra->dyscr);
  double    *dzscr = (instance_xtra->dzscr);
  double    *dzfrm = (instance_xtra->dzfrm);

  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;
  WellDataStat     *well_data_stat;

  TimeCycleData    *time_cycle_data;

  VectorUpdateCommHandle       *handle = NULL;

  SubgridArray     *subgrids;
  SubregionArray   *subregion_array;

  Subgrid          *subgrid,
                   *well_subgrid,
                   *tmp_subgrid;
  Subregion        *subregion;
  Subvector        *subvector,
                   *subvector_scal,
                   *subvector_rhs,
                   *subvector_xvel,
                   *subvector_yvel,
                   *subvector_zvel;

  ComputePkg       *compute_pkg;
  Region           *compute_reg = NULL;

  int compute_i, sr, sg, well;
  int ix, iy, iz;
  int nx, ny, nz;
  double dx, dy, dz;
  int nx_s, ny_s, nz_s,
      nx_w, ny_w, nz_w,
      nx_xv, ny_xv, nz_xv,
      nx_yv, ny_yv, nz_yv,
      nx_zv, ny_zv, nz_zv;

  int i, j, k, si, wi, xi, yi, zi;
  int index, flopest;
  int cycle_number, interval_number;

  double           *s, *sn;
  double           *rhs, *scal;
  double           *xvel_u, *xvel_l, *yvel_u, *yvel_l, *zvel_u, *zvel_l;
  double           *uedge, *vedge, *wedge, *betaedge;
  double           *phi;

  int lohi[6], dlohi[6];
  double hx[3];
  double dt;
  double cell_volume, field_sum, cell_change, well_stat;
  double well_value, input_s, volume, flux;

  amps_Invoice result_invoice;


  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  BeginTiming(public_xtra->time_index);


  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  scale = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  right_hand_side = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /*-----------------------------------------------------------------------
   * Initialize some data
   *-----------------------------------------------------------------------*/

  dt = deltat;

  subgrids = GridSubgrids(grid);

  /*-----------------------------------------------------------------------
   * Advect on all the subgrids
   *-----------------------------------------------------------------------*/

  flopest = 0;

  compute_pkg = GridComputePkg(VectorGrid(old_saturation),
                               VectorUpdateGodunov);

  for (compute_i = 0; compute_i < 2; compute_i++)
  {
    switch (compute_i)
    {
      case 0:
        handle = InitVectorUpdate(old_saturation, VectorUpdateGodunov);
        compute_reg = ComputePkgIndRegion(compute_pkg);
        break;

      case 1:
        FinalizeVectorUpdate(handle);
        compute_reg = ComputePkgDepRegion(compute_pkg);
        break;
    }

    ForSubregionArrayI(sr, compute_reg)
    {
      subregion_array = RegionSubregionArray(compute_reg, sr);
      subgrid = SubgridArraySubgrid(subgrids, sr);

      /**** Get locations for subvector data of vectors passed in ****/
      s = SubvectorData(VectorSubvector(old_saturation, sr));
      sn = SubvectorData(VectorSubvector(new_saturation, sr));

      uedge = SubvectorData(VectorSubvector(x_velocity, sr));
      vedge = SubvectorData(VectorSubvector(y_velocity, sr));
      wedge = SubvectorData(VectorSubvector(z_velocity, sr));
      betaedge = SubvectorData(VectorSubvector(z_permeability, sr));

      phi = SubvectorData(VectorSubvector(solid_mass_factor, sr));

      /***** Compute extents of data *****/
      dlohi[0] = SubgridIX(subgrid);
      dlohi[1] = SubgridIY(subgrid);
      dlohi[2] = SubgridIZ(subgrid);
      dlohi[3] = SubgridIX(subgrid) + (SubgridNX(subgrid) - 1);
      dlohi[4] = SubgridIY(subgrid) + (SubgridNY(subgrid) - 1);
      dlohi[5] = SubgridIZ(subgrid) + (SubgridNZ(subgrid) - 1);

      /***** Compute the grid spacing *****/
      hx[0] = SubgridDX(subgrid);
      hx[1] = SubgridDY(subgrid);
      hx[2] = SubgridDZ(subgrid);

      ForSubregionI(sg, subregion_array)
      {
        subregion = SubregionArraySubregion(subregion_array, sg);

        /**** Compute the extents of computational subregion *****/
        lohi[0] = SubregionIX(subregion);
        lohi[1] = SubregionIY(subregion);
        lohi[2] = SubregionIZ(subregion);
        lohi[3] = SubregionIX(subregion) + (SubregionNX(subregion) - 1);
        lohi[4] = SubregionIY(subregion) + (SubregionNY(subregion) - 1);
        lohi[5] = SubregionIZ(subregion) + (SubregionNZ(subregion) - 1);


        /***** Make the call to the Godunov advection routine *****/
        CALL_SADVECT(s, sn,
                     uedge, vedge, wedge, betaedge, phi,
                     viscosity, density, gravity,
                     slx, sly, slz,
                     lohi, dlohi, hx, dt,
                     sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,
                     dxscr, dyscr, dzscr, dzfrm);
      }
    }
  }

  IncFLOPCount(flopest);

  /*-----------------------------------------------------------------------
   * Set up source terms, right hand side and scaling term
   *-----------------------------------------------------------------------*/

  index = phase;

  InitVectorAll(scale, 0.0);
  InitVectorAll(right_hand_side, 0.0);

  if (WellDataNumWells(well_data) > 0)
  {
    time_cycle_data = WellDataTimeCycleData(well_data);
    for (well = 0; well < WellDataNumPressWells(well_data); well++)
    {
      well_data_physical = WellDataPressWellPhysical(well_data, well);
      cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
      interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);
      well_data_value = WellDataPressWellIntervalValue(well_data, well, interval_number);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);
        subvector_xvel = VectorSubvector(x_velocity, sg);
        subvector_yvel = VectorSubvector(y_velocity, sg);
        subvector_zvel = VectorSubvector(z_velocity, sg);

        nx_w = SubvectorNX(subvector_scal);       /* scal & rhs share nx_w */
        ny_w = SubvectorNY(subvector_scal);       /* scal & rhs share ny_w */
        nz_w = SubvectorNZ(subvector_scal);       /* scal & rhs share nz_w */

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

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          xvel_l = SubvectorElt(subvector_xvel, ix, iy, iz);
          xvel_u = SubvectorElt(subvector_xvel, ix + 1, iy, iz);

          yvel_l = SubvectorElt(subvector_yvel, ix, iy, iz);
          yvel_u = SubvectorElt(subvector_yvel, ix, iy + 1, iz);

          zvel_l = SubvectorElt(subvector_zvel, ix, iy, iz);
          zvel_u = SubvectorElt(subvector_zvel, ix, iy, iz + 1);

          if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
          {
            if (WellDataValueDeltaSaturationPtrs(well_data_value))
            {
              input_s = WellDataValueDeltaSaturationPtr(well_data_value, index);
            }
            else
            {
              input_s = WellDataValueSaturationValue(well_data_value, index);
            }

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
              scal[wi] = flux;
              rhs[wi] = -flux * input_s;
            });
          }
          else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
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
              scal[wi] = flux;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }
    }

    for (well = 0; well < WellDataNumFluxWells(well_data); well++)
    {
      well_data_physical = WellDataFluxWellPhysical(well_data, well);
      cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
      interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);
      well_data_value = WellDataFluxWellIntervalValue(well_data, well, interval_number);

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

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);

        nx_w = SubvectorNX(subvector_scal);       /* scal & rhs share nx_w */
        ny_w = SubvectorNY(subvector_scal);       /* scal & rhs share ny_w */
        nz_w = SubvectorNZ(subvector_scal);       /* scal & rhs share nz_w */

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
          {
            if (WellDataValueDeltaSaturationPtrs(well_data_value))
            {
              input_s = WellDataValueDeltaSaturationPtr(well_data_value, index);
            }
            else
            {
              input_s = WellDataValueSaturationValue(well_data_value, index);
            }
            input_s = WellDataValueSaturationValue(well_data_value, index);

            wi = 0;
            BoxLoopI1(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
            {
              scal[wi] += flux;
              rhs[wi] -= flux * input_s;
            });
          }
          else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
          {
            wi = 0;
            BoxLoopI1(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
            {
              scal[wi] -= flux;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }
    }
  }


  /*-----------------------------------------------------------------------
   * Compute well effects
   *-----------------------------------------------------------------------*/

  ForSubgridI(sg, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, sg);

    subvector = VectorSubvector(new_saturation, sg);
    subvector_scal = VectorSubvector(scale, sg);
    subvector_rhs = VectorSubvector(right_hand_side, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx_s = SubvectorNX(subvector);
    ny_s = SubvectorNY(subvector);
    nz_s = SubvectorNZ(subvector);

    nx_w = SubvectorNX(subvector_scal);       /* scal & rhs share nx_w */
    ny_w = SubvectorNY(subvector_scal);       /* scal & rhs share ny_w */
    nz_w = SubvectorNZ(subvector_scal);       /* scal & rhs share nz_w */

    sn = SubvectorElt(subvector, ix, iy, iz);
    rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
    scal = SubvectorElt(subvector_scal, ix, iy, iz);

    si = 0; wi = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              wi, nx_w, ny_w, nz_w, 1, 1, 1,
              si, nx_s, ny_s, nz_s, 1, 1, 1,
    {
      sn[si] = (sn[si] - dt * rhs[wi]) / (1 + dt * scal[wi]);
    });
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
      well_data_stat = WellDataPressWellStat(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      well_stat = 0.0;
      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(new_saturation, sg);
        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);

        nx_s = SubvectorNX(subvector);
        ny_s = SubvectorNY(subvector);
        nz_s = SubvectorNZ(subvector);

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

          sn = SubvectorElt(subvector, ix, iy, iz);
          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
          {
            wi = 0; si = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      si, nx_s, ny_s, nz_s, 1, 1, 1,
            {
              cell_change = -dt * (scal[wi] * sn[si] + rhs[wi]);
              well_stat += cell_change * cell_volume;
            });
          }
          else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
          {
            wi = 0; si = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      si, nx_s, ny_s, nz_s, 1, 1, 1,
            {
              cell_change = -dt * (scal[wi] * sn[si]);
              well_stat += cell_change * cell_volume;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }

      result_invoice = amps_NewInvoice("%d", &well_stat);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      WellDataStatDeltaSaturation(well_data_stat, index) = well_stat;
      WellDataStatSaturationStat(well_data_stat, index) += well_stat;
    }

    for (well = 0; well < WellDataNumFluxWells(well_data); well++)
    {
      well_data_physical = WellDataFluxWellPhysical(well_data, well);
      well_data_stat = WellDataFluxWellStat(well_data, well);

      well_subgrid = WellDataPhysicalSubgrid(well_data_physical);

      well_stat = 0.0;
      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);

        subvector = VectorSubvector(new_saturation, sg);
        subvector_scal = VectorSubvector(scale, sg);
        subvector_rhs = VectorSubvector(right_hand_side, sg);

        nx_s = SubvectorNX(subvector);
        ny_s = SubvectorNY(subvector);
        nz_s = SubvectorNZ(subvector);

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

          sn = SubvectorElt(subvector, ix, iy, iz);
          rhs = SubvectorElt(subvector_rhs, ix, iy, iz);
          scal = SubvectorElt(subvector_scal, ix, iy, iz);

          if (WellDataPhysicalAction(well_data_physical) == INJECTION_WELL)
          {
            wi = 0; si = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      si, nx_s, ny_s, nz_s, 1, 1, 1,
            {
              cell_change = -dt * (scal[wi] * sn[si] + rhs[wi]);
              well_stat += cell_change * cell_volume;
            });
          }
          else if (WellDataPhysicalAction(well_data_physical) == EXTRACTION_WELL)
          {
            wi = 0; si = 0;
            BoxLoopI2(i, j, k,
                      ix, iy, iz, nx, ny, nz,
                      wi, nx_w, ny_w, nz_w, 1, 1, 1,
                      si, nx_s, ny_s, nz_s, 1, 1, 1,
            {
              cell_change = -dt * (scal[wi] * sn[si]);
              well_stat += cell_change * cell_volume;
            });
          }
          FreeSubgrid(tmp_subgrid);        /* done with temporary subgrid */
        }
      }

      result_invoice = amps_NewInvoice("%d", &well_stat);
      amps_AllReduce(amps_CommWorld, result_invoice, amps_Add);
      amps_FreeInvoice(result_invoice);

      WellDataStatDeltaSaturation(well_data_stat, index) = well_stat;
      WellDataStatSaturationStat(well_data_stat, index) += well_stat;
    }
  }

#if 1
  /*-----------------------------------------------------------------------
   * Informational computation and printing.
   *-----------------------------------------------------------------------*/

  field_sum = ComputeTotalConcen(ProblemDataGrDomain(problem_data),
                                 grid, new_saturation);

  if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Saturation volume for phase %1d at time %f = %f\n", phase, time, field_sum);
  }

  IncFLOPCount(VectorSize(new_saturation));
#endif

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(right_hand_side);
  FreeVector(scale);

  /*-----------------------------------------------------------------------
   * End timing
   *-----------------------------------------------------------------------*/

  EndTiming(public_xtra->time_index);
}


/*--------------------------------------------------------------------------
 * SatGodunovInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SatGodunovInitInstanceXtra(
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
    (instance_xtra->slx) = temp_data;
    temp_data += (max_nx + 2 + 2) * (max_ny + 2 + 2);
    (instance_xtra->sly) = temp_data;
    temp_data += (max_nx + 2 + 2) * (max_ny + 2 + 2);
    (instance_xtra->slz) = temp_data;
    temp_data += (max_nx + 2 + 2) * (max_ny + 2 + 2) * 3;
    (instance_xtra->sbot) = temp_data;
    temp_data += (max_nx + 3 + 3) * (max_ny + 3 + 3);
    (instance_xtra->stop) = temp_data;
    temp_data += (max_nx + 3 + 3) * (max_ny + 3 + 3);
    (instance_xtra->sbotp) = temp_data;
    temp_data += (max_nx + 3 + 3) * (max_ny + 3 + 3);
    (instance_xtra->sfrt) = temp_data;
    temp_data += (max_nx + 3 + 3) * (max_ny + 3 + 3);
    (instance_xtra->sbck) = temp_data;
    temp_data += (max_nx + 3 + 3) * (max_ny + 3 + 3);
    (instance_xtra->sleft) = temp_data;
    temp_data += (max_nx + 3 + 3);
    (instance_xtra->sright) = temp_data;
    temp_data += (max_nx + 3 + 3);
    (instance_xtra->sfluxz) = temp_data;
    temp_data += (max_nx + 3 + 3);
    (instance_xtra->dxscr) = temp_data;
    temp_data += (max_nx + 3 + 3) * 4;
    (instance_xtra->dyscr) = temp_data;
    temp_data += (max_ny + 3 + 3) * 4;
    (instance_xtra->dzscr) = temp_data;
    temp_data += (max_nx + 3 + 3) * 3;
    (instance_xtra->dzfrm) = temp_data;
    temp_data += (max_nx + 3 + 3) * 3;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SatGodunovFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SatGodunovFreeInstanceXtra()
{
  PFModule     *this_module = ThisPFModule;
  InstanceXtra *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SatGodunovNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *SatGodunovNewPublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra;


  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->time_index) = RegisterTiming("Godunov Saturation");

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * SatGodunovFreePublicXtra
 *--------------------------------------------------------------------------*/

void SatGodunovFreePublicXtra()
{
  PFModule     *this_module = ThisPFModule;
  PublicXtra   *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * SatGodunovSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SatGodunovSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int max_nx = (instance_xtra->max_nx);
  int max_ny = (instance_xtra->max_ny);

  int sz = 0;

  /* add local TempData size to `sz' */
  sz += (max_nx + 2 + 2) * (max_ny + 2 + 2);
  sz += (max_nx + 2 + 2) * (max_ny + 2 + 2);
  sz += (max_nx + 2 + 2) * (max_ny + 2 + 2) * 3;

  sz += (max_nx + 3 + 3) * (max_ny + 3 + 3);
  sz += (max_nx + 3 + 3) * (max_ny + 3 + 3);
  sz += (max_nx + 3 + 3) * (max_ny + 3 + 3);
  sz += (max_nx + 3 + 3) * (max_ny + 3 + 3);
  sz += (max_nx + 3 + 3) * (max_ny + 3 + 3);
  sz += (max_nx + 3 + 3);
  sz += (max_nx + 3 + 3);
  sz += (max_nx + 3 + 3);
  sz += (max_nx + 3 + 3) * 4;
  sz += (max_ny + 3 + 3) * 4;
  sz += (max_nx + 3 + 3) * 3;
  sz += (max_nx + 3 + 3) * 3;

  return sz;
}
