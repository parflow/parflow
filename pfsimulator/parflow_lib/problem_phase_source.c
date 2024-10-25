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
#include <float.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;

  int    *type;
  void  **data;
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  NameArray regions;
  int num_regions;


  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
  int function_type;
} Type1;                       /* Known forcing term on entire domain */


/*--------------------------------------------------------------------------
 * PhaseSource
 *--------------------------------------------------------------------------*/

void         PhaseSource(
                         Vector *     phase_source,
                         int          phase,
                         Problem *    problem,
                         ProblemData *problem_data,
                         double       time)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;

  ReservoirData         *reservoir_data = ProblemDataReservoirData(problem_data);
  ReservoirDataPhysical *reservoir_data_physical;

  TimeCycleData    *time_cycle_data;

  Vector           *perm_x = ProblemDataPermeabilityX(problem_data);
  Vector           *perm_y = ProblemDataPermeabilityY(problem_data);
  Vector           *perm_z = ProblemDataPermeabilityZ(problem_data);

  Grid             *grid = VectorGrid(phase_source);

  Type0            *dummy0;
  Type1            *dummy1;

  SubgridArray     *subgrids = GridSubgrids(grid);

  Subgrid          *subgrid, *well_subgrid, *tmp_subgrid, *reservoir_release_subgrid;
  Subvector        *px_sub, *py_sub, *pz_sub, *ps_sub;

  double           *data, *px, *py, *pz;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;

  int nx_p, ny_p, nz_p;
  int nx_ps, ny_ps, nz_ps;

  int is, i, j, k;

  /* Locals associated with wells and reservoirs*/
  int well;
  int cycle_number, interval_number;
  double volume, flux, well_value, reservoir_value;

// SGS FIXME why is this needed?
#undef max
  double area_x, area_y, area_z, area_sum;
  double avg_x, avg_y, avg_z;
  double dx, dy, dz;

  /*-----------------------------------------------------------------------
   * Put in any user defined sources for this phase
   *-----------------------------------------------------------------------*/

  InitVectorAll(phase_source, 0.0);

  switch ((public_xtra->type[phase]))
  {
    case 0:
    {
      int num_regions;
      int     *region_indices;
      double  *values;

      GrGeomSolid  *gr_solid;
      double value;
      int ir;

      dummy0 = (Type0*)(public_xtra->data[phase]);

      num_regions = (dummy0->num_regions);
      region_indices = (dummy0->region_indices);
      values = (dummy0->values);

      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
        value = values[ir];

        ForSubgridI(is, subgrids)
        {
          subgrid = SubgridArraySubgrid(subgrids, is);
          ps_sub = VectorSubvector(phase_source, is);

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
            int ips = SubvectorEltIndex(ps_sub, i, j, k);

            data[ips] = value;
          });
        }
      }

      break;
    }  /* End case 0 */

    case 1:
    {
      GrGeomSolid  *gr_domain;
      int function_type;

      dummy1 = (Type1*)(public_xtra->data[phase]);

      function_type = (dummy1->function_type);

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);
        ps_sub = VectorSubvector(phase_source, is);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        /* RDF: assume resolution is the same in all 3 directions */
        r = SubgridRX(subgrid);

        data = SubvectorData(ps_sub);

        switch (function_type)
        {
          case 1: /* p= x */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              /* nonlinear case -div(p grad p) = f */
              data[ips] = -1.0;
            });
            break;
          } /* End case 1 */

          case 2: /* p= x+y+z */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              /* nonlinear case -div(p grad p) = f */
              data[ips] = -3.0;
            });
            break;
          } /* End case 2 */

          case 3: /* p= x^3y^2 + sinxy + 1 */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              double x = RealSpaceX(i, SubgridRX(subgrid));
              double y = RealSpaceY(j, SubgridRY(subgrid));
              /* nonlinear case -div(p grad p) = f */
              data[ips] = -pow((3 * x * x * y * y + y * cos(x * y)), 2) - pow((2 * x * x * x * y + x * cos(x * y)), 2) - (x * x * x * y * y + sin(x * y) + 1) * (6 * x * y * y + 2 * x * x * x - (x * x + y * y) * sin(x * y));
            });
            break;
          } /* End case 3 */

          case 4: /* f for p = x^3y^4 + x^2 + sinxy cosy + 1 */
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              double x = RealSpaceX(i, SubgridRX(subgrid));
              double y = RealSpaceY(j, SubgridRY(subgrid));

              data[ips] = -pow(3 * x * x * pow(y, 4) + 2 * x + y * cos(x * y) * cos(y), 2) - pow(4 * x * x * x * y * y * y + x * cos(x * y) * cos(y) - sin(x * y) * sin(y), 2) - (x * x * x * pow(y, 4) + x * x + sin(x * y) * cos(y) + 1) * (6 * x * pow(y, 4) + 2 - (x * x + y * y + 1) * sin(x * y) * cos(y) + 12 * x * x * x * y * y - 2 * x * cos(x * y) * sin(y));
            });
            break;
          } /* End case 4 */

          case 5: /* f = xyz-y^2z^2t^2-x^2z^2t^2-x^2y^2t^2 (p=xyzt+1)*/
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              double x = RealSpaceX(i, SubgridRX(subgrid));
              double y = RealSpaceY(j, SubgridRY(subgrid));
              double z = RealSpaceZ(k, SubgridRZ(subgrid));

              data[ips] = x * y * z - time * time * (y * y * z * z + x * x * z * z + x * x * y * y);
            });
            break;
          } /* End case 5 */

          case 6: /* f = xyz-y^2z^2t^2-2x^2z^2t^2-3x^2y^2t^2 (p=xyzt+1,
                   *                                          K=(1; 2; 3) )*/
          {
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
            {
              int ips = SubvectorEltIndex(ps_sub, i, j, k);
              double x = RealSpaceX(i, SubgridRX(subgrid));
              double y = RealSpaceY(j, SubgridRY(subgrid));
              double z = RealSpaceZ(k, SubgridRZ(subgrid));

              data[ips] = x * y * z
                          - time * time * (y * y * z * z + x * x * z * z * 2.0 + x * x * y * y * 3.0);
            });
            break;
          } /* End case 6 */
        }   /* End switch statement on function_types */
      }     /* End subgrid loop */

      break;
    }  /* End case 1 for input types */
  }    /* End switch statement for input types */

  /*-----------------------------------------------------------------------
   * Put in any flux wells from the well package
   *-----------------------------------------------------------------------*/

  if (WellDataNumFluxWells(well_data) > 0)
  {
    time_cycle_data = WellDataTimeCycleData(well_data);

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
      else if (WellDataPhysicalAction(well_data_physical)
               == EXTRACTION_WELL)

      {
        well_value = -WellDataValuePhaseValue(well_data_value, phase);
      }

      volume = WellDataPhysicalSize(well_data_physical);
      flux = well_value / volume;

      avg_x = WellDataPhysicalAveragePermeabilityX(well_data_physical);
      avg_y = WellDataPhysicalAveragePermeabilityY(well_data_physical);
      avg_z = WellDataPhysicalAveragePermeabilityZ(well_data_physical);

      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);

        px_sub = VectorSubvector(perm_x, is);
        py_sub = VectorSubvector(perm_y, is);
        pz_sub = VectorSubvector(perm_z, is);

        ps_sub = VectorSubvector(phase_source, is);

        nx_p = SubvectorNX(ps_sub);
        ny_p = SubvectorNY(ps_sub);
        nz_p = SubvectorNZ(ps_sub);

        nx_ps = SubvectorNX(ps_sub);
        ny_ps = SubvectorNY(ps_sub);
        nz_ps = SubvectorNZ(ps_sub);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          /*  If an intersection;  loop over it, and insert value  */
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          dx = SubgridDX(tmp_subgrid);
          dy = SubgridDY(tmp_subgrid);
          dz = SubgridDZ(tmp_subgrid);

          area_x = dy * dz;
          area_y = dx * dz;
          area_z = dx * dy;
          area_sum = area_x + area_y + area_z;

          px = SubvectorElt(px_sub, ix, iy, iz);
          py = SubvectorElt(py_sub, ix, iy, iz);
          pz = SubvectorElt(pz_sub, ix, iy, iz);

          data = SubvectorElt(ps_sub, ix, iy, iz);

          int ip = 0;
          int ips = 0;

          if (WellDataPhysicalMethod(well_data_physical)
              == FLUX_WEIGHTED)
          {
            BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                      ip, nx_p, ny_p, nz_p, 1, 1, 1,
                      ips, nx_ps, ny_ps, nz_ps, 1, 1, 1,
            {
              double weight = (px[ip] / avg_x) * (area_x / area_sum)
                              + (py[ip] / avg_y) * (area_y / area_sum)
                              + (pz[ip] / avg_z) * (area_z / area_sum);
              data[ips] += weight * flux;
            });
          }
          else
          {
            double weight = -FLT_MAX;
            if (WellDataPhysicalMethod(well_data_physical)
                == FLUX_STANDARD)
              weight = 1.0;
            else if (WellDataPhysicalMethod(well_data_physical)
                     == FLUX_PATTERNED)
              weight = 0.0;
            BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                      ip, nx_p, ny_p, nz_p, 1, 1, 1,
                      ips, nx_ps, ny_ps, nz_ps, 1, 1, 1,
            {
              data[ips] += weight * flux;
            });
          }
          /* done with this temporary subgrid */
          FreeSubgrid(tmp_subgrid);
        }
      }
    }
  }  /* End well data */

  if (ReservoirDataNumReservoirs(reservoir_data) > 0)
  {
    for (int reservoir = 0; reservoir < ReservoirDataNumReservoirs(reservoir_data); reservoir++)
    {
      reservoir_data_physical = ReservoirDataReservoirPhysical(reservoir_data, reservoir);
      reservoir_release_subgrid = ReservoirDataPhysicalReleaseSubgrid(reservoir_data_physical);
      reservoir_value = ReservoirDataPhysicalReleaseRate(reservoir_data_physical);
      volume = ReservoirDataPhysicalSize(reservoir_data_physical);
      flux = reservoir_value / (volume);
      //If we are overfull need to release the rest of the flux
      if (reservoir_data_physical->storage > reservoir_data_physical->max_storage)
      {
        flux = (reservoir_data_physical->storage - reservoir_data_physical->max_storage) / (volume);
      }
      ForSubgridI(is, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, is);

        px_sub = VectorSubvector(perm_x, is);
        py_sub = VectorSubvector(perm_y, is);
        pz_sub = VectorSubvector(perm_z, is);

        ps_sub = VectorSubvector(phase_source, is);

        nx_p = SubvectorNX(ps_sub);
        ny_p = SubvectorNY(ps_sub);
        nz_p = SubvectorNZ(ps_sub);

        nx_ps = SubvectorNX(ps_sub);
        ny_ps = SubvectorNY(ps_sub);
        nz_ps = SubvectorNZ(ps_sub);

        if (reservoir_data_physical->storage > reservoir_data_physical->min_release_storage)
        {
          reservoir_data_physical = ReservoirDataReservoirPhysical(reservoir_data, reservoir);
          /*  Get the intersection of the reservoir with the subgrid  */
          if ((tmp_subgrid = IntersectSubgrids(subgrid, reservoir_release_subgrid)))
          {
            /*  If an intersection;  loop over it, and insert value  */
            ix = SubgridIX(tmp_subgrid);
            iy = SubgridIY(tmp_subgrid);
            iz = SubgridIZ(tmp_subgrid);

            nx = SubgridNX(tmp_subgrid);
            ny = SubgridNY(tmp_subgrid);
            nz = SubgridNZ(tmp_subgrid);

            dx = SubgridDX(tmp_subgrid);
            dy = SubgridDY(tmp_subgrid);
            dz = SubgridDZ(tmp_subgrid);

            px = SubvectorElt(px_sub, ix, iy, iz);
            py = SubvectorElt(py_sub, ix, iy, iz);
            pz = SubvectorElt(pz_sub, ix, iy, iz);

            data = SubvectorElt(ps_sub, ix, iy, iz);

            int ips = 0;
            double weight = 1.0;

            BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                      ips, nx_ps, ny_ps, nz_ps, 1, 1, 1,
            {
              data[ips] += weight * flux;
              ReservoirDataPhysicalReleaseAmountInSolver(reservoir_data_physical) = flux * volume;
            });
          }
        }
      }
    }
  }
}




/*--------------------------------------------------------------------------
 * PhaseSourceInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseSourceInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;


#if 0
  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
#endif
  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PhaseSourceFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PhaseSourceFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * PhaseSourceNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseSourceNewPublicXtra(
                                    int num_phases)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;
  Type1         *dummy1;

  int num_regions, ir;

  int i;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;

  NameArray type_na;
  NameArray function_type_na;

  type_na = NA_NewNameArray("Constant PredefinedFunction");

  function_type_na = NA_NewNameArray("dum0 X XPlusYPlusZ X3Y2PlusSinXYPlus1 \
                                       X3Y4PlusX2PlusSinXYCosYPlus1 \
                                       XYZTPlus1 XYZTPlus1PermTensor");

  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->num_phases) = num_phases;

  (public_xtra->type) = ctalloc(int, num_phases);
  (public_xtra->data) = ctalloc(void *, num_phases);

  for (i = 0; i < num_phases; i++)
  {
    sprintf(key, "PhaseSources.%s.Type",
            NA_IndexToName(GlobalsPhaseNames, i));
    switch_name = GetString(key);

    public_xtra->type[i] = NA_NameToIndexExitOnError(type_na, switch_name, key);

    switch ((public_xtra->type[i]))
    {
      case 0:
      {
        dummy0 = ctalloc(Type0, 1);

        sprintf(key, "PhaseSources.%s.GeomNames",
                NA_IndexToName(GlobalsPhaseNames, i));
        switch_name = GetString(key);

        dummy0->regions = NA_NewNameArray(switch_name);

        num_regions = (dummy0->num_regions) = NA_Sizeof(dummy0->regions);

        (dummy0->region_indices) = ctalloc(int, num_regions);
        (dummy0->values) = ctalloc(double, num_regions);

        for (ir = 0; ir < num_regions; ir++)
        {
          dummy0->region_indices[ir] =
            NA_NameToIndex(GlobalsGeomNames,
                           NA_IndexToName(dummy0->regions, ir));

          sprintf(key, "PhaseSources.%s.Geom.%s.Value",
                  NA_IndexToName(GlobalsPhaseNames, i),
                  NA_IndexToName(dummy0->regions, ir));
          dummy0->values[ir] = GetDouble(key);
        }

        (public_xtra->data[i]) = (void*)dummy0;

        break;
      }      /* End case 0 */

      case 1:
      {
        dummy1 = ctalloc(Type1, 1);

        sprintf(key, "PhaseSources.%s.PredefinedFunction",
                NA_IndexToName(GlobalsPhaseNames, i));
        switch_name = GetString(key);

        dummy1->function_type =
          NA_NameToIndexExitOnError(function_type_na, switch_name, key);

        (public_xtra->data[i]) = (void*)dummy1;

        break;
      }      /* End case 1 */

      default:
      {
        InputError("Invalid switch value <%s> for key <%s>", switch_name, key);
      }
    }     /* End case statement */
  }


  NA_FreeNameArray(type_na);
  NA_FreeNameArray(function_type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseSourceFreePublicXtra
 *-------------------------------------------------------------------------*/

void  PhaseSourceFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0       *dummy0;
  Type1       *dummy1;

  int i;


  if (public_xtra)
  {
    for (i = 0; i < (public_xtra->num_phases); i++)
    {
      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = (Type0*)(public_xtra->data[i]);

          NA_FreeNameArray(dummy0->regions);

          tfree(dummy0->region_indices);
          tfree(dummy0->values);
          tfree(dummy0);
          break;
        }

        case 1:
        {
          dummy1 = (Type1*)(public_xtra->data[i]);

          tfree(dummy1);
          break;
        }
      }
    }

    tfree(public_xtra->data);
    tfree(public_xtra->type);

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * PhaseSourceSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PhaseSourceSizeOfTempData()
{
  return 0;
}
