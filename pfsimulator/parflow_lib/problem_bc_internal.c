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
* Routine for setting up internal boundary conditions
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  NameArray internal_bc_names;
  int num_conditions;

  int     *type;
  void   **data;
} PublicXtra;

typedef struct {
  double xlocation;
  double ylocation;
  double zlocation;
  double value;
} Type0;                      /* basic point condition */

typedef void InstanceXtra;


/*--------------------------------------------------------------------------
 * BCInternal:
 *   Add interior boundary conditions.
 *--------------------------------------------------------------------------*/

void BCInternal(
                Problem *    problem,
                ProblemData *problem_data,
                Matrix *     A,
                Vector *     f,
                double       time)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  PFModule      *phase_density = ProblemPhaseDensity(problem);

  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;

  TimeCycleData    *time_cycle_data;

  int num_conditions = (public_xtra->num_conditions);

  Type0            *dummy0;

  Grid             *grid = VectorGrid(f);

  SubgridArray     *internal_bc_subgrids;

  Subgrid          *subgrid, *ibc_subgrid, *well_subgrid, *new_subgrid;

  Submatrix        *A_sub;
  Subvector        *f_sub;

  double           *internal_bc_conditions, *mp;

  double Z;

  double dz;
  int ix, iy, iz;
  int nx, ny, nz;
  int rx, ry, rz;
  int process;

  int i, j, k, i_sft, j_sft, k_sft;
  int grid_index, ibc_sg, well, index;
  int cycle_number, interval_number;

  double dtmp, ptmp, head, phead;

  int stencil[7][3] = { { 0, 0, 0 },
                        { -1, 0, 0 },
                        { 1, 0, 0 },
                        { 0, -1, 0 },
                        { 0, 1, 0 },
                        { 0, 0, -1 },
                        { 0, 0, 1 } };

  /***** Some constants for the routine *****/

  /* Hard-coded assumption for constant density. */
  PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, NULL, NULL, &ptmp, &dtmp, CALCFCN));
  dtmp = ProblemGravity(problem) * dtmp;

  /*--------------------------------------------------------------------
   * gridify the internal boundary locations (should be done elsewhere?)
   *--------------------------------------------------------------------*/

  if (num_conditions > 0)
  {
    internal_bc_subgrids = NewSubgridArray();
    internal_bc_conditions = ctalloc(double, num_conditions);

    for (i = 0; i < num_conditions; i++)
    {
      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = (Type0*)(public_xtra->data[i]);

          ix = IndexSpaceX((dummy0->xlocation), 0);
          iy = IndexSpaceY((dummy0->ylocation), 0);
          iz = IndexSpaceZ((dummy0->zlocation), 0);

          nx = 1;
          ny = 1;
          nz = 1;

          rx = 0;
          ry = 0;
          rz = 0;

          process = amps_Rank(amps_CommWorld);

          new_subgrid = NewSubgrid(ix, iy, iz,
                                   nx, ny, nz,
                                   rx, ry, rz,
                                   process);

          AppendSubgrid(new_subgrid, internal_bc_subgrids);

          internal_bc_conditions[i] = (dummy0->value);

          break;
        }
      }
    }
  }

  /* Note: the following two loops can be combined, so
   *  I will not merge the one above with the one below */
  /*--------------------------------------------------------------------
   * Put in the internal conditions using the subgrids computed above
   *--------------------------------------------------------------------*/

  if (num_conditions > 0)
  {
    for (grid_index = 0; grid_index < GridNumSubgrids(grid); grid_index++)
    {
      subgrid = GridSubgrid(grid, grid_index);

      A_sub = MatrixSubmatrix(A, grid_index);
      f_sub = VectorSubvector(f, grid_index);

      ForSubgridI(ibc_sg, internal_bc_subgrids)
      {
        ibc_subgrid = SubgridArraySubgrid(internal_bc_subgrids, ibc_sg);

        ix = SubgridIX(ibc_subgrid);
        iy = SubgridIY(ibc_subgrid);
        iz = SubgridIZ(ibc_subgrid);

        nx = SubgridNX(ibc_subgrid);
        ny = SubgridNY(ibc_subgrid);
        nz = SubgridNZ(ibc_subgrid);

        Z = RealSpaceZ(0, SubgridRZ(ibc_subgrid));

        BoxLoopI0(i, j, k,
                  ix, iy, iz, nx, ny, nz,
        {
          /* @RMM - Fixed bug in internal BC's (down below, only x coord was assigned, not y and z) and
           * changed notion of BC to be pressure head, not head potential to make more consistent with
           * other BC ideas and PF -- to change back, uncomment the "-dtmp*..." portion below */
          phead = internal_bc_conditions[ibc_sg];        //-  dtmp * (Z + k*dz);

          /* set column elements */
          for (index = 1; index < 7; index++)
          {
            i_sft = i - stencil[index][0];
            j_sft = j - stencil[index][1];
            k_sft = k - stencil[index][2];

            if (((i_sft >= SubgridIX(subgrid)) &&
                 (i_sft < SubgridIX(subgrid) + SubgridNX(subgrid))) &&
                ((j_sft >= SubgridIY(subgrid)) &&
                 (j_sft < SubgridIY(subgrid) + SubgridNY(subgrid))) &&
                ((k_sft >= SubgridIZ(subgrid)) &&
                 (k_sft < SubgridIZ(subgrid) + SubgridNZ(subgrid))))
            {
              mp = SubmatrixElt(A_sub, index, i_sft, j_sft, k_sft);
              SubvectorElt(f_sub, i_sft, j_sft, k_sft)[0] -= *mp * phead;
              *mp = 0.0;
            }
          }

          /* set row elements */
          if (((i >= SubgridIX(subgrid)) &&
               (i < SubgridIX(subgrid) + SubgridNX(subgrid))) &&
              ((j >= SubgridIY(subgrid)) &&
               (j < SubgridIY(subgrid) + SubgridNY(subgrid))) &&
              ((k >= SubgridIZ(subgrid)) &&
               (k < SubgridIZ(subgrid) + SubgridNZ(subgrid))))
          {
            SubmatrixElt(A_sub, 0, i, j, k)[0] = 1.0;
            for (index = 1; index < 7; index++)
            {
              SubmatrixElt(A_sub, index, i, j, k)[0] = 0.0;
            }
            SubvectorElt(f_sub, i, j, k)[0] = phead;
          }
        });
      }
    }

    FreeSubgridArray(internal_bc_subgrids);
    tfree(internal_bc_conditions);
  }

  /*--------------------------------------------------------------------
   * Put in any pressure wells from the well package
   *--------------------------------------------------------------------*/

  if (WellDataNumPressWells(well_data) > 0)
  {
    time_cycle_data = WellDataTimeCycleData(well_data);

    for (grid_index = 0; grid_index < GridNumSubgrids(grid); grid_index++)
    {
      subgrid = GridSubgrid(grid, grid_index);

      A_sub = MatrixSubmatrix(A, grid_index);
      f_sub = VectorSubvector(f, grid_index);

      for (well = 0; well < WellDataNumPressWells(well_data); well++)
      {
        well_data_physical = WellDataPressWellPhysical(well_data, well);
        cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);

        well_data_value = WellDataPressWellIntervalValue(well_data, well, interval_number);

        well_subgrid = WellDataPhysicalSubgrid(well_data_physical);
        head = WellDataValuePhaseValue(well_data_value, 0);

        ix = SubgridIX(well_subgrid);
        iy = SubgridIY(well_subgrid);
        iz = SubgridIZ(well_subgrid);

        nx = SubgridNX(well_subgrid);
        ny = SubgridNY(well_subgrid);
        nz = SubgridNZ(well_subgrid);

        dz = SubgridDZ(well_subgrid);

        Z = RealSpaceZ(0, SubgridRZ(well_subgrid));

        BoxLoopI0(i, j, k,
                  ix, iy, iz, nx, ny, nz,
        {
          phead = head - dtmp * (Z + k * dz);

          /* set column elements */
          for (index = 1; index < 7; index++)
          {
            i_sft = i - stencil[index][0];
            j_sft = j - stencil[index][1];
            k_sft = k - stencil[index][2];

            if (((i_sft >= SubgridIX(subgrid)) &&
                 (i_sft < SubgridIX(subgrid) + SubgridNX(subgrid))) &&
                ((j_sft >= SubgridIY(subgrid)) &&
                 (j_sft < SubgridIY(subgrid) + SubgridNY(subgrid))) &&
                ((k_sft >= SubgridIZ(subgrid)) &&
                 (k_sft < SubgridIZ(subgrid) + SubgridNZ(subgrid))))
            {
              mp = SubmatrixElt(A_sub, index, i_sft, j_sft, k_sft);
              SubvectorElt(f_sub, i_sft, j_sft, k_sft)[0] -= *mp * phead;
              *mp = 0.0;
            }
          }

          /* set row elements */
          if (((i >= SubgridIX(subgrid)) &&
               (i < SubgridIX(subgrid) + SubgridNX(subgrid))) &&
              ((j >= SubgridIY(subgrid)) &&
               (j < SubgridIY(subgrid) + SubgridNY(subgrid))) &&
              ((k >= SubgridIZ(subgrid)) &&
               (k < SubgridIZ(subgrid) + SubgridNZ(subgrid))))
          {
            SubmatrixElt(A_sub, 0, i, j, k)[0] = 1.0;
            for (index = 1; index < 7; index++)
            {
              SubmatrixElt(A_sub, index, i, j, k)[0] = 0.0;
            }
            SubvectorElt(f_sub, i, j, k)[0] = phead;
          }
        });
      }
    }
  }
}


/*--------------------------------------------------------------------------
 * BCInternalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *BCInternalInitInstanceXtra()
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
 * BCInternalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   BCInternalFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * BCInternalNewPublicXtra
 *--------------------------------------------------------------------------*/


PFModule  *BCInternalNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;

  int num_conditions, i;

  char *internal_bc_names;

  char key[IDB_MAX_KEY_LEN];

  public_xtra = ctalloc(PublicXtra, 1);

  internal_bc_names = GetStringDefault("InternalBC.Names", "");
  public_xtra->internal_bc_names = NA_NewNameArray(internal_bc_names);

  (public_xtra->num_conditions) = num_conditions =
    NA_Sizeof(public_xtra->internal_bc_names);

  if (num_conditions > 0)
  {
    (public_xtra->type) = ctalloc(int, num_conditions);
    (public_xtra->data) = ctalloc(void *, num_conditions);

    for (i = 0; i < num_conditions; i++)
    {
      /***************************************************************/
      /*  For the forseeable future we only have one  so just insert */
      /*   that into the type field without asking for input (which  */
      /* would mean that the input file would have to change - YUCK) */
      /***************************************************************/
      (public_xtra->type[i]) = 0;

      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = ctalloc(Type0, 1);

          sprintf(key, "InternalBC.%s.X",
                  NA_IndexToName(public_xtra->internal_bc_names, i));
          dummy0->xlocation = GetDouble(key);

          sprintf(key, "InternalBC.%s.Y",
                  NA_IndexToName(public_xtra->internal_bc_names, i));
          dummy0->ylocation = GetDouble(key);

          sprintf(key, "InternalBC.%s.Z",
                  NA_IndexToName(public_xtra->internal_bc_names, i));
          dummy0->zlocation = GetDouble(key);

          sprintf(key, "InternalBC.%s.Value",
                  NA_IndexToName(public_xtra->internal_bc_names, i));
          dummy0->value = GetDouble(key);

          (public_xtra->data[i]) = (void*)dummy0;

          break;
        }
      }
    }
  }

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * BCInternalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  BCInternalFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0         *dummy0;

  int num_conditions, i;

  if (public_xtra)
  {
#if 1
    /* sgs there is a memory problem here. this has already been freed? */
    NA_FreeNameArray(public_xtra->internal_bc_names);
#endif

    num_conditions = (public_xtra->num_conditions);

    if (num_conditions > 0)
    {
      for (i = 0; i < num_conditions; i++)
      {
        switch ((public_xtra->type[i]))
        {
          case 0:
          {
            dummy0 = (Type0*)(public_xtra->data[i]);
            tfree(dummy0);

            break;
          }
        }
      }

      tfree(public_xtra->data);
      tfree(public_xtra->type);
    }

    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * BCInternalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  BCInternalSizeOfTempData()
{
  return 0;
}
