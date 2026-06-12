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
**********************************************************************EHEADER
*
*****************************************************************************
* Collection of utility functions for reactive transport coupling
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/
#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_memory.h"
#include "alquimia/alquimia_util.h"
#endif


double InterpolateTimeCycle(double total_cycle_length, double subcycle_dt)
{
  return subcycle_dt / total_cycle_length;
}

/*--------------------------------------------------------------------------
 * TransportSaturation
 * Calculates saturation delta for transient simulations
 * Places old saturation values into vector with 2 ghost layers
 *--------------------------------------------------------------------------*/

void TransportSaturation(Vector *sat_transport_start, Vector *delta_sat, Vector *old_sat, Vector *new_sat)
{
  Grid       *grid = VectorGrid(sat_transport_start);
  Subgrid    *subgrid;

  VectorUpdateCommHandle  *handle;

  Subvector  *os_sub;
  Subvector  *st_sub;

  double     *os, *st;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_os, ny_os, nz_os;
  int nx_st, ny_st, nz_st;
  int sg, i, j, k, os_i, st_i;

  PFVDiff(new_sat, old_sat, delta_sat);

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    os_sub = VectorSubvector(old_sat, sg);
    st_sub = VectorSubvector(sat_transport_start, sg);

    nx_os = SubvectorNX(os_sub);
    ny_os = SubvectorNY(os_sub);
    nz_os = SubvectorNZ(os_sub);

    nx_st = SubvectorNX(st_sub);
    ny_st = SubvectorNY(st_sub);
    nz_st = SubvectorNZ(st_sub);

    os = SubvectorElt(os_sub, ix, iy, iz);
    st = SubvectorElt(st_sub, ix, iy, iz);

    os_i = 0;
    st_i = 0;

    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              os_i, nx_os, ny_os, nz_os, 1, 1, 1,
              st_i, nx_st, ny_st, nz_st, 1, 1, 1,
    {
      st[st_i] = os[os_i];
    });
  }

  // scatter ghosts
  handle = InitVectorUpdate(sat_transport_start, VectorUpdateAll2);
  FinalizeVectorUpdate(handle);
}


int SubgridNumCells(Grid *grid)
{
  SubgridArray  *subgrids = GridSubgrids(grid);
  Subgrid       *subgrid;
  int is;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int num_cells = 0;
  int ai = 0;

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    BoxLoopI1(i, j, k,
              ix, iy, iz, nx, ny, nz,
              ai, nx, ny, nz, 1, 1, 1,
    {
      num_cells++;
    });
  }
  return num_cells;
}


void SelectReactTransTimeStep(double max_velocity, double CFL,
                              double PF_dt, double *advect_react_dt,
                              int *num_rt_iterations)
{
  double cfl_dt;

  cfl_dt = CFL / max_velocity;

  if (PF_dt <= cfl_dt)
  {
    *advect_react_dt = PF_dt;
    *num_rt_iterations = 1;
  }
  else
  {
    *num_rt_iterations = (int)ceil(PF_dt / cfl_dt);
    *advect_react_dt = PF_dt / *num_rt_iterations;
  }
}


#ifdef HAVE_ALQUIMIA
void CutTimeStepandSolveSingleCell(AlquimiaInterface chem, AlquimiaState *chem_state, AlquimiaProperties *chem_properties, void *chem_engine, AlquimiaAuxiliaryData *chem_aux_data, AlquimiaEngineStatus *chem_status, double original_dt)
{
  double tenth_dt = 0.1 * original_dt;

  amps_Printf("Original dt of %e seconds failed. Attempting to solve 10 smaller timesteps of %e seconds.\n", original_dt, tenth_dt);

  for (int i = 0; i < 10; i++)
  {
    // reduce dt by factor of 10, subcycle geochem solve 10 times
    chem.ReactionStepOperatorSplit(&chem_engine,
                                   tenth_dt, chem_properties,
                                   chem_state,
                                   chem_aux_data,
                                   chem_status);
    if (!chem_status->converged)
    {
      amps_Printf("Geochemical engine failed to converge at reduced timestep of %e seconds.\n", tenth_dt);
      PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
    }
  }
}


void WriteChemChkpt(Grid *                 grid,
                    AlquimiaSizes *        chem_sizes,
                    AlquimiaState *        chem_state,
                    AlquimiaAuxiliaryData *chem_aux_data,
                    AlquimiaProperties *   chem_properties,
                    char *                 file_prefix,
                    char *                 file_suffix)
{
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;

  int num_chem_vars;
  int is;
  int p;
  int num_cells;
  long size;
  double tmp;
  int ai = 0;

  char file_extn[7] = "pfb";
  char filename[255];
  amps_File file;

  BeginTiming(PFBTimingIndex);

  p = amps_Rank(amps_CommWorld);

  if (p == 0)
    size = 6 * amps_SizeofDouble + 5 * amps_SizeofInt;
  else
    size = 0;


  // how big is the chemistry problem?
  num_chem_vars = chem_sizes->num_primary + // total mobile
                  chem_sizes->num_minerals * 3 + // volfx, ssa, rate constant
                  chem_sizes->num_surface_sites + // surface sites
                  chem_sizes->num_ion_exchange_sites + // ion exchange sites
                  chem_sizes->num_isotherm_species * 3 + // isotherm, Freundlich, Langmuir
                  chem_sizes->num_aqueous_kinetics + // aqueous rate constant
                  chem_sizes->num_aux_doubles + // aux data doubles
                  chem_sizes->num_aux_integers; // aux num ints

  if (chem_sizes->num_sorbed > 0)
  {
    num_chem_vars += chem_sizes->num_primary; // sorbed immobile primary
  }

  ForSubgridI(is, subgrids)
  {
    size += 9 * amps_SizeofInt;
  }

  num_cells = SubgridNumCells(grid);
  size += num_cells * num_chem_vars * amps_SizeofDouble;

  /* open file */
  sprintf(filename, "%s.%s.%s", file_prefix, file_suffix, file_extn);

  if ((file = amps_FFopen(amps_CommWorld, filename, "wb", size)) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  /* Compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }


  if (p == 0)
  {
    amps_WriteDouble(file, &BackgroundX(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundY(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundZ(GlobalsBackground), 1);

#if 0
    amps_WriteInt(file, &BackgroundNX(GlobalsBackground), 1);
    amps_WriteInt(file, &BackgroundNY(GlobalsBackground), 1);
    amps_WriteInt(file, &BackgroundNZ(GlobalsBackground), 1);
#else
    amps_WriteInt(file, &SubgridNX(GridBackground(grid)), 1);
    amps_WriteInt(file, &SubgridNY(GridBackground(grid)), 1);
    amps_WriteInt(file, &SubgridNZ(GridBackground(grid)), 1);
#endif

    amps_WriteDouble(file, &BackgroundDX(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundDY(GlobalsBackground), 1);
    amps_WriteDouble(file, &BackgroundDZ(GlobalsBackground), 1);

    amps_WriteInt(file, &num_subgrids, 1);
    amps_WriteInt(file, &num_chem_vars, 1);
  }


  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    int i, j, k;
    int ix = SubgridIX(subgrid);
    int iy = SubgridIY(subgrid);
    int iz = SubgridIZ(subgrid);

    int nx = SubgridNX(subgrid);
    int ny = SubgridNY(subgrid);
    int nz = SubgridNZ(subgrid);

    amps_WriteInt(file, &ix, 1);
    amps_WriteInt(file, &iy, 1);
    amps_WriteInt(file, &iz, 1);

    amps_WriteInt(file, &nx, 1);
    amps_WriteInt(file, &ny, 1);
    amps_WriteInt(file, &nz, 1);

    amps_WriteInt(file, &SubgridRX(subgrid), 1);
    amps_WriteInt(file, &SubgridRY(subgrid), 1);
    amps_WriteInt(file, &SubgridRZ(subgrid), 1);

    BoxLoopI1(i, j, k,
              ix, iy, iz, nx, ny, nz,
              ai, nx, ny, nz, 1, 1, 1,
    {
      int chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

      for (int idx = 0; idx < chem_sizes->num_primary; idx++)
      {
        amps_WriteDouble(file, &chem_state[chem_index].total_mobile.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_minerals; idx++)
      {
        amps_WriteDouble(file, &chem_state[chem_index].mineral_volume_fraction.data[idx], 1);
        amps_WriteDouble(file, &chem_state[chem_index].mineral_specific_surface_area.data[idx], 1);
        amps_WriteDouble(file, &chem_properties[chem_index].mineral_rate_cnst.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_surface_sites; idx++)
      {
        amps_WriteDouble(file, &chem_state[chem_index].surface_site_density.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_ion_exchange_sites; idx++)
      {
        amps_WriteDouble(file, &chem_state[chem_index].cation_exchange_capacity.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_isotherm_species; idx++)
      {
        amps_WriteDouble(file, &chem_properties[chem_index].isotherm_kd.data[idx], 1);
        amps_WriteDouble(file, &chem_properties[chem_index].freundlich_n.data[idx], 1);
        amps_WriteDouble(file, &chem_properties[chem_index].langmuir_b.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_aqueous_kinetics; idx++)
      {
        amps_WriteDouble(file, &chem_properties[chem_index].aqueous_kinetic_rate_cnst.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_aux_doubles; idx++)
      {
        amps_WriteDouble(file, &chem_aux_data[chem_index].aux_doubles.data[idx], 1);
      }


      if (chem_sizes->num_sorbed > 0)
      {
        for (int idx = 0; idx < chem_sizes->num_primary; idx++)
          amps_WriteDouble(file, &chem_state[chem_index].total_immobile.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_aux_integers; idx++)
      {
        tmp = (double)chem_aux_data[chem_index].aux_ints.data[idx];
        amps_WriteDouble(file, &tmp, 1);
      }
    });
  }
  amps_FFclose(file);
  EndTiming(PFBTimingIndex);
}



void ReadChemChkpt(Grid *                 grid,
                   AlquimiaSizes *        chem_sizes,
                   AlquimiaState *        chem_state,
                   AlquimiaAuxiliaryData *chem_aux_data,
                   AlquimiaProperties *   chem_properties,
                   char *                 filename)
{
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;

  int is;
  int p, P;
  int num_chars;

  amps_File file;
  double X, Y, Z;
  int NX, NY, NZ;
  double DX, DY, DZ;
  int num_vars;
  double tmp;
  int ai = 0;

  BeginTiming(PFBTimingIndex);

  p = amps_Rank(amps_CommWorld);
  P = amps_Size(amps_CommWorld);

  if (((num_chars = strlen(filename)) < 4) ||
      (strcmp(".pfb", &filename[num_chars - 4])))
  {
    amps_Printf("Error: %s is not in pfb format\n", filename);
    exit(1);
  }

  if ((file = amps_FFopen(amps_CommWorld, filename, "rb", 0)) == NULL)
  {
    amps_Printf("Error: can't open input file %s\n", filename);
    exit(1);
  }

  if (p == 0)
  {
    amps_ReadDouble(file, &X, 1);
    amps_ReadDouble(file, &Y, 1);
    amps_ReadDouble(file, &Z, 1);

    amps_ReadInt(file, &NX, 1);
    amps_ReadInt(file, &NY, 1);
    amps_ReadInt(file, &NZ, 1);

    amps_ReadDouble(file, &DX, 1);
    amps_ReadDouble(file, &DY, 1);
    amps_ReadDouble(file, &DZ, 1);

    amps_ReadInt(file, &P, 1);
    amps_ReadInt(file, &num_vars, 1);
  }


  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    int ix, iy, iz;
    int nx, ny, nz;
    int rx, ry, rz;
    int i, j, k;

    (void)subgrid;

    amps_ReadInt(file, &ix, 1);
    amps_ReadInt(file, &iy, 1);
    amps_ReadInt(file, &iz, 1);

    amps_ReadInt(file, &nx, 1);
    amps_ReadInt(file, &ny, 1);
    amps_ReadInt(file, &nz, 1);

    amps_ReadInt(file, &rx, 1);
    amps_ReadInt(file, &ry, 1);
    amps_ReadInt(file, &rz, 1);


    BoxLoopI1(i, j, k,
              ix, iy, iz, nx, ny, nz,
              ai, nx, ny, nz, 1, 1, 1,
    {
      int chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

      for (int idx = 0; idx < chem_sizes->num_primary; idx++)
      {
        amps_ReadDouble(file, &chem_state[chem_index].total_mobile.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_minerals; idx++)
      {
        amps_ReadDouble(file, &chem_state[chem_index].mineral_volume_fraction.data[idx], 1);
        amps_ReadDouble(file, &chem_state[chem_index].mineral_specific_surface_area.data[idx], 1);
        amps_ReadDouble(file, &chem_properties[chem_index].mineral_rate_cnst.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_surface_sites; idx++)
      {
        amps_ReadDouble(file, &chem_state[chem_index].surface_site_density.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_ion_exchange_sites; idx++)
      {
        amps_ReadDouble(file, &chem_state[chem_index].cation_exchange_capacity.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_isotherm_species; idx++)
      {
        amps_ReadDouble(file, &chem_properties[chem_index].isotherm_kd.data[idx], 1);
        amps_ReadDouble(file, &chem_properties[chem_index].freundlich_n.data[idx], 1);
        amps_ReadDouble(file, &chem_properties[chem_index].langmuir_b.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_aqueous_kinetics; idx++)
      {
        amps_ReadDouble(file, &chem_properties[chem_index].aqueous_kinetic_rate_cnst.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_aux_doubles; idx++)
      {
        amps_ReadDouble(file, &chem_aux_data[chem_index].aux_doubles.data[idx], 1);
      }


      if (chem_sizes->num_sorbed > 0)
      {
        for (int idx = 0; idx < chem_sizes->num_primary; idx++)
          amps_ReadDouble(file, &chem_state[chem_index].total_immobile.data[idx], 1);
      }


      for (int idx = 0; idx < chem_sizes->num_aux_integers; idx++)
      {
        amps_ReadDouble(file, &tmp, 1);
        chem_aux_data[chem_index].aux_ints.data[idx] = (int)tmp;
      }
    });
  }
  amps_FFclose(file);
  EndTiming(PFBTimingIndex);
}
#endif
#endif /* HAVE_ALQUIMIA */
