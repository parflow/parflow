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
* AllocatePFChemData allocates PF Vectors for the geochemical variables.
* Right now PF Vectors are allocated for every variable in the problem,
* even if the user sin't printing them. This may? get expensive for large problems.
*
* AllocateChemCells allocates alquimia state, proerties, auxdata, and
* auxoutputdata for every cell
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_memory.h"

void AllocatePFChemData(AlquimiaDataPF *alquimia_data, Grid *grid)
{
  int i;

  //pH
  alquimia_data->pH = NewVectorType(grid, 1, 0, vector_cell_centered);
  InitVectorAll(alquimia_data->pH, 0.0);

  // num_primary - primary activity and free ion concen
  if (alquimia_data->chem_sizes.num_primary > 0)
  {
    alquimia_data->primary_free_ion_concentrationPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_primary);
    alquimia_data->primary_activity_coeffPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_primary);

    for (i = 0; i < alquimia_data->chem_sizes.num_primary; i++)
    {
      alquimia_data->primary_free_ion_concentrationPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->primary_free_ion_concentrationPF[i], 0.0);

      alquimia_data->primary_activity_coeffPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->primary_activity_coeffPF[i], 0.0);
    }
  }

  // num_sorbed - total_immobile - allocate for num_primary
  if (alquimia_data->chem_sizes.num_sorbed > 0)
  {
    alquimia_data->total_immobilePF = ctalloc(Vector *, alquimia_data->chem_sizes.num_primary);

    for (i = 0; i < alquimia_data->chem_sizes.num_primary; i++)
    {
      alquimia_data->total_immobilePF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->total_immobilePF[i], 0.0);
    }
  }

  // num_minerals - volfx, SSA, SI, rate
  if (alquimia_data->chem_sizes.num_minerals > 0)
  {
    alquimia_data->mineral_volume_fractionsPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_minerals);
    alquimia_data->mineral_specific_surfacePF = ctalloc(Vector *, alquimia_data->chem_sizes.num_minerals);
    alquimia_data->mineral_saturation_indexPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_minerals);
    alquimia_data->mineral_reaction_ratePF = ctalloc(Vector *, alquimia_data->chem_sizes.num_minerals);

    for (i = 0; i < alquimia_data->chem_sizes.num_minerals; i++)
    {
      alquimia_data->mineral_volume_fractionsPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->mineral_volume_fractionsPF[i], 0.0);

      alquimia_data->mineral_specific_surfacePF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->mineral_specific_surfacePF[i], 0.0);

      alquimia_data->mineral_saturation_indexPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->mineral_saturation_indexPF[i], 0.0);

      alquimia_data->mineral_reaction_ratePF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->mineral_reaction_ratePF[i], 0.0);
    }
  }

  // num_aqueous_complexes - secondary activity and free ion
  if (alquimia_data->chem_sizes.num_aqueous_complexes > 0)
  {
    alquimia_data->secondary_free_ion_concentrationPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_aqueous_complexes);
    alquimia_data->secondary_activity_coeffPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_aqueous_complexes);

    for (i = 0; i < alquimia_data->chem_sizes.num_aqueous_complexes; i++)
    {
      alquimia_data->secondary_free_ion_concentrationPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->secondary_free_ion_concentrationPF[i], 0.0);

      alquimia_data->secondary_activity_coeffPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->secondary_activity_coeffPF[i], 0.0);
    }
  }

  // num_surface_sites - site density
  if (alquimia_data->chem_sizes.num_surface_sites > 0)
  {
    alquimia_data->surface_site_densityPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_surface_sites);

    for (i = 0; i < alquimia_data->chem_sizes.num_surface_sites; i++)
    {
      alquimia_data->surface_site_densityPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->surface_site_densityPF[i], 0.0);
    }
  }

  // num_ion_exchange_sites - cation exchange capacity
  if (alquimia_data->chem_sizes.num_ion_exchange_sites > 0)
  {
    alquimia_data->cation_exchange_capacityPF = ctalloc(Vector *, alquimia_data->chem_sizes.num_ion_exchange_sites);

    for (i = 0; i < alquimia_data->chem_sizes.num_ion_exchange_sites; i++)
    {
      alquimia_data->cation_exchange_capacityPF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->cation_exchange_capacityPF[i], 0.0);
    }
  }

  // num_aqueous_kinetics - aqueous kinetic rate
  if (alquimia_data->chem_sizes.num_aqueous_kinetics > 0)
  {
    alquimia_data->aqueous_kinetic_ratePF = ctalloc(Vector *, alquimia_data->chem_sizes.num_aqueous_kinetics);

    for (i = 0; i < alquimia_data->chem_sizes.num_aqueous_kinetics; i++)
    {
      alquimia_data->aqueous_kinetic_ratePF[i] = NewVectorType(grid, 1, 0, vector_cell_centered);
      InitVectorAll(alquimia_data->aqueous_kinetic_ratePF[i], 0.0);
    }
  }
}



void AllocateChemCells(AlquimiaDataPF *alquimia_data, Grid *grid, ProblemData *problem_data)
{
  SubgridArray  *subgrids = GridSubgrids(grid);
  Subgrid       *subgrid;
  int is;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int chem_index;
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
      chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

      AllocateAlquimiaState(&alquimia_data->chem_sizes, &alquimia_data->chem_state[chem_index]);
      AllocateAlquimiaProperties(&alquimia_data->chem_sizes, &alquimia_data->chem_properties[chem_index]);
      AllocateAlquimiaAuxiliaryData(&alquimia_data->chem_sizes, &alquimia_data->chem_aux_data[chem_index]);
      AllocateAlquimiaAuxiliaryOutputData(&alquimia_data->chem_sizes, &alquimia_data->chem_aux_output[chem_index]);
    });
  }
  AllocateAlquimiaState(&alquimia_data->chem_sizes, &alquimia_data->chem_state_temp);
  AllocateAlquimiaAuxiliaryData(&alquimia_data->chem_sizes, &alquimia_data->chem_aux_data_temp);
  AllocateAlquimiaProperties(&alquimia_data->chem_sizes, &alquimia_data->chem_properties_temp);
}
#endif
