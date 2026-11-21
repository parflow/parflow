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
* Frees the entire AlquimiaDataPF struct
*
*
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"


/** @brief Free everything allocated for the chemistry coupling: the per-cell
 * Alquimia containers and the ParFlow-side output vectors in AlquimiaDataPF.
 *
 * @param alquimia_data the chemistry data container to tear down
 * @param grid the computational grid
 * @param problem_data the problem data (active-cell mask, ...)
 */
void FreeAlquimiaDataPF(AlquimiaDataPF *alquimia_data, Grid *grid, ProblemData *problem_data)
{
  SubgridArray  *subgrids = GridSubgrids(grid);
  Subgrid       *subgrid;
  int is;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int chem_index;
  int ai = 0;

  int num_primary = alquimia_data->chem_sizes.num_primary;
  int num_minerals = alquimia_data->chem_sizes.num_minerals;
  int num_surf_sites = alquimia_data->chem_sizes.num_surface_sites;
  int num_ion_exchange_sites = alquimia_data->chem_sizes.num_ion_exchange_sites;
  int num_aqueous_kinetics = alquimia_data->chem_sizes.num_aqueous_kinetics;
  int num_aqueous_complexes = alquimia_data->chem_sizes.num_aqueous_complexes;
  int num_sorbed = alquimia_data->chem_sizes.num_sorbed;

  // pH
  //num_primary - concen (total_mobile), primary activity and free ion
  if (num_primary > 0)
  {
    for (int concen = 0; concen < num_primary; concen++)
    {
      FreeVector(alquimia_data->primary_free_ion_concentrationPF[concen]);
      FreeVector(alquimia_data->primary_activity_coeffPF[concen]);
    }

    FreeVector(alquimia_data->pH);
    tfree(alquimia_data->primary_free_ion_concentrationPF);
    tfree(alquimia_data->primary_activity_coeffPF);
  }


  if (num_minerals > 0)
  {
    for (int mineral = 0; mineral < num_minerals; mineral++)
    {
      FreeVector(alquimia_data->mineral_volume_fractionsPF[mineral]);
      FreeVector(alquimia_data->mineral_specific_surfacePF[mineral]);
      FreeVector(alquimia_data->mineral_saturation_indexPF[mineral]);
      FreeVector(alquimia_data->mineral_reaction_ratePF[mineral]);
    }

    tfree(alquimia_data->mineral_volume_fractionsPF);
    tfree(alquimia_data->mineral_specific_surfacePF);
    tfree(alquimia_data->mineral_saturation_indexPF);
    tfree(alquimia_data->mineral_reaction_ratePF);
  }


  if (num_surf_sites > 0)
  {
    for (int surf = 0; surf < num_surf_sites; surf++)
    {
      FreeVector(alquimia_data->surface_site_densityPF[surf]);
    }

    tfree(alquimia_data->surface_site_densityPF);
  }


  if (num_ion_exchange_sites > 0)
  {
    for (int ion = 0; ion < num_ion_exchange_sites; ion++)
    {
      FreeVector(alquimia_data->cation_exchange_capacityPF[ion]);
    }

    tfree(alquimia_data->cation_exchange_capacityPF);
  }


  if (num_aqueous_kinetics > 0)
  {
    for (int rate = 0; rate < num_aqueous_kinetics; rate++)
    {
      FreeVector(alquimia_data->aqueous_kinetic_ratePF[rate]);
    }

    tfree(alquimia_data->aqueous_kinetic_ratePF);
  }


  if (num_aqueous_complexes > 0)
  {
    for (int complex = 0; complex < num_aqueous_complexes; complex++)
    {
      FreeVector(alquimia_data->secondary_free_ion_concentrationPF[complex]);
      FreeVector(alquimia_data->secondary_activity_coeffPF[complex]);
    }

    tfree(alquimia_data->secondary_free_ion_concentrationPF);
    tfree(alquimia_data->secondary_activity_coeffPF);
  }


  if (num_sorbed > 0)
  {
    for (int sorbed = 0; sorbed < num_primary; sorbed++)
    {
      FreeVector(alquimia_data->total_immobilePF[sorbed]);
    }

    tfree(alquimia_data->total_immobilePF);
  }


  // Destroy boundary and initial conditions.
  for (int i = 0; i < alquimia_data->bc_condition_list.size; i++)
  {
    FreeAlquimiaState(&alquimia_data->chem_bc_state[i]);
    FreeAlquimiaAuxiliaryData(&alquimia_data->chem_bc_aux_data[i]);
    FreeAlquimiaProperties(&alquimia_data->chem_bc_properties[i]);
  }
  FreeAlquimiaGeochemicalConditionVector(&alquimia_data->bc_condition_list);
  FreeAlquimiaGeochemicalConditionVector(&alquimia_data->ic_condition_list);

  // Destroy cell-by-cell data
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

      FreeAlquimiaState(&alquimia_data->chem_state[chem_index]);
      FreeAlquimiaProperties(&alquimia_data->chem_properties[chem_index]);
      FreeAlquimiaAuxiliaryData(&alquimia_data->chem_aux_data[chem_index]);
      FreeAlquimiaAuxiliaryOutputData(&alquimia_data->chem_aux_output[chem_index]);
    });
  }

  tfree(alquimia_data->chem_state);
  tfree(alquimia_data->chem_properties);
  tfree(alquimia_data->chem_aux_data);
  tfree(alquimia_data->chem_aux_output);
  tfree(alquimia_data->chem_bc_state);
  tfree(alquimia_data->chem_bc_aux_data);
  tfree(alquimia_data->chem_bc_properties);

  FreeAlquimiaProblemMetaData(&alquimia_data->chem_metadata);

  // free temp alquimia data
  FreeAlquimiaState(&alquimia_data->chem_state_temp);
  FreeAlquimiaProperties(&alquimia_data->chem_properties_temp);
  FreeAlquimiaAuxiliaryData(&alquimia_data->chem_aux_data_temp);


  // Destroy chemistry engine.
  alquimia_data->chem.Shutdown(&alquimia_data->chem_engine,
                               &alquimia_data->chem_status);
  FreeAlquimiaEngineStatus(&alquimia_data->chem_status);

  tfree(alquimia_data->print_flags);
  tfree(alquimia_data);
}
#endif

