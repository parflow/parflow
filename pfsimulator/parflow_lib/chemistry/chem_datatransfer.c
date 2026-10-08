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
* Transfers data from AlquimiaState and AlquimiaAuxiliaryOutputData structs
* to ParFlow Vectors
*
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_containers.h"


/** @brief Copy post-advection mobile concentrations from ParFlow vectors into
 * the per-cell Alquimia state, ahead of a reaction step.
 *
 * @param chem_state the Alquimia state to fill (per cell)
 * @param chem_sizes the Alquimia problem sizes (primary species count, ...)
 * @param concentrations per-contaminant mobile concentration vectors (source)
 * @param problem_data the problem data
 */
void AdvectedPrimaryToChem(AlquimiaState* chem_state, AlquimiaSizes* chem_sizes, Vector **concentrations, ProblemData *problem_data)
{
  Grid          *grid = VectorGrid(concentrations[0]);
  SubgridArray  *subgrids = GridSubgrids(grid);
  GrGeomSolid   *gr_domain;
  Subgrid       *subgrid;
  Subvector     *concen_sub;
  int is;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int chem_index, pf_index;
  int concen;
  double *concen_dat;

  gr_domain = ProblemDataGrDomain(problem_data);
  int num_primary = chem_sizes->num_primary;

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    for (concen = 0; concen < num_primary; concen++)
    {
      concen_sub = VectorSubvector(concentrations[concen], is);
      concen_dat = SubvectorData(concen_sub);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index = SubvectorEltIndex(concen_sub, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

        chem_state[chem_index].total_mobile.data[concen] = concen_dat[pf_index];
      });
    }
  }
}



/** @brief Unpack the per-cell Alquimia state and auxiliary output into the
 * ParFlow chemistry output vectors (pH, mineral, sorbed, free-ion, ... fields).
 *
 * @param alquimia_data the per-cell Alquimia state/properties/aux and the
 *        destination PF output vectors
 * @param grid the computational grid
 * @param problem_data the problem data
 */
void ChemDataToPFVectors(AlquimiaDataPF *alquimia_data, Grid *grid, ProblemData *problem_data)
{
  SubgridArray  *subgrids = GridSubgrids(grid);
  GrGeomSolid   *gr_domain;
  Subgrid       *subgrid;
  int is;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int chem_index, pf_index1, pf_index2, pf_index3, pf_index4;
  double *pf_dat1, *pf_dat2, *pf_dat3, *pf_dat4;
  Subvector *pf_sub1, *pf_sub2, *pf_sub3, *pf_sub4;


  gr_domain = ProblemDataGrDomain(problem_data);
  int num_primary = alquimia_data->chem_sizes.num_primary;
  int num_minerals = alquimia_data->chem_sizes.num_minerals;
  int num_surf_sites = alquimia_data->chem_sizes.num_surface_sites;
  int num_ion_exchange_sites = alquimia_data->chem_sizes.num_ion_exchange_sites;
  int num_aqueous_kinetics = alquimia_data->chem_sizes.num_aqueous_kinetics;
  int num_aqueous_complexes = alquimia_data->chem_sizes.num_aqueous_complexes;
  int num_sorbed = alquimia_data->chem_sizes.num_sorbed;

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    // pH
    if (num_primary > 0)
    {
      pf_sub1 = VectorSubvector(alquimia_data->pH, is);
      pf_dat1 = SubvectorData(pf_sub1);
      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;
        pf_dat1[pf_index1] = alquimia_data->chem_aux_output[chem_index].pH;
      });
    }

    //num_primary - primary activity and primary free ion
    for (int concen = 0; concen < num_primary; concen++)
    {
      pf_sub2 = VectorSubvector(alquimia_data->primary_free_ion_concentrationPF[concen], is);
      pf_dat2 = SubvectorData(pf_sub2);

      pf_sub3 = VectorSubvector(alquimia_data->primary_activity_coeffPF[concen], is);
      pf_dat3 = SubvectorData(pf_sub3);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index2 = SubvectorEltIndex(pf_sub2, i, j, k);
        pf_index3 = SubvectorEltIndex(pf_sub3, i, j, k);

        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

        pf_dat2[pf_index2] = alquimia_data->chem_aux_output[chem_index].primary_free_ion_concentration.data[concen];
        pf_dat3[pf_index3] = alquimia_data->chem_aux_output[chem_index].primary_activity_coeff.data[concen];
      });
    }

    // num_minerals -volfx, SI, SSA, rate
    for (int mineral = 0; mineral < num_minerals; mineral++)
    {
      pf_sub1 = VectorSubvector(alquimia_data->mineral_volume_fractionsPF[mineral], is);
      pf_dat1 = SubvectorData(pf_sub1);

      pf_sub2 = VectorSubvector(alquimia_data->mineral_specific_surfacePF[mineral], is);
      pf_dat2 = SubvectorData(pf_sub2);

      pf_sub3 = VectorSubvector(alquimia_data->mineral_saturation_indexPF[mineral], is);
      pf_dat3 = SubvectorData(pf_sub3);

      pf_sub4 = VectorSubvector(alquimia_data->mineral_reaction_ratePF[mineral], is);
      pf_dat4 = SubvectorData(pf_sub4);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
        pf_index2 = SubvectorEltIndex(pf_sub2, i, j, k);
        pf_index3 = SubvectorEltIndex(pf_sub3, i, j, k);
        pf_index4 = SubvectorEltIndex(pf_sub4, i, j, k);

        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

        pf_dat1[pf_index1] = alquimia_data->chem_state[chem_index].mineral_volume_fraction.data[mineral];
        pf_dat2[pf_index2] = alquimia_data->chem_state[chem_index].mineral_specific_surface_area.data[mineral];
        pf_dat3[pf_index3] = alquimia_data->chem_aux_output[chem_index].mineral_saturation_index.data[mineral];
        pf_dat4[pf_index4] = alquimia_data->chem_aux_output[chem_index].mineral_reaction_rate.data[mineral];
      });
    }

    //num_surf_sites - site density
    for (int surf = 0; surf < num_surf_sites; surf++)
    {
      pf_sub1 = VectorSubvector(alquimia_data->surface_site_densityPF[surf], is);
      pf_dat1 = SubvectorData(pf_sub1);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;
        pf_dat1[pf_index1] = alquimia_data->chem_state[chem_index].surface_site_density.data[surf];
      });
    }

    // num_ion_exchange_sites - CEC
    for (int ion = 0; ion < num_ion_exchange_sites; ion++)
    {
      pf_sub1 = VectorSubvector(alquimia_data->cation_exchange_capacityPF[ion], is);
      pf_dat1 = SubvectorData(pf_sub1);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;
        pf_dat1[pf_index1] = alquimia_data->chem_state[chem_index].cation_exchange_capacity.data[ion];
      });
    }

    // num_aqueous_kinetics - aqueous kinetic rate
    for (int rate = 0; rate < num_aqueous_kinetics; rate++)
    {
      pf_sub1 = VectorSubvector(alquimia_data->aqueous_kinetic_ratePF[rate], is);
      pf_dat1 = SubvectorData(pf_sub1);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;
        pf_dat1[pf_index1] = alquimia_data->chem_aux_output[chem_index].aqueous_kinetic_rate.data[rate];
      });
    }

    // num_aqueous_complexes - activity andd free ion
    for (int complex = 0; complex < num_aqueous_complexes; complex++)
    {
      pf_sub1 = VectorSubvector(alquimia_data->secondary_free_ion_concentrationPF[complex], is);
      pf_dat1 = SubvectorData(pf_sub1);

      pf_sub2 = VectorSubvector(alquimia_data->secondary_activity_coeffPF[complex], is);
      pf_dat2 = SubvectorData(pf_sub2);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
        pf_index2 = SubvectorEltIndex(pf_sub2, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;
        pf_dat1[pf_index1] = alquimia_data->chem_aux_output[chem_index].secondary_free_ion_concentration.data[complex];
        pf_dat2[pf_index2] = alquimia_data->chem_aux_output[chem_index].secondary_activity_coeff.data[complex];
      });
    }

    // num_sorbed - total_immobile- 1 per primary
    if (num_sorbed > 0)
    {
      for (int sorbed = 0; sorbed < num_primary; sorbed++)
      {
        pf_sub1 = VectorSubvector(alquimia_data->total_immobilePF[sorbed], is);
        pf_dat1 = SubvectorData(pf_sub1);

        GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
        {
          pf_index1 = SubvectorEltIndex(pf_sub1, i, j, k);
          chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;
          pf_dat1[pf_index1] = alquimia_data->chem_state[chem_index].total_immobile.data[sorbed];
        });
      }
    }
  }
}




/** @brief Copy the reacted mobile concentrations from the per-cell Alquimia
 * state back into the ParFlow concentration vectors, after a reaction step.
 *
 * @param chem_state the reacted Alquimia state (per cell, source)
 * @param chem_sizes the Alquimia problem sizes
 * @param concentrations per-contaminant mobile concentration vectors (updated)
 * @param problem_data the problem data
 */
void ReactedPrimaryToPF(AlquimiaState* chem_state, AlquimiaSizes* chem_sizes, Vector **concentrations, ProblemData *problem_data)
{
  Grid          *grid = VectorGrid(concentrations[0]);
  SubgridArray  *subgrids = GridSubgrids(grid);
  GrGeomSolid   *gr_domain;
  Subgrid       *subgrid;
  Subvector     *concen_sub;
  int is;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int chem_index, pf_index;
  int concen;
  double *concen_dat;

  gr_domain = ProblemDataGrDomain(problem_data);
  int num_primary = chem_sizes->num_primary;

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    for (concen = 0; concen < num_primary; concen++)
    {
      concen_sub = VectorSubvector(concentrations[concen], is);
      concen_dat = SubvectorData(concen_sub);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        pf_index = SubvectorEltIndex(concen_sub, i, j, k);
        chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

        concen_dat[pf_index] = chem_state[chem_index].total_mobile.data[concen];
      });
    }
  }
}
#endif

