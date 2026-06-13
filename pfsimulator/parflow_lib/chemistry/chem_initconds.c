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
* Allocates and processes the geochemical conditions in the incoming list
* on a cell-by-cell basis.
*
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"


void ProcessGeochemICs(AlquimiaDataPF *alquimia_data, Grid *grid, ProblemData *problem_data, int num_ic_conds, NameArray ic_cond_na, Vector * saturation)
{
  double water_density = 998.0; // density of water in kg/m**3
  double aqueous_pressure = 101325.0; // pressure in Pa.
  Subgrid       *subgrid;
  Subvector     *chem_ind_sub;
  Subvector     *por_sub, *sat_sub;
  int is = 0;
  int i, j, k;
  int ix, iy, iz;
  int nx, ny, nz;
  int dx, dy, dz;
  int r;
  int chem_index, cond_index, por_index, sat_index;
  double *chem_ind, *por, *sat;
  GrGeomSolid   *gr_domain;
  char* name;

  SubgridArray  *subgrids = GridSubgrids(grid);

  gr_domain = ProblemDataGrDomain(problem_data);
  subgrid = SubgridArraySubgrid(subgrids, is);



  // assign interior geochemical conditions
  if (num_ic_conds > 0)
  {
    AllocateAlquimiaGeochemicalConditionVector(num_ic_conds, &alquimia_data->ic_condition_list);
    for (int i = 0; i < num_ic_conds; i++)
    {
      name = NA_IndexToName(ic_cond_na, i);
      AllocateAlquimiaGeochemicalCondition(strlen(name), 0, 0, &alquimia_data->ic_condition_list.data[i]);
      strcpy(alquimia_data->ic_condition_list.data[i].name, name);
    }
  }


  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    double vol = dx * dy * dz;

    // RDF: assume resolution is the same in all 3 directions
    r = SubgridRX(subgrid);

    chem_ind_sub = VectorSubvector(ProblemDataGeochemCond(problem_data), is);
    chem_ind = SubvectorData(chem_ind_sub);

    por_sub = VectorSubvector(ProblemDataPorosity(problem_data), is);
    por = SubvectorData(por_sub);

    sat_sub = VectorSubvector(saturation, is);
    sat = SubvectorData(sat_sub);

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      cond_index = SubvectorEltIndex(chem_ind_sub, i, j, k);
      por_index = SubvectorEltIndex(por_sub, i, j, k);
      sat_index = SubvectorEltIndex(sat_sub, i, j, k);

      chem_index = (i - ix) + (j - iy) * nx + (k - iz) * nx * ny;

      alquimia_data->chem_properties[chem_index].volume = vol;
      alquimia_data->chem_properties[chem_index].saturation = sat[sat_index];

      alquimia_data->chem_state[chem_index].water_density = water_density;
      alquimia_data->chem_state[chem_index].temperature = 25.0;
      alquimia_data->chem_state[chem_index].porosity = por[por_index];
      alquimia_data->chem_state[chem_index].aqueous_pressure = aqueous_pressure;

      alquimia_data->chem.ProcessCondition(&alquimia_data->chem_engine,
                                           &alquimia_data->ic_condition_list.data[(int)chem_ind[cond_index]],
                                           &alquimia_data->chem_properties[chem_index],
                                           &alquimia_data->chem_state[chem_index],
                                           &alquimia_data->chem_aux_data[chem_index],
                                           &alquimia_data->chem_status);
      if (alquimia_data->chem_status.error != 0)
      {
        printf("ProcessGeochemICs: initialization error: %s\n",
               alquimia_data->chem_status.message);
        PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
      }

      alquimia_data->chem.GetAuxiliaryOutput(&alquimia_data->chem_engine,
                                             &alquimia_data->chem_properties[chem_index],
                                             &alquimia_data->chem_state[chem_index],
                                             &alquimia_data->chem_aux_data[chem_index],
                                             &alquimia_data->chem_aux_output[chem_index],
                                             &alquimia_data->chem_status);
      if (alquimia_data->chem_status.error != 0)
      {
        amps_Printf("GetAuxiliaryOutput() auxiliary output fetch failed: %s\n",
                    alquimia_data->chem_status.message);
        PARFLOW_ERROR("Geochemical engine error, exiting simulation.\n");
      }
    });
  }
}
#endif

