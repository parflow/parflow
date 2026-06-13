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
* Allocates and processes the boundary conditions named in the input file.
* The only data we need from the processed conditions are the total mobile
* concentrations located in chem_bc_state.
*
*****************************************************************************/

#include "parflow.h"
#include "pf_alquimia.h"

#ifdef HAVE_ALQUIMIA
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"



/** @brief Process the named geochemical boundary conditions through the engine
 * into Alquimia states, so inflow concentrations can be applied at the
 * boundary patches (BCConcentration).
 *
 * @param alquimia_data the chemistry data container (boundary-condition states)
 * @param num_bc_conds the number of named boundary geochemical conditions
 * @param bc_cond_na the names of those conditions (from the engine input deck)
 */
void ProcessGeochemBCs(AlquimiaDataPF *alquimia_data, int num_bc_conds, NameArray bc_cond_na)
{
  char* name;
  double water_density = 998.0;    // density of water in kg/m**3
  double aqueous_pressure = 101325.0; // pressure in Pa.

  if (num_bc_conds > 0)
  {
    AllocateAlquimiaGeochemicalConditionVector(num_bc_conds, &alquimia_data->bc_condition_list);
    alquimia_data->chem_bc_properties = ctalloc(AlquimiaProperties, num_bc_conds);
    alquimia_data->chem_bc_state = ctalloc(AlquimiaState, num_bc_conds);
    alquimia_data->chem_bc_aux_data = ctalloc(AlquimiaAuxiliaryData, num_bc_conds);

    for (int i = 0; i < num_bc_conds; i++)
    {
      name = NA_IndexToName(bc_cond_na, i);
      AllocateAlquimiaGeochemicalCondition(strlen(name), 0, 0, &alquimia_data->bc_condition_list.data[i]);
      strcpy(alquimia_data->bc_condition_list.data[i].name, name);

      AllocateAlquimiaState(&alquimia_data->chem_sizes, &alquimia_data->chem_bc_state[i]);
      AllocateAlquimiaProperties(&alquimia_data->chem_sizes, &alquimia_data->chem_bc_properties[i]);
      AllocateAlquimiaAuxiliaryData(&alquimia_data->chem_sizes, &alquimia_data->chem_bc_aux_data[i]);
    }

    for (int i = 0; i < num_bc_conds; i++)
    {
      alquimia_data->chem_bc_state[i].water_density = water_density;
      alquimia_data->chem_bc_state[i].temperature = 25.0;
      alquimia_data->chem_bc_state[i].porosity = 1.0;
      alquimia_data->chem_bc_state[i].aqueous_pressure = aqueous_pressure;

      alquimia_data->chem.ProcessCondition(&alquimia_data->chem_engine,
                                           &alquimia_data->bc_condition_list.data[i],
                                           &alquimia_data->chem_bc_properties[i],
                                           &alquimia_data->chem_bc_state[i],
                                           &alquimia_data->chem_bc_aux_data[i],
                                           &alquimia_data->chem_status);
    }
  }
}
#endif
