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

#ifndef CHEM_DATASTRUCTS_H
#define CHEM_DATASTRUCTS_H

#include "alquimia/alquimia_containers.h"
#include "alquimia/alquimia_interface.h"


typedef struct _ChemPrintFlags {
  int print_primary_mobile;
  int silo_primary_mobile;
  int print_mineral_rate;
  int silo_mineral_rate;
  int print_mineral_volfx;
  int silo_mineral_volfx;
  int print_mineral_surfarea;
  int silo_mineral_surfarea;
  int print_surf_dens;
  int silo_surf_dens;
  int print_CEC;
  int silo_CEC;
  int print_pH;
  int silo_pH;
  int print_aqueous_rate;
  int silo_aqueous_rate;
  int print_mineral_SI;
  int silo_mineral_SI;
  int print_primary_freeion;
  int silo_primary_freeion;
  int print_primary_activity;
  int silo_primary_activity;
  int print_secondary_freeion;
  int silo_secondary_freeion;
  int print_secondary_activity;
  int silo_secondary_activity;
  int print_sorbed;
  int silo_sorbed;
} ChemPrintFlags;




typedef struct _AlquimiaDataPF {
  // Per-cell chemistry data.
  AlquimiaProperties* chem_properties;
  AlquimiaState* chem_state;
  AlquimiaAuxiliaryData* chem_aux_data;
  AlquimiaAuxiliaryOutputData* chem_aux_output;

  // Chemistry engine -- one of each of these per thread in general.
  AlquimiaInterface chem;
  void* chem_engine;
  AlquimiaEngineStatus chem_status;

  // Chemistry metadata.
  AlquimiaSizes chem_sizes;
  AlquimiaProblemMetaData chem_metadata;

  // Initial and boundary conditions.
  AlquimiaState* chem_bc_state;
  AlquimiaAuxiliaryData* chem_bc_aux_data;
  AlquimiaProperties* chem_bc_properties;
  AlquimiaGeochemicalConditionVector ic_condition_list, bc_condition_list;

  // engine functionality
  AlquimiaEngineFunctionality chem_engine_functionality;

  // PF Vectors, needed for printing
  //state vars - also concen vector, but keep that outside
  Vector **total_immobilePF; //immobile portion of primary, num_primary or 0
  Vector **mineral_volume_fractionsPF; // num_minerals
  Vector **mineral_specific_surfacePF; // num_minerals
  Vector **surface_site_densityPF; // num_surface_sites
  Vector **cation_exchange_capacityPF; //num_ion_exchange_sites

  //aux output data
  Vector *pH; // ony one value per cell
  Vector **aqueous_kinetic_ratePF; // num_aqueous_kinetics
  Vector **mineral_saturation_indexPF; // num_minerals
  Vector **mineral_reaction_ratePF; // num_minerals
  Vector **primary_free_ion_concentrationPF; // num_primary
  Vector **primary_activity_coeffPF; // num_primary
  Vector **secondary_free_ion_concentrationPF; // num_aqueous_complexes
  Vector **secondary_activity_coeffPF; // num_aqueous_complexes

  //printing flags
  ChemPrintFlags *print_flags;

  // temporary containers to hold single cell solution
  AlquimiaState chem_state_temp;
  AlquimiaAuxiliaryData chem_aux_data_temp;
  AlquimiaProperties chem_properties_temp;
} AlquimiaDataPF;

#endif

