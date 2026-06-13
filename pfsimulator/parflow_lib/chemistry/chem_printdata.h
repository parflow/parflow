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

#ifndef CHEM_PRINTDATA_H
#define CHEM_PRINTDATA_H

#include "metadata.h"


/* chem_printdata.c */
void PrintChemistryData(ChemPrintFlags *print_flags, AlquimiaSizes *chem_sizes, AlquimiaProblemMetaData *chem_metadata,
                        double t, int file_number, char* file_prefix, int *any_file_dumped, Vector **concentrations,
                        Vector **total_immobilePF, Vector **mineral_specific_surfacePF, Vector **mineral_volume_fractionsPF, Vector **surface_site_densityPF,
                        Vector **cation_exchange_capacityPF, Vector *pH, Vector **aqueous_kinetic_ratePF, Vector **mineral_saturation_indexPF,
                        Vector **mineral_reaction_ratePF, Vector **primary_free_ion_concentrationPF, Vector **primary_activity_coeffPF,
                        Vector **secondary_free_ion_concentrationPF, Vector **secondary_activity_coeffPF);

void CreateChemistryMetadata(ChemPrintFlags *print_flags, AlquimiaSizes *chem_sizes, AlquimiaProblemMetaData *chem_metadata, double t,
                             char* file_prefix, MetadataItem *js_outputs);

void CreateChemMetadataEntry(MetadataItem *js_outputs, AlquimiaVectorString *names, char* file_prefix, char *postfix_format,
                             const char* field_name, const char* field_units, int size, double time);

void UpdateChemistryMetadata(ChemPrintFlags *print_flags, double t, char* file_prefix, int file_number, MetadataItem *js_outputs);
#endif

