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
* Print all of the chemistry data the user asked for
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/
#include "parflow.h"
#include "pf_alquimia.h"


#ifdef HAVE_ALQUIMIA

/** @brief Write the enabled chemistry output fields as PFB/Silo for this dump.
 *
 * The set of fields written (pH, mineral volume fractions / surface areas /
 * rates / saturation indices, sorbed and free-ion species, surface-site
 * densities, CEC, activity coefficients, aqueous rates) is selected by the
 * Chemistry.Print and WriteSilo flags captured in print_flags. Sets
 * any_file_dumped when anything is written.
 */
void PrintChemistryData(ChemPrintFlags *print_flags, AlquimiaSizes *chem_sizes, AlquimiaProblemMetaData *chem_metadata, double t, int file_number,
                        char* file_prefix, int *any_file_dumped, Vector **concentrations, Vector **total_immobilePF, Vector **mineral_specific_surfacePF, Vector **mineral_volume_fractionsPF,
                        Vector **surface_site_densityPF, Vector **cation_exchange_capacityPF, Vector *pH, Vector **aqueous_kinetic_ratePF, Vector **mineral_saturation_indexPF,
                        Vector **mineral_reaction_ratePF, Vector **primary_free_ion_concentrationPF, Vector **primary_activity_coeffPF, Vector **secondary_free_ion_concentrationPF,
                        Vector **secondary_activity_coeffPF)
{
  char file_type[2048], file_postfix[2048];

  //primary mobile
  if (print_flags->print_primary_mobile)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "PrimaryMobile.%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    concentrations[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_primary_mobile)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      sprintf(file_type, "PrimaryMobile");
      WriteSilo(file_prefix, file_type, file_postfix, concentrations[i],
                t, file_number, "Concen");
    }
    *any_file_dumped = 1;
  }



  //primary sorbed
  if (print_flags->print_sorbed && chem_sizes->num_sorbed > 0)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "PrimarySorbed.%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    total_immobilePF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_sorbed && chem_sizes->num_sorbed > 0)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      sprintf(file_type, "PrimarySorbed");
      WriteSilo(file_prefix, file_type, file_postfix, total_immobilePF[i],
                t, file_number, "Sorbed");
    }
    *any_file_dumped = 1;
  }



  //mineral volfx
  if (print_flags->print_mineral_volfx)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "MineralVolfx.%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    mineral_volume_fractionsPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_mineral_volfx)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      sprintf(file_type, "MineralVolfx");
      WriteSilo(file_prefix, file_type, file_postfix, mineral_volume_fractionsPF[i],
                t, file_number, "Volfx");
    }
    *any_file_dumped = 1;
  }



  //mineral surface area
  if (print_flags->print_mineral_surfarea)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "MineralSurfArea.%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    mineral_specific_surfacePF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_mineral_surfarea)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      sprintf(file_type, "MineralSurfArea");
      WriteSilo(file_prefix, file_type, file_postfix, mineral_specific_surfacePF[i],
                t, file_number, "SurfArea");
    }
    *any_file_dumped = 1;
  }



  //mineral saturation index
  if (print_flags->print_mineral_SI)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "MineralSI.%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    mineral_saturation_indexPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_mineral_SI)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      sprintf(file_type, "MineralSI");
      WriteSilo(file_prefix, file_type, file_postfix, mineral_saturation_indexPF[i],
                t, file_number, "SatIndex");
    }
    *any_file_dumped = 1;
  }



  //mineral kinetic rate
  if (print_flags->print_mineral_rate)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "MineralRate.%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    mineral_reaction_ratePF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_mineral_rate)
  {
    for (int i = 0; i < chem_sizes->num_minerals; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->mineral_names.data[i], file_number);
      sprintf(file_type, "MineralRate");
      WriteSilo(file_prefix, file_type, file_postfix, mineral_reaction_ratePF[i],
                t, file_number, "MineralRate");
    }
    *any_file_dumped = 1;
  }



  //surface site density
  if (print_flags->print_surf_dens)
  {
    for (int i = 0; i < chem_sizes->num_surface_sites; i++)
    {
      sprintf(file_postfix, "SurfSiteDens.%02d.%s.%05d", i, chem_metadata->surface_site_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    surface_site_densityPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_surf_dens)
  {
    for (int i = 0; i < chem_sizes->num_surface_sites; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->surface_site_names.data[i], file_number);
      sprintf(file_type, "SurfSiteDens");
      WriteSilo(file_prefix, file_type, file_postfix, surface_site_densityPF[i],
                t, file_number, "SurfSiteDens");
    }
    *any_file_dumped = 1;
  }



  //cation exchange capacity
  if (print_flags->print_CEC)
  {
    for (int i = 0; i < chem_sizes->num_ion_exchange_sites; i++)
    {
      sprintf(file_postfix, "CEC.%02d.%s.%05d", i, chem_metadata->ion_exchange_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    cation_exchange_capacityPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_CEC)
  {
    for (int i = 0; i < chem_sizes->num_ion_exchange_sites; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->ion_exchange_names.data[i], file_number);
      sprintf(file_type, "CEC");
      WriteSilo(file_prefix, file_type, file_postfix, cation_exchange_capacityPF[i],
                t, file_number, "CEC");
    }
    *any_file_dumped = 1;
  }



  //pH
  if (print_flags->print_pH)
  {
    sprintf(file_postfix, "pH.%05d", file_number);
    WritePFBinary(file_prefix, file_postfix,
                  pH);
    *any_file_dumped = 1;
  }

  if (print_flags->silo_pH)
  {
    sprintf(file_postfix, "%05d", file_number);
    sprintf(file_type, "pH");
    WriteSilo(file_prefix, file_type, file_postfix, pH,
              t, file_number, "pH");
    *any_file_dumped = 1;
  }



  //aqueous kinetic rate
  if (print_flags->print_aqueous_rate)
  {
    for (int i = 0; i < chem_sizes->num_aqueous_kinetics; i++)
    {
      sprintf(file_postfix, "AqueousRate.%02d.%s.%05d", i, chem_metadata->aqueous_kinetic_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    aqueous_kinetic_ratePF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_aqueous_rate)
  {
    for (int i = 0; i < chem_sizes->num_aqueous_kinetics; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->aqueous_kinetic_names.data[i], file_number);
      sprintf(file_type, "AqueousRate");
      WriteSilo(file_prefix, file_type, file_postfix, aqueous_kinetic_ratePF[i],
                t, file_number, "AqueousRate");
    }
    *any_file_dumped = 1;
  }



  //primary free ion concentration
  if (print_flags->print_primary_freeion)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "PrimaryFreeIon.%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    primary_free_ion_concentrationPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_primary_freeion)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      sprintf(file_type, "PrimaryFreeIon");
      WriteSilo(file_prefix, file_type, file_postfix, primary_free_ion_concentrationPF[i],
                t, file_number, "PrimaryFreeIon");
    }
    *any_file_dumped = 1;
  }



  //primary activity
  if (print_flags->print_primary_activity)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "PrimaryActivity.%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      WritePFBinary(file_prefix, file_postfix,
                    primary_activity_coeffPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_primary_activity)
  {
    for (int i = 0; i < chem_sizes->num_primary; i++)
    {
      sprintf(file_postfix, "%02d.%s.%05d", i, chem_metadata->primary_names.data[i], file_number);
      sprintf(file_type, "PrimaryActivity");
      WriteSilo(file_prefix, file_type, file_postfix, primary_activity_coeffPF[i],
                t, file_number, "PrimaryActivity");
    }
    *any_file_dumped = 1;
  }



  //secondary free ion concentration
  if (print_flags->print_secondary_freeion)
  {
    for (int i = 0; i < chem_sizes->num_aqueous_complexes; i++)
    {
      sprintf(file_postfix, "SecondaryFreeIon.%02d.%05d", i, file_number);
      WritePFBinary(file_prefix, file_postfix,
                    secondary_free_ion_concentrationPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_secondary_freeion)
  {
    for (int i = 0; i < chem_sizes->num_aqueous_complexes; i++)
    {
      sprintf(file_postfix, "%02d.%05d", i, file_number);
      sprintf(file_type, "SecondaryFreeIon");
      WriteSilo(file_prefix, file_type, file_postfix, secondary_free_ion_concentrationPF[i],
                t, file_number, "SecondaryFreeIon");
    }
    *any_file_dumped = 1;
  }



  //secondary activity
  if (print_flags->print_secondary_activity)
  {
    for (int i = 0; i < chem_sizes->num_aqueous_complexes; i++)
    {
      sprintf(file_postfix, "SecondaryActivity.%02d.%05d", i, file_number);
      WritePFBinary(file_prefix, file_postfix,
                    secondary_activity_coeffPF[i]);
    }
    *any_file_dumped = 1;
  }

  if (print_flags->silo_secondary_activity)
  {
    for (int i = 0; i < chem_sizes->num_aqueous_complexes; i++)
    {
      sprintf(file_postfix, "%02d.%05d", i, file_number);
      sprintf(file_type, "SecondaryActivity");
      WriteSilo(file_prefix, file_type, file_postfix, secondary_activity_coeffPF[i],
                t, file_number, "SecondaryActivity");
    }
    *any_file_dumped = 1;
  }
}

void CreateChemMetadataEntry(MetadataItem *js_outputs, AlquimiaVectorString *names, char* file_prefix, char *postfix_format,
                             const char* field_name, const char* field_units, int size, double time)
{
  const char *file_postfix[size];
  char temp[2048];

  if (names)
  {
    for (int i = 0; i < size; i++)
    {
      sprintf(temp, postfix_format, i, names->data[i]);
      file_postfix[i] = strdup(temp);
    }
  }
  else
  {
    for (int i = 0; i < size; i++)
    {
      sprintf(temp, postfix_format, i);
      file_postfix[i] = strdup(temp);
    }
  }
  MetadataAddDynamicField(*js_outputs, file_prefix, time, 0, field_name, field_units, "cell", "subsurface",
                          size, file_postfix);
  for (int i = 0; i < size; i++)
    free((char *)file_postfix[i]);
}

void CreateChemistryMetadata(ChemPrintFlags *print_flags, AlquimiaSizes *chem_sizes, AlquimiaProblemMetaData *chem_metadata, double t,
                             char* file_prefix, MetadataItem *js_outputs)
{
  if (print_flags->print_primary_mobile)
  {
    char *postfix_fmt = "PrimaryMobile.%02d.%s";
    char *field_name = "PrimaryMobile";
    char *field_units = "mol l^-1";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->primary_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_primary, t);
  }

  if (print_flags->print_sorbed && chem_sizes->num_sorbed > 0)
  {
    char *postfix_fmt = "PrimarySorbed.%02d.%s";
    char *field_name = "PrimarySorbed";
    char *field_units = "mol m^-3";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->primary_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_primary, t);
  }

  if (print_flags->print_mineral_volfx)
  {
    char *postfix_fmt = "MineralVolfx.%02d.%s";
    char *field_name = "MineralVolfx";
    char *field_units = "[-]";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->mineral_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_minerals, t);
  }

  if (print_flags->print_mineral_surfarea)
  {
    char *postfix_fmt = "MineralSurfArea.%02d.%s";
    char *field_name = "MineralSurfArea";
    char *field_units = "m^2 m^-3";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->mineral_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_minerals, t);
  }

  if (print_flags->print_mineral_SI)
  {
    char *postfix_fmt = "MineralSI.%02d.%s";
    char *field_name = "MineralSI";
    char *field_units = "[-]";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->mineral_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_minerals, t);
  }

  if (print_flags->print_mineral_rate)
  {
    char *postfix_fmt = "MineralRate.%02d.%s";
    char *field_name = "MineralRate";
    char *field_units = "mol m^-3 s^-1";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->mineral_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_minerals, t);
  }

  if (print_flags->print_surf_dens)
  {
    char *postfix_fmt = "SurfSiteDens.%02d.%s";
    char *field_name = "SurfSiteDens";
    char *field_units = "mol m^-3";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->surface_site_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_surface_sites, t);
  }

  if (print_flags->print_CEC)
  {
    char *postfix_fmt = "CEC.%02d.%s";
    char *field_name = "CEC";
    char *field_units = "mol m^-3";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->ion_exchange_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_ion_exchange_sites, t);
  }

  if (print_flags->print_pH)
  {
    const char *postfix_fmt[] = { "pH" };
    char *field_name = "pH";
    char *field_units = "[-]";
    MetadataAddDynamicField(*js_outputs, file_prefix, t, 0, field_name, field_units, "cell", "subsurface",
                            1, postfix_fmt);
  }

  if (print_flags->print_aqueous_rate)
  {
    char *postfix_fmt = "AqueousRate.%02d.%s";
    char *field_name = "AqueousRate";
    char *field_units = "mol m^-3 s^-1";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->aqueous_kinetic_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_aqueous_kinetics, t);
  }

  if (print_flags->print_primary_freeion)
  {
    char *postfix_fmt = "PrimaryFreeIon.%02d.%s";
    char *field_name = "PrimaryFreeIon";
    char *field_units = "mol kg^-1 H2O";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->primary_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_primary, t);
  }

  if (print_flags->print_primary_activity)
  {
    char *postfix_fmt = "PrimaryActivity.%02d.%s";
    char *field_name = "PrimaryActivity";
    char *field_units = "[-]";
    CreateChemMetadataEntry(js_outputs, &chem_metadata->primary_names, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_primary, t);
  }

  if (print_flags->print_secondary_freeion)
  {
    char *postfix_fmt = "SecondaryFreeIon.%02d";
    char *field_name = "SecondaryFreeIon";
    char *field_units = "mol kg^-1 H2O";
    CreateChemMetadataEntry(js_outputs, NULL, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_aqueous_complexes, t);
  }

  if (print_flags->print_secondary_activity)
  {
    char *postfix_fmt = "SecondaryActivity.%02d";
    char *field_name = "SecondaryActivity";
    char *field_units = "[-]";
    CreateChemMetadataEntry(js_outputs, NULL, file_prefix, postfix_fmt, field_name, field_units, chem_sizes->num_aqueous_complexes, t);
  }
}


void UpdateChemistryMetadata(ChemPrintFlags *print_flags, double t, char* file_prefix, int file_number, MetadataItem *js_outputs)
{
  if (print_flags->print_primary_mobile)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "PrimaryMobile", "mol l^-1", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_sorbed)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "PrimarySorbed", "mol m^-3", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_mineral_volfx)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "MineralVolfx", "[-]", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_mineral_surfarea)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "MineralSurfArea", "m^2 m^-3", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_mineral_SI)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "MineralSI", "[-]", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_mineral_rate)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "MineralRate", "mol m^-3 s^-1", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_surf_dens)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "SurfSiteDens", "mol m^-3", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_CEC)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "CEC", "mol m^-3", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_pH)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "pH", "[-]", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_aqueous_rate)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "AqueousRate", "mol m^-3 s^-1", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_primary_freeion)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "PrimaryFreeIon", "mol kg^-1 H2O", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_primary_activity)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "PrimaryActivity", "[-]", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_secondary_freeion)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "SecondaryFreeIon", "mol kg^-1 H2O", "cell", "subsurface", 0, NULL);
  }

  if (print_flags->print_secondary_activity)
  {
    MetadataAddDynamicField(*js_outputs, file_prefix, t, file_number, "SecondaryActivity", "[-]", "cell", "subsurface", 0, NULL);
  }
}

#endif

