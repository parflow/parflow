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

#include "parflow.h"

#include "seepage.h"


/**
 * @brief Populate Seepage patch information from input database
 *
 * Parses input database and builds lookup table to identify which patch_id's are
 * Seepage patches.   Seepage patches are specified using the input key:
 *   Patch.<patch_name>.BCPressure.Seepage = True
 *
 * @param publix_xtra nl_function publix extra
 * @return None
 */
void
PopulateSeepagePatchesFromBCPressure(SeepageLookup *seepage)
{
  char *patch_names;

  patch_names = GetStringDefault("BCPressure.PatchNames", NULL);
  if (patch_names == NULL || patch_names[0] == '\0')
  {
    return;
  }

  NameArray patches_na = NA_NewNameArray(patch_names);
  int num_patches = NA_Sizeof(patches_na);

  if (num_patches <= 0)
  {
    NA_FreeNameArray(patches_na);
    return;
  }

  char key[IDB_MAX_KEY_LEN];
  char *geom_name = GetString("Domain.GeomName");
  int domain_index = NA_NameToIndexExitOnError(GlobalsGeomNames, geom_name, "Domain.GeomName");

  NameArray switch_na = NA_NewNameArray("False True");

  seepage->seepage_patches = ctalloc(int, num_patches);

  for (int idx = 0; idx < num_patches; idx++)
  {
    char *patch_name = NA_IndexToName(patches_na, idx);
    /* Only consider patches that use the OverlandKinematic BC type */
    sprintf(key, "Patch.%s.BCPressure.Type", patch_name);
    char *type_name = GetStringDefault(key, NULL);
    if (type_name == NULL || strcmp(type_name, "OverlandKinematic") != 0)
    {
      continue;
    }

    sprintf(key, "Patch.%s.BCPressure.Seepage", patch_name);
    char *switch_name = GetStringDefault(key, "False");
    int seepage_flag = NA_NameToIndexExitOnError(switch_na, switch_name, key);

    if (!seepage_flag)
    {
      continue;
    }

    int patch_id = NA_NameToIndex(GlobalsGeometries[domain_index]->patches, patch_name);
    if (patch_id < 0)
    {
      amps_Printf("Invalid patch name <%s> in Patch.%s.BCPressure.Seepage\n", patch_name, patch_name);
      NA_InputError(GlobalsGeometries[domain_index]->patches, patch_name, "");
    }

    /* patch_id is a seepage patch */
    seepage->seepage_patches[patch_id] = 1;
  }

  NA_FreeNameArray(patches_na);
  NA_FreeNameArray(switch_na);
}


void SeepageLookupFree(SeepageLookup *seepage)
{
  if (seepage->seepage_patches)
  {
    tfree(seepage->seepage_patches);
  }
}
