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
/*****************************************************************************
*
* Routines for reading grid used to specify how the problem is
* distributed to processors.
*
*
*****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * ReadProcessSubgrid
 *--------------------------------------------------------------------------*/

Subgrid    *ReadProcessSubgrid(int subgrid_num)
{
  char base_name[IDB_MAX_KEY_LEN-4];
  char key_name[IDB_MAX_KEY_LEN];

  sprintf(base_name, "ProcessGrid.%i", subgrid_num);

  sprintf(key_name, "%s.P", base_name);
  int p = GetInt(key_name);

  sprintf(key_name, "%s.IX", base_name);
  int ix = GetInt(key_name);
  sprintf(key_name, "%s.IY", base_name);
  int iy = GetInt(key_name);
  sprintf(key_name, "%s.IZ", base_name);
  int iz = GetInt(key_name);

  sprintf(key_name, "%s.RX", base_name);
  int rx = GetIntDefault(key_name, 0);
  sprintf(key_name, "%s.RY", base_name);
  int ry = GetIntDefault(key_name, 0);
  sprintf(key_name, "%s.RZ", base_name);
  int rz = GetIntDefault(key_name, 0);

  sprintf(key_name, "%s.NX", base_name);
  int nx = GetInt(key_name);
  sprintf(key_name, "%s.NY", base_name);
  int ny = GetInt(key_name);
  sprintf(key_name, "%s.NZ", base_name);
  int nz = GetInt(key_name);

  return NewSubgrid(ix, iy, iz, nx, ny, nz, rx, ry, rz, p);
}

/*--------------------------------------------------------------------------
 * ReadProcessGrid
 *--------------------------------------------------------------------------*/

Grid      *ReadProcessGrid()
{
  int num_user_subgrids = GetIntDefault("ProcessGrid.NumSubgrids", 0);

  if (num_user_subgrids > 0)
  {
    int i;

    /* read user_subgrids */
    SubgridArray  *process_all_subgrids = NewSubgridArray();
    SubgridArray  *process_subgrids = NewSubgridArray();
    for (i = 0; i < num_user_subgrids; i++)
    {
      Subgrid* subgrid = ReadProcessSubgrid(i);
      AppendSubgrid(subgrid, process_all_subgrids);

      if (amps_Rank() == SubgridProcess(subgrid))
      {
        AppendSubgrid(subgrid, process_subgrids);
      }
    }

    /* create user_grid */
    return NewGrid(process_subgrids, process_all_subgrids);
  }
  else
  {
    return NULL;
  }
}


/*--------------------------------------------------------------------------
 * FreeUserGrid
 *--------------------------------------------------------------------------*/

void  FreeProcessGrid(
                      Grid *process_grid)
{
  FreeGrid(process_grid);
}

