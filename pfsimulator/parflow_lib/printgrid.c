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
* Print the grid structure to a file
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * PrintGrid
 *--------------------------------------------------------------------------*/

void   PrintGrid(
                 char *filename,
                 Grid *grid)
{
  amps_File file;

  SubgridArray  *s_a;
  Subgrid       *s;
  int i;


  if ((file = amps_Fopen(filename, "a")) == NULL)
  {
    amps_Printf("Error: can't open output file %s\n", filename);
    exit(1);
  }

  amps_Fprintf(file, "=================================================\n");

  /* print background grid info */
  amps_Fprintf(file, "Background Grid:\n");
  amps_Fprintf(file, "( X,  Y,  Z) = (%f, %f, %f)\n",
               BackgroundX(GlobalsBackground),
               BackgroundY(GlobalsBackground),
               BackgroundZ(GlobalsBackground));
  amps_Fprintf(file, "(NX, NY, NZ) = (%d, %d, %d)\n",
               BackgroundNX(GlobalsBackground),
               BackgroundNY(GlobalsBackground),
               BackgroundNZ(GlobalsBackground));
  amps_Fprintf(file, "(DX, DY, DZ) = (%f, %f, %f)\n",
               BackgroundDX(GlobalsBackground),
               BackgroundDY(GlobalsBackground),
               BackgroundDZ(GlobalsBackground));

  /* print subgrids */
  ForSubgridI(i, (s_a = GridSubgrids(grid)))
  {
    s = SubgridArraySubgrid(s_a, i);

    amps_Fprintf(file, "Subgrids(%d):\n", i);
    amps_Fprintf(file, "( x,  y,  z) = (%d, %d, %d)\n",
                 SubgridIX(s), SubgridIY(s), SubgridIZ(s));
    amps_Fprintf(file, "(nx, ny, nz) = (%d, %d, %d)\n",
                 SubgridNX(s), SubgridNY(s), SubgridNZ(s));

    amps_Fprintf(file, "(rx, ry, rz) = (%d, %d, %d)\n",
                 SubgridRX(s), SubgridRY(s), SubgridRZ(s));
    amps_Fprintf(file, "process = %d\n", SubgridProcess(s));
  }

  /* print all_subgrids */
  ForSubgridI(i, (s_a = GridAllSubgrids(grid)))
  {
    s = SubgridArraySubgrid(s_a, i);

    amps_Fprintf(file, "AllSubgrids(%d):\n", i);
    amps_Fprintf(file, "( x,  y,  z) = (%d, %d, %d)\n",
                 SubgridIX(s), SubgridIY(s), SubgridIZ(s));
    amps_Fprintf(file, "(nx, ny, nz) = (%d, %d, %d)\n",
                 SubgridNX(s), SubgridNY(s), SubgridNZ(s));
    amps_Fprintf(file, "(rx, ry, rz) = (%d, %d, %d)\n",
                 SubgridRX(s), SubgridRY(s), SubgridRZ(s));
    amps_Fprintf(file, "process = %d\n", SubgridProcess(s));
  }

  amps_Fprintf(file, "=================================================\n");

  fflush(file);
  amps_Fclose(file);
}
