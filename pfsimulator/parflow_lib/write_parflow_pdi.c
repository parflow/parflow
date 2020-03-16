/*BHEADER*********************************************************************
 *
 *  This file is part of Parflow. For details, see
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
* Routines to write a Vector to a file in full or scattered form using PDI.
*
*****************************************************************************/

#include "parflow.h"
#ifdef HAVE_PDI
#include <pdi.h>
#include <paraconf.h>
#endif
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

#ifdef HAVE_PDI
static bool is2Ddefined = false;
static bool is3Ddefined = false;
static bool isTdefined = false;
#endif

void     WritePDI(
                       char *  file_prefix,
                       char *  file_suffix,
                       Vector *v)
{
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;
  
  int g;
  int p;

  long size;

  char file_extn[7] = "pfb";
  char filename[255];
  amps_File file;
  
  BeginTiming(PFBTimingIndex);
  
  p = amps_Rank(amps_CommWorld);

  if (p == 0)
    size = 6 * amps_SizeofDouble + 4 * amps_SizeofInt;
  else
    size = 0;
    
  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    size += SizeofPFBinarySubvector(subvector, subgrid);
  }
  
  /* open file */
  sprintf(filename, "%s.%s", file_prefix, file_suffix);
  
  /* Compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }
  
  if (p == 0)
  {
    FILE *fp;
    fp = fopen("conf.yml", "w");
    fprintf(fp, "pdi:\n");
    fprintf(fp, "  metadata:\n");
    fprintf(fp, "  X: double\n");
    fprintf(fp, "  Y: double\n");
    fprintf(fp, "  Z: double\n");
    fclose(fp);
    
    // load the configuration tree
    PC_tree_t conf = PC_parse_path("conf.yml");
    PDI_init(PC_get(conf, ".pdi"));
    
    PDI_share("X", &BackgroundX(GlobalsBackground), PDI_OUT);
    PDI_share("Y", &BackgroundY(GlobalsBackground), PDI_OUT);
    PDI_share("Z", &BackgroundY(GlobalsBackground), PDI_OUT);
  }
  
  EndTiming(PFBTimingIndex);
  
}
