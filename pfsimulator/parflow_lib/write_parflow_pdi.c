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
                       int     iteration,
                       Vector *v)
{
  Grid           *grid = VectorGrid(v);
  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *subvector;
  
  int g;
  int p;

  long size;
  
  BeginTiming(PFBTimingIndex);
  
  p = amps_Rank(amps_CommWorld);

  if (p == 0)
    size = 6 * amps_SizeofDouble + 4 * amps_SizeofInt;
  else
    size = 0;
  
  /* Compute total size */
  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(v, g);

    size += SizeofPFBinarySubvector(subvector, subgrid);
  }
  
  /* Compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }
  
  char filename[32] = "";
  /* open file */
  sprintf(filename, "%s.%s", file_prefix, file_suffix);
  
  if (p == 0)
  {
    FILE *fp;
    fp = fopen("conf.yml", "w");
    fprintf(fp, "pdi:\n");
    fprintf(fp, "  metadata:\n");
    fprintf(fp, "    it: int\n");
    fprintf(fp, "    postfix:  { type: array, subtype: char, size: 32 }\n");
    fprintf(fp, "    X: double\n");
    fprintf(fp, "    Y: double\n");
    fprintf(fp, "    Z: double\n");
    fprintf(fp, "    NX: int\n");
    fprintf(fp, "    NY: int\n");
    fprintf(fp, "    NZ: int\n");
    fprintf(fp, "    num_subgrids: int\n");
    fprintf(fp, "  data:\n");
    // fprintf(fp, "    subvector:\n");
    // fprintf(fp, "    type: record\n");
    // fprintf(fp, "    buffersize: 4\n");
    // fprintf(fp, "    members:\n");
    // fprintf(fp, "      size:\n");
    // fprintf(fp, "        disp: 0\n");
    // fprintf(fp, "        type: int\n");
    fprintf(fp, "  plugins:   \n");
    fprintf(fp, "    mpi:   \n");
    fprintf(fp, "    decl_hdf5:   \n");
    fprintf(fp, "    - file: test.${it}.h5   \n");
    fprintf(fp, "      communicator: $MPI_COMM_WORLD   \n");
    fprintf(fp, "      write: [ X, Y, Z, NX, NY, NZ, num_subgrids ]   \n");
    fprintf(fp, "          \n");
    fclose(fp);
    
    // load the configuration tree
    PC_tree_t conf = PC_parse_path("conf.yml");
    PDI_init(PC_get(conf, ".pdi"));
    
    PDI_share("it", &iteration, PDI_OUT);
    PDI_reclaim("it");
    
    PDI_share("postfix", &filename, PDI_OUT);
    PDI_reclaim("postfix");
    
    PDI_share("X", &BackgroundX(GlobalsBackground), PDI_OUT);
    PDI_reclaim("X");
    PDI_share("Y", &BackgroundY(GlobalsBackground), PDI_OUT);
    PDI_reclaim("Y");
    PDI_share("Z", &BackgroundY(GlobalsBackground), PDI_OUT);
    PDI_reclaim("Z");
    
    PDI_share("NX", &SubgridNX(GridBackground(grid)), PDI_OUT);
    PDI_reclaim("NX");
    PDI_share("NY", &SubgridNY(GridBackground(grid)), PDI_OUT);
    PDI_reclaim("NY");
    PDI_share("NZ", &SubgridNZ(GridBackground(grid)), PDI_OUT);
    PDI_reclaim("NZ");
    
    PDI_share("num_subgrids", &num_subgrids, PDI_OUT);
    PDI_reclaim("num_subgrids");
    
    // PDI_share("NX", &SubgridNX(GridBackground(grid)), PDI_OUT);
    // PDI_reclaim("NX");
    // PDI_share("NY", &SubgridNY(GridBackground(grid)), PDI_OUT);
    // PDI_reclaim("NY");
    // PDI_share("NZ", &SubgridNZ(GridBackground(grid)), PDI_OUT);
    // PDI_reclaim("NZ");
    
    // finalize PDI
    PDI_finalize();
  }
  
  EndTiming(PFBTimingIndex);
  
}
