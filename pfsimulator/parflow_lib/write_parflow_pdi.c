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
  
  //BeginTiming(PFBTimingIndex);
  
  /* Current rank */
  p = amps_Rank(amps_CommWorld);
  
  /* Compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }
  
  // File name to be passed to PDI
  char filename[32] = "";
  sprintf(filename, "%s.%s", file_prefix, file_suffix);
  
  if (p == 0)
  {
    /* Size of the Vector structure for PDI */
    int vector_pdi_buffer_size = sizeof(Vector);
    /* disp set dynamically for PDI due to SHMEM_OBJECTS */
    int grid_disp       = ((char*)&(v->grid)) - ((char*)v);
    int data_space_disp = ((char*)&(v->data_space)) - ((char*)v);
    int size_disp       = ((char*)&(v->size)) - ((char*)v);
      
    printf("Size of Vector: %ld\n", sizeof(Vector));
    printf("Size of Grid: %ld\n", sizeof(Grid));
    printf("Data size until **subvectors: %ld\n", ((char*)&(v->subvectors)) - (char*)v);
    printf("Data size until data_size: %ld\n", ((char*)&(v->data_size)) - ((char*)v));
    printf("Data size until grid: %d\n", grid_disp);
    printf("Data size until grid: %d\n", data_space_disp);
    printf("Data size until size: %d\n", size_disp);
    // printf("Data size until data_size: %ld\n", ((char*)&vv)-((char*)&vv.data_size));
    // ((char*)&a_struct)-((char*)&a_struct.grid)
    
    // load the configuration tree
    PC_tree_t conf = PC_parse_path("conf.yml");
    PDI_init(conf);
    
    PDI_expose("it", &iteration, PDI_OUT);
    PDI_expose("parflowrank", &p, PDI_OUT);
    
    PDI_expose("filename", &filename, PDI_OUT);
    
    PDI_expose("X", &BackgroundX(GlobalsBackground), PDI_OUT);
    PDI_expose("Y", &BackgroundY(GlobalsBackground), PDI_OUT);
    PDI_expose("Z", &BackgroundY(GlobalsBackground), PDI_OUT);
    
    PDI_expose("NX", &SubgridNX(GridBackground(grid)), PDI_OUT);
    PDI_expose("NY", &SubgridNY(GridBackground(grid)), PDI_OUT);
    PDI_expose("NZ", &SubgridNZ(GridBackground(grid)), PDI_OUT);
    
    PDI_expose("DX", &BackgroundDX(GlobalsBackground), PDI_OUT);
    PDI_expose("DY", &BackgroundDY(GlobalsBackground), PDI_OUT);
    PDI_expose("DZ", &BackgroundDZ(GlobalsBackground), PDI_OUT);
    
    PDI_expose("num_subgrids", &num_subgrids, PDI_OUT);
    
    subvector = VectorSubvector(v, 0);
    int temp_data_size = SubvectorDataSize(subvector);
    PDI_expose("temp_data_size", &temp_data_size, PDI_OUT);
    
    int num_grid = 1;
    PDI_expose("num_grid", &num_grid, PDI_OUT);
    
    PDI_expose("vector_pdi_buffer_size", &vector_pdi_buffer_size, PDI_OUT);
    PDI_expose("grid_disp",       &grid_disp, PDI_OUT);
    PDI_expose("data_space_disp", &data_space_disp, PDI_OUT);
    PDI_expose("size_disp",       &size_disp, PDI_OUT);
    
    PDI_expose("vector_data", v, PDI_OUT);
    
    // finalize PDI
    PDI_finalize();
  }
  
  //EndTiming(PFBTimingIndex);
  
}
