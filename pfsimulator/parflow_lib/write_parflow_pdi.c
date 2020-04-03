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
    Vector vv;
    printf("Size of Vector: %ld\n", sizeof(Vector));
    printf("Size of Grid: %ld\n", sizeof(Grid));
    printf("Data size until **subvectors: %ld\n", ((char*)&v)-((char*)&v->subvectors[0][0]));
    printf("Data size until data_size: %ld\n", ((char*)&vv)-((char*)&vv.data_size));
    printf("Data size until size: %ld\n", ((char*)&vv)-((char*)&vv.size));
    // printf("Data size until data_size: %ld\n", ((char*)&vv)-((char*)&vv.data_size));
    // ((char*)&a_struct)-((char*)&a_struct.grid)
    
    
    
    const char pdi_conf[] = "\
.types:\n\
  - &ComputePkg\n\
    type: record\n\
  - &Subregion\n\
    type: record\n\
    buffersize: 56\n\
    members:\n\
      ix:      {disp: 0, type: int}\n\
      iy:      {disp: 4, type: int}\n\
      iz:      {disp: 8, type: int}\n\
      nx:      {disp: 12, type: int}\n\
      ny:      {disp: 16, type: int}\n\
      nz:      {disp: 20, type: int}\n\
      sx:      {disp: 24, type: int}\n\
      sy:      {disp: 28, type: int}\n\
      sz:      {disp: 32, type: int}\n\
      rx:      {disp: 36, type: int}\n\
      ry:      {disp: 40, type: int}\n\
      rz:      {disp: 44, type: int}\n\
      level:   {disp: 48, type: int}\n\
      process: {disp: 52, type: int}\n\
  - &Subvector\n\
    type: record \n\
    buffersize: 32\n\
    members:\n\
      data:\n\
        disp: 0\n\
        type: pointer\n\
        subtype:\n\
          type: array\n\
          size: $temp_data_size\n\
          subtype: double\n\
      allocated:\n\
        disp: 8\n\
        type: int\n\
      data_space:\n\
        disp: 16\n\
        type: pointer\n\
        subtype: *Subregion\n\
      data_size:\n\
        disp: 24\n\
        type: int\n\
  - &SubregionArray\n\
    type: record\n\
    buffersize: 16\n\
    members:\n\
      subregions:\n\
        disp: 0\n\
        type: pointer\n\
        subtype:\n\
          type: array\n\
          size: $temp_num_grid\n\
          subtype:\n\
            type: pointer\n\
            subtype: *Subregion\n\
      size:\n\
        disp: 8\n\
        type: int\n\
  - &Grid\n\
    type: record\n\
    buffersize: 40\n\
    members:\n\
      subgrids:\n\
        disp: 0\n\
        type: pointer\n\
        subtype:\n\
          type: array\n\
          size: 1\n\
          subtype: *SubregionArray\n\
      all_subgrids:\n\
        disp: 8\n\
        type: pointer\n\
        subtype:\n\
          type: array\n\
          size: 1\n\
          subtype: *SubregionArray\n\
      size:\n\
        disp: 16\n\
        type: int\n\
logging: trace\n\
metadata:\n\
  it: int\n\
  temp_data_size: int\n\
  temp_num_grid: int\n\
  postfix:  { type: array, subtype: char, size: 32 }\n\
  X: double\n\
  Y: double\n\
  Z: double\n\
  NX: double\n\
  NY: double\n\
  NZ: double\n\
  num_subgrids: int\n\
data:\n\
  vector_data:\n\
    type: record\n\
    buffersize: 128\n\
    members:\n\
      subvectors:\n\
        disp: 0\n\
        type: pointer\n\
        subtype:\n\
          type: array\n\
          size: 1\n\
          subtype:\n\
            type: pointer\n\
            subtype: *Subvector\n\
      data_size:\n\
        disp: 8\n\
        type: int\n\
";
// compute_pkgs:\n\
//   disp: 16\n\
//   type: pointer\n\
//   subtype:\n\
//     type: array\n\
//     size: 1\n\
//     subtype: *ComputePkg\n\
// background:\n
FILE *fp;
fp = fopen("conf.yml", "w");
fprintf(fp, pdi_conf);
#ifdef SHMEM_OBJECTS
  fprintf(fp,"      int: shmem_offset\n");
#endif
const char pdi_conf_2[] = "\
      grid:\n\
        disp: 16\n\
        type: pointer\n\
        subtype: *Grid\n\
      data_space:\n\
        disp: 24\n\
        type: pointer\n\
        subtype: *SubregionArray\n\
      size:\n\
        disp: 32\n\
        type: int\n\
plugins:   \n\
  decl_hdf5:   \n\
  - file: test.${it}.h5   \n\
    write: [ X, Y, Z, NX, NY, NZ, num_subgrids, temp_data_size, temp_num_grid, vector_data ]   \n\
";
    fprintf(fp, pdi_conf_2);
    //fprintf(fp, "  mpi:   \n");
    //fprintf(fp, "    communicator: $MPI_COMM_WORLD   \n");
    fclose(fp);
    
    // load the configuration tree
    PC_tree_t conf = PC_parse_path("conf.yml");
    PDI_init(conf);
    
    PDI_expose("it", &iteration, PDI_OUT);
    
    PDI_expose("postfix", &filename, PDI_OUT);
    
    PDI_expose("X", &BackgroundX(GlobalsBackground), PDI_OUT);
    PDI_expose("Y", &BackgroundY(GlobalsBackground), PDI_OUT);
    PDI_expose("Z", &BackgroundY(GlobalsBackground), PDI_OUT);
    
    PDI_expose("NX", &SubgridNX(GridBackground(grid)), PDI_OUT);
    PDI_expose("NY", &SubgridNY(GridBackground(grid)), PDI_OUT);
    PDI_expose("NZ", &SubgridNZ(GridBackground(grid)), PDI_OUT);
    
    PDI_expose("num_subgrids", &num_subgrids, PDI_OUT);
    
    subvector = VectorSubvector(v, 0);
    int temp_data_size = SubvectorDataSize(subvector);
    PDI_expose("temp_data_size", &temp_data_size, PDI_OUT);
    
    int temp_num_grid = 1;
    PDI_expose("temp_num_grid", &temp_num_grid, PDI_OUT);
    
    PDI_expose("vector_data", v, PDI_OUT);
    
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
