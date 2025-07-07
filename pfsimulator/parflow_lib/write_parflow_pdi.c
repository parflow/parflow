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
* Routines to write data Vector to a file in full or scattered form using PDI.
*
*****************************************************************************/

#include "parflow.h"
#ifdef PARFLOW_HAVE_PDI
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


/**
 * @brief Writes vector data to a PDI (Parallel Data Interface) output file.
 *
 * This function generates an output file in HDF5 format using PDI, exposing relevant
 * vector and grid data. It is primarily used for exporting simulation data from
 * the ParFlow framework. The function determines the number of subgrids, constructs
 * a filename, and exposes various metadata and vector attributes to PDI.
 *
 * @param file_prefix    Prefix for the output filename.
 * @param file_suffix    Suffix for the output filename.
 * @param file_number    Numerical identifier for the file.
 * @param v              Pointer to the Vector structure containing data to write.
 * @param with_tolerance tolerance.
 * @param drop_tolerance Value of drop tolerance.
 *
 * @warning The function relies on PDI and requires a valid PDI configuration (`conf.yml`).
 *          Ensure PDI is correctly initialized and available in the environment.
 *
 * @see PDI_expose(), PDI_init(), PDI_finalize() at https://pdi.dev/1.8/modules.html
 */

void     WritePDI(
                  char *  file_prefix,
                  char *  file_suffix,
                  int     file_number,
                  Vector *v,
                  int     with_tolerance,
                  double  drop_tolerance)
{
#ifdef PARFLOW_HAVE_PDI
  Grid           *grid = VectorGrid(v);
  Subvector      *subvector;

  int p;

  /* start PDI timer */
  BeginTiming(PDITimingIndex);

  /* current rank */
  p = amps_Rank(amps_CommWorld);
  /* compute number of patches to write */
  int num_subgrids = GridNumSubgrids(grid);
  {
    amps_Invoice invoice = amps_NewInvoice("%i", &num_subgrids);

    amps_AllReduce(amps_CommWorld, invoice, amps_Add);

    amps_FreeInvoice(invoice);
  }

  /* file name to be passed to PDI */
  char filename[255] = "";
  sprintf(filename, "%s.%s.h5.%d", file_prefix, file_suffix, p);
  /* size of the Vector structure */
  int vector_pdi_buffer_size = sizeof(Vector);
  /* disp set dynamically for PDI due to SHMEM_OBJECTS */
  int grid_disp = ((char*)&(v->grid)) - ((char*)v);
  int data_space_disp = ((char*)&(v->data_space)) - ((char*)v);
  int size_disp = ((char*)&(v->size)) - ((char*)v);
  /* load PDI Specification tree */
  PC_tree_t conf = PC_parse_path("conf.yml");

  /* initialize PDI */
  PDI_init(conf);

  PDI_expose("parflowrank", &p, PDI_OUT);
  PDI_expose("filename", &filename, PDI_OUT);
  PDI_expose("file_number", &file_number, PDI_OUT);
  PDI_expose("drop_tolerance", &drop_tolerance, PDI_OUT);
  PDI_expose("with_tolerance", &with_tolerance, PDI_OUT);
  PDI_expose("X", &BackgroundX(GlobalsBackground), PDI_OUT);
  PDI_expose("Y", &BackgroundY(GlobalsBackground), PDI_OUT);
  PDI_expose("Z", &BackgroundZ(GlobalsBackground), PDI_OUT);
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
  PDI_expose("grid_disp", &grid_disp, PDI_OUT);
  PDI_expose("data_space_disp", &data_space_disp, PDI_OUT);
  PDI_expose("size_disp", &size_disp, PDI_OUT);
  PDI_expose("sparse_vector_data", v, PDI_OUT);

  /* finalize PDI */
  PDI_finalize();

  /* stop PDI timer */
  EndTiming(PDITimingIndex);
#endif
}
