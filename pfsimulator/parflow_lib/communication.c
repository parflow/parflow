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
*****************************************************************************/

#include <math.h>

#include "parflow.h"
#include "amps.h"
#include "communication.h"

/*--------------------------------------------------------------------------
 * NewCommPkgInfo:
 *   Sets up `offset', `len_array', and `stride_array' in `loop_array'.
 *   Assumes that `comm_sub' is contained in `data_sub' (i.e. `comm_sub' is
 *   larger than `data_sub'), and that `comm_sub' and `data_sub' live on
 *   the same index space.
 *--------------------------------------------------------------------------*/

int  NewCommPkgInfo(
                    Subregion *data_sr,
                    Subregion *comm_sr,
                    int        index,
                    int        num_vars, /* number of variables in the vector */
                    int *      loop_array)
{
  int    *offset = loop_array;
  int    *len_array = loop_array + 1;
  int    *stride_array = loop_array + 5;

  int sa[4];

  int ix, iy, iz;
  int nx, ny, nz;
  int sx, sy, sz;

  int i, j, dim;


  ix = SubregionIX(data_sr);
  iy = SubregionIY(data_sr);
  iz = SubregionIZ(data_sr);

  nx = SubregionNX(data_sr);
  ny = SubregionNY(data_sr);
  nz = SubregionNZ(data_sr);

  sx = SubregionSX(data_sr);
  sy = SubregionSY(data_sr);
  sz = SubregionSZ(data_sr);

  offset[0] = (((SubregionIX(comm_sr) - ix) / sx) +
               ((SubregionIY(comm_sr) - iy) / sy) * nx +
               ((SubregionIZ(comm_sr) - iz) / sz) * nx * ny +
               index * num_vars * nx * ny * nz);

  sx = SubregionSX(comm_sr) / sx;
  sy = SubregionSY(comm_sr) / sy;
  sz = SubregionSZ(comm_sr) / sz;

  len_array[0] = SubregionNX(comm_sr);
  len_array[1] = SubregionNY(comm_sr);
  len_array[2] = SubregionNZ(comm_sr);
  len_array[3] = num_vars;

  sa[0] = (sx);
  sa[1] = (sy) * (nx);
  sa[2] = (sz) * (ny) * (nx);
  sa[3] = (1) * (nz) * (ny) * (nx);

  /* eliminate dimensions with len_array = 1 */
  dim = 4;
  for (i = 0; i < dim; i++)
  {
    if (len_array[i] == 1)
    {
      for (j = i; j < 3; j++)
      {
        len_array[j] = len_array[j + 1];
        sa[j] = sa[j + 1];
      }
      dim--;
    }
  }

#if 0
  /* sort the array according to len_array (largest to smallest) */
  for (i = (dim - 1); i > 0; i--)
    for (j = 0; j < i; j++)
      if (len_array[j] < len_array[j + 1])
      {
        i_tmp = len_array[j];
        len_array[j] = len_array[j + 1];
        len_array[j + 1] = i_tmp;

        i_tmp = sa[j];
        sa[j] = sa[j + 1];
        sa[j + 1] = i_tmp;
      }
#endif

  /* if every len was 1 we need to fix to communicate at least one */
  if (!dim)
    dim++;

  for (i = 0; i < dim; i++)
  {
    stride_array[i] = sa[i];
    for (j = 0; j < i; j++)
      stride_array[i] -= sa[j] * (len_array[j] - 1);
  }

  return dim;
}


/*--------------------------------------------------------------------------
 * NewCommPkg:
 *   `send_region' and `recv_region' are "regions" of `grid'.
 *--------------------------------------------------------------------------*/

CommPkg         *NewCommPkg(
                            Region *        send_region,
                            Region *        recv_region,
                            SubregionArray *data_space,
                            int             num_vars, /* number of variables in the vector */
                            double *        data)
{
  CommPkg         *new_comm_pkg;

  amps_Invoice invoice;

  SubregionArray  *comm_sra;
  Subregion       *comm_sr;

  Subregion       *data_sr;

  int  *loop_array;

  int  *send_proc_array = NULL;
  int  *recv_proc_array = NULL;

  int num_send_subregions;
  int num_recv_subregions;

  int num_send_procs;
  int num_recv_procs;

  int proc;
  int i, j, p;

  int dim;

  new_comm_pkg = ctalloc(CommPkg, 1);

  /*------------------------------------------------------
   * compute num send and recv subregions
   *------------------------------------------------------*/

  num_send_subregions = 0;
  num_recv_subregions = 0;
  ForSubregionI(i, data_space)
  {
    num_send_subregions +=
      SubregionArraySize(RegionSubregionArray(send_region, i));
    num_recv_subregions +=
      SubregionArraySize(RegionSubregionArray(recv_region, i));
  }

  /*------------------------------------------------------
   * compute num send and recv processes and proc_array's
   *------------------------------------------------------*/

  num_send_procs = 0;
  if (num_send_subregions)
  {
    new_comm_pkg->send_ranks = send_proc_array =
      talloc(int, num_send_subregions);

    for (i = 0; i < num_send_subregions; i++)
      send_proc_array[i] = -1;

    ForSubregionArrayI(i, send_region)
    {
      comm_sra = RegionSubregionArray(send_region, i);
      ForSubregionI(j, comm_sra)
      {
        comm_sr = SubregionArraySubregion(comm_sra, j);
        proc = SubregionProcess(comm_sr);

        for (p = 0; p < num_send_procs; p++)
          if (proc == send_proc_array[p])
            break;
        if (p >= num_send_procs)
          send_proc_array[num_send_procs++] = proc;
      }
    }
  }

  num_recv_procs = 0;
  if (num_recv_subregions)
  {
    new_comm_pkg->recv_ranks =
      recv_proc_array = talloc(int, num_recv_subregions);

    for (i = 0; i < num_recv_subregions; i++)
      recv_proc_array[i] = -1;

    ForSubregionArrayI(i, recv_region)
    {
      comm_sra = RegionSubregionArray(recv_region, i);
      ForSubregionI(j, comm_sra)
      {
        comm_sr = SubregionArraySubregion(comm_sra, j);
        proc = SubregionProcess(comm_sr);

        for (p = 0; p < num_recv_procs; p++)
          if (proc == recv_proc_array[p])
            break;
        if (p >= num_recv_procs)
          recv_proc_array[num_recv_procs++] = proc;
      }
    }
  }

  /*------------------------------------------------------
   * Set up CommPkg
   *------------------------------------------------------*/

  if (num_send_procs || num_recv_procs)
    loop_array = (new_comm_pkg->loop_array)
                   = talloc(int, (num_send_subregions + num_recv_subregions) * 9);

  /* set up send info */
  if (num_send_procs)
  {
    new_comm_pkg->num_send_invoices = num_send_procs;
    (new_comm_pkg->send_invoices) =
      ctalloc(amps_Invoice, num_send_procs);

    for (p = 0; p < num_send_procs; p++)
    {
      num_send_subregions = 0;
      ForSubregionI(i, data_space)
      {
        data_sr = SubregionArraySubregion(data_space, i);
        comm_sra = RegionSubregionArray(send_region, i);

        ForSubregionI(j, comm_sra)
        {
          comm_sr = SubregionArraySubregion(comm_sra, j);

          if (SubregionProcess(comm_sr) == send_proc_array[p])
          {
            dim = NewCommPkgInfo(data_sr, comm_sr, i, num_vars,
                                 loop_array);

            invoice =
              amps_NewInvoice("%&.&D(*)",
                              loop_array + 1,
                              loop_array + 5,
                              dim,
                              data + loop_array[0]);

            amps_AppendInvoice(&(new_comm_pkg->send_invoices[p]),
                               invoice);

            num_send_subregions++;
            loop_array += 9;
          }
        }
      }
    }
  }

  /* set up recv info */
  if (num_recv_procs)
  {
    new_comm_pkg->num_recv_invoices = num_recv_procs;
    (new_comm_pkg->recv_invoices) =
      ctalloc(amps_Invoice, num_recv_procs);

    for (p = 0; p < num_recv_procs; p++)
    {
      num_recv_subregions = 0;
      ForSubregionI(i, data_space)
      {
        data_sr = SubregionArraySubregion(data_space, i);
        comm_sra = RegionSubregionArray(recv_region, i);

        ForSubregionI(j, comm_sra)
        {
          comm_sr = SubregionArraySubregion(comm_sra, j);

          if (SubregionProcess(comm_sr) == recv_proc_array[p])
          {
            dim = NewCommPkgInfo(data_sr, comm_sr, i, num_vars,
                                 loop_array);

            invoice =
              amps_NewInvoice("%&.&D(*)",
                              loop_array + 1,
                              loop_array + 5,
                              dim,
                              data + loop_array[0]);

            amps_AppendInvoice(&(new_comm_pkg->recv_invoices[p]),
                               invoice);

            num_recv_subregions++;
            loop_array += 9;
          }
        }
      }
    }
  }

  new_comm_pkg->package = amps_NewPackage(amps_CommWorld,
                                          new_comm_pkg->num_send_invoices,
                                          new_comm_pkg->send_ranks,
                                          new_comm_pkg->send_invoices,
                                          new_comm_pkg->num_recv_invoices,
                                          new_comm_pkg->recv_ranks,
                                          new_comm_pkg->recv_invoices);



  return new_comm_pkg;
}


/*--------------------------------------------------------------------------
 * FreeCommPkg:
 *--------------------------------------------------------------------------*/

void FreeCommPkg(
                 CommPkg *pkg)
{
  if (pkg)
  {
    int i;

    amps_FreePackage(pkg->package);

    for (i = pkg->num_send_invoices; i--;)
    {
      amps_FreeInvoice(pkg->send_invoices[i]);
    }

    for (i = pkg->num_recv_invoices; i--;)
    {
      amps_FreeInvoice(pkg->recv_invoices[i]);
    }

    tfree(pkg->send_invoices);
    tfree(pkg->recv_invoices);

    tfree(pkg->send_ranks);
    tfree(pkg->recv_ranks);

    tfree(pkg->loop_array);

    tfree(pkg);
  }
}


/*--------------------------------------------------------------------------
 * InitCommunication:
 *--------------------------------------------------------------------------*/

CommHandle  *InitCommunication(
                               CommPkg *comm_pkg)
{
  PUSH_NVTX("amps_IExchangePackage",6)
  CommHandle* handle = (CommHandle*)amps_IExchangePackage(comm_pkg->package);
  POP_NVTX

  return handle;
}


/*--------------------------------------------------------------------------
 * FinalizeCommunication:
 *--------------------------------------------------------------------------*/

void         FinalizeCommunication(
                                   CommHandle *handle)
{
  PUSH_NVTX("amps_Wait",1)
  (void)amps_Wait((amps_Handle)handle);
  POP_NVTX
}



