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
 **********************************************************************EHEADER*/
/*****************************************************************************
*
*****************************************************************************/

#include <math.h>
#include <assert.h>

#include "parflow.h"
#include "amps.h"
#include "communication.h"

#ifdef HAVE_P4EST
#include "parflow_p4est_dependences.h"
#endif

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

#ifdef HAVE_P4EST
static int ComputeTag(Subregion *sender, Subregion *receiver)
{
  int k;
  int tag;
  int gidx = SubgridGhostIdx(sender);
  int sender_lidx;
  const int dim = GlobalsNumProcsZ > 1 ? 3 : 2;
  p4est_gloidx_t scoord[dim], rcoord[dim];
  int t[3] = { 0, 0, 0 };

  parflow_p4est_get_brick_coord(sender, globals->grid3d->pfgrid, scoord);
  parflow_p4est_get_brick_coord(receiver, globals->grid3d->pfgrid, rcoord);

  for (k = 0; k < dim; k++)
    t[k] = parflow_p4est_int_compare(scoord[k], rcoord[k]);

  tag = 9 * t[2] + 3 * t[1] + t[0];

  sender_lidx = (gidx < -1) ?
              (-2-gidx) / (1 << dim) : SubgridLocIdx(sender);

  tag += 27 * sender_lidx;

  return tag;
}

static double* FetchData(void *container, parflow_p4est_container_type_t ctype,
                         int s_idx)
{
    double *data;
    Vector *V;
    Matrix *M;

    switch (ctype)
    {
    case parflow_p4est_vector:
        V = (Vector *) container;
        M = NULL;
        data = SubvectorData(VectorSubvector(V, s_idx));
        break;

    case parflow_p4est_matrix:
        V = NULL;
        M = (Matrix *) container;
        data = SubmatrixData(MatrixSubmatrix(M, s_idx));
        break;

    default:
        SC_ABORT_NOT_REACHED ();
    }

    return  data;
}
#endif

/*--------------------------------------------------------------------------
 * NewCommPkg:
 *   `send_region' and `recv_region' are "regions" of `grid'.
 *--------------------------------------------------------------------------*/

CommPkg         *NewCommPkg(
                            Region *        send_region,
                            Region *        recv_region,
                            SubregionArray *data_space,
                            int             s_idx,
                            int             num_vars, /* number of variables in the vector */
                            double *        data
#ifndef HAVE_P4EST
                            )
#else
                            ,
                            void *numericalObject, parflow_p4est_container_type_t ctype
                            )
#endif
{
  CommPkg         *new_comm_pkg;

  amps_Invoice invoice;

  SubregionArray  *send_sra = RegionSubregionArray(send_region, s_idx);
  SubregionArray  *recv_sra = RegionSubregionArray(recv_region, s_idx);

  Subregion       *send_sr, *recv_sr;
  Subregion       *data_sr;

  int  *loop_array;

  int num_send_subregions;
  int num_recv_subregions;

  int j, p;

  int dim;

#ifdef HAVE_P4EST
  int t[3];
  int tag;
  int which_child, which_ghost;
  double * g_data;
  parflow_p4est_grid_t *pfgrid = globals->grid3d->pfgrid;
  Subregion *g_data_sr;
#endif

  new_comm_pkg = ctalloc(CommPkg, 1);

  /*------------------------------------------------------
   * compute num send and recv subregions
   *------------------------------------------------------*/

  num_send_subregions = 0;
  num_recv_subregions = 0;
  if (SubregionArraySize(data_space))
  {
    assert(s_idx >= 0 && s_idx < SubregionArraySize(data_space));
    num_send_subregions += SubregionArraySize(send_sra);
    num_recv_subregions += SubregionArraySize(recv_sra);
  }

  /*------------------------------------------------------
   * Set up CommPkg
   *------------------------------------------------------*/

  if (num_send_subregions || num_recv_subregions)
    loop_array = (new_comm_pkg->loop_array)
                   = talloc(int, (num_send_subregions + num_recv_subregions) * 9);

  /* set up send info */
  if (num_send_subregions)
  {
    new_comm_pkg->send_ranks =
      talloc(int, num_send_subregions);
    new_comm_pkg->num_send_invoices = num_send_subregions;
    new_comm_pkg->send_invoices =
      ctalloc(amps_Invoice, num_send_subregions);

    data_sr = SubregionArraySubregion(data_space, s_idx);

    p = 0;
    ForSubregionI(j, send_sra)
    {
      send_sr = SubregionArraySubregion(send_sra, j);
      new_comm_pkg->send_ranks[p] = SubregionProcess(send_sr);

      dim = NewCommPkgInfo(data_sr, send_sr, 0, num_vars,
                           loop_array);
      invoice =
        amps_NewInvoice("%&.&D(*)",
                        loop_array + 1,
                        loop_array + 5,
                        dim,
                        data + loop_array[0]);

      if (USE_P4EST)
      {
#ifdef HAVE_P4EST
        BeginTiming(P4ESTSetupTimingIndex);
        tag = ComputeTag(data_sr, send_sr);
        amps_SetInvoiceTag(invoice, tag);
        EndTiming(P4ESTSetupTimingIndex);
#else
        PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
      }

      amps_AppendInvoice(&(new_comm_pkg->send_invoices[p]),
                         invoice);
      p++;
      loop_array += 9;
    }
    assert(p == num_send_subregions);
  }

  /* set up recv info */
  if (num_recv_subregions)
  {
    new_comm_pkg->recv_ranks = talloc(int, num_recv_subregions);
    new_comm_pkg->num_recv_invoices = num_recv_subregions;
    new_comm_pkg->recv_invoices =
      ctalloc(amps_Invoice, num_recv_subregions);

    data_sr = SubregionArraySubregion(data_space, s_idx);

    p = 0;
    ForSubregionI(j, recv_sra)
    {
      recv_sr = SubregionArraySubregion(recv_sra, j);
      new_comm_pkg->recv_ranks[p] = SubregionProcess(recv_sr);

      which_ghost = -1;
#ifdef HAVE_P4EST
      if (SubgridLevel(data_sr) < SubgridLevel(recv_sr))
      {
          which_child = parflow_p4est_check_neigh(recv_sr, data_sr, pfgrid);
          which_ghost = data_sr->ghostChildren[which_child];
          g_data_sr = SubregionArraySubregion(data_space, which_ghost);
          g_data = FetchData(numericalObject,ctype, which_ghost);
      }
#endif

      dim = NewCommPkgInfo(which_ghost > 0 ? g_data_sr : data_sr,
                           recv_sr, 0, num_vars,
                           loop_array);
      invoice =
        amps_NewInvoice("%&.&D(*)",
                        loop_array + 1,
                        loop_array + 5,
                        dim,
                        (which_ghost > 0 ? g_data : data) + loop_array[0]);

      if (USE_P4EST)
      {
#ifdef HAVE_P4EST
        BeginTiming(P4ESTSetupTimingIndex);
        tag = ComputeTag(recv_sr, data_sr);
        amps_SetInvoiceTag(invoice, tag);
        EndTiming(P4ESTSetupTimingIndex);
#else
        PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
      }

      amps_AppendInvoice(&(new_comm_pkg->recv_invoices[p]),
                         invoice);
      p++;
      loop_array += 9;
    }
    assert(p == num_recv_subregions);
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
  return (CommHandle*)amps_IExchangePackage(comm_pkg->package);
}


/*--------------------------------------------------------------------------
 * FinalizeCommunication:
 *--------------------------------------------------------------------------*/

void         FinalizeCommunication(
                                   CommHandle *handle)
{
  (void)amps_Wait((amps_Handle)handle);
}



