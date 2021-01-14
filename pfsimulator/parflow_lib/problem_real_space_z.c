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

#include "parflow.h"
#include "globals.h"

#ifdef HAVE_P4EST
#include "parflow_p4est_dependences.h"
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
} PublicXtra;

typedef void InstanceXtra;

#ifdef HAVE_P4EST
static int ComputeTag(Subgrid *send_sg, Subgrid *recv_sg)
{
  int tag;
  int tz = parflow_p4est_int_compare(SubregionIZ(send_sg), SubregionIZ(recv_sg));

  if (tz)
  {
    tag = tz < 0 ? 1 : 0;
  }
  else
  {
    PARFLOW_ERROR("Trying to compute invalid tag");
  }

  tag += 2 * SubgridLocIdx(send_sg);

  return tag;
}
#endif

/*--------------------------------------------------------------------------
 * realSpaceZ values
 *--------------------------------------------------------------------------*/
void realSpaceZ(ProblemData *problem_data, Vector *rsz)
{
  Grid           *grid = VectorGrid(rsz);

  SubgridArray   *subgrids = GridSubgrids(grid);
  Subgrid        *subgrid;
  Subvector      *dz_sub;
  Subvector      *rsz_sub;

  amps_Invoice invoice;
  VectorUpdateCommHandle       *handle;

  Vector      *dz_mult = ProblemDataZmult(problem_data);

  double         *rsz_data, *dz_data, z[GridNumSubgrids(grid)], *zz;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int is, i, j, k, l, ips = 0;
  int srcRank, dstRank;

  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

  k = 0; //@RMM bug fix for processor outside active domain
#ifdef HAVE_P4EST
  SubgridArray       *all_subgrids = GridAllSubgrids(grid);
  Subgrid            *neigh_subgrid;
  sc_mempool_t       *sendbuf;
  sc_mempool_t       *old_sendbuf = NULL;
  sc_array_t         *new_requests;
  sc_array_t         *old_requests = NULL;
  double             *zbuf;
  int                *z_levels = grid->z_levels;
  sc_MPI_Request     *outreq;
  int                num_z_levels;
  int                tag, mpiret;
  int                ll, sidx;
#endif

  /*-----------------------------------------------------------------------
   * real_space_z
   *-----------------------------------------------------------------------*/

  InitVectorAll(rsz, 1.0);

#ifdef HAVE_P4EST
    if (USE_P4EST)
      num_z_levels = GlobalsNumProcsZ;
    else
      num_z_levels = 1;
#endif

#ifdef HAVE_P4EST
  for (ll = 0; ll < num_z_levels; ++ll)
  {
    if (USE_P4EST)
    {
      BeginTiming(P4ESTSolveTimingIndex);
      sendbuf = sc_mempool_new(sizeof(double));
      new_requests = sc_array_new(sizeof(sc_MPI_Request));
      EndTiming(P4ESTSolveTimingIndex);
    }
#endif
  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    if (USE_P4EST)
    {
#ifdef HAVE_P4EST
      /** This subgrid does not lie in this level, skip it */
      if (SubgridIZ(subgrid) != z_levels[ll])
        continue;
#endif
    }

    dz_sub = VectorSubvector(dz_mult, is);
    rsz_sub = VectorSubvector(rsz, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    /* RDF: assume resolution is the same in all 3 directions */
    r = SubgridRX(subgrid);

    dz_data = SubvectorData(dz_sub);
    rsz_data = SubvectorData(rsz_sub);

    z[is] = BackgroundZ(GlobalsBackground);

    if (USE_P4EST)
    {
#ifdef HAVE_P4EST
      BeginTiming(P4ESTSolveTimingIndex);

      sidx = SubgridMinusZneigh(subgrid);

      /** We have a neighboring subgrid from bellow, receive its partial sum */
      if (sidx >= 0)
      {
        neigh_subgrid = SubgridArraySubgrid(all_subgrids, sidx);
        srcRank = SubgridProcess(neigh_subgrid);
        tag = ComputeTag(neigh_subgrid, subgrid);
        mpiret = sc_MPI_Recv(&z[is], 1, sc_MPI_DOUBLE, srcRank,
                             tag, amps_CommWorld, sc_MPI_STATUS_IGNORE);
        SC_CHECK_MPI(mpiret);
      }

      EndTiming(P4ESTSolveTimingIndex);
#endif
    }
    else
    {
      /* Receive partial sum from rank below current rank.  This is lower z value for this rank. */
      if (GlobalsR > 0)
      {
        invoice = amps_NewInvoice("%d", &z[is]);
        srcRank = pqr_to_process(GlobalsP,
                                 GlobalsQ,
                                 GlobalsR - 1,
                                 GlobalsNumProcsX,
                                 GlobalsNumProcsY,
                                 GlobalsNumProcsZ);

        amps_Recv(amps_CommWorld, srcRank, invoice);
        amps_FreeInvoice(invoice);
      }
    }

    zz = ctalloc(double, (nz));

    /*
     * This is very ugly, the loop macro is allocating memory,
     * since this loop is breaking out of the macro need to make sure
     * the allocation is freed.
     *
     * Is there really no better way to compute the ips value needed here?
     */

    for (l = iz; l < iz + nz; l++)
    {
      int *breaking_out_PV_visiting = 0;
      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, l, nx, ny, 1,
      {
        //we need one index of level l which is inside the domain
        ips = SubvectorEltIndex(rsz_sub, i, j, k);

        breaking_out_PV_visiting = PV_visiting;
	// Can't just do a break since GrGeom loops are multiple levels deep.
        goto  breakout;
      });

breakout:;

      /* 
       * If we broke out of GrGeomInLoop we found point in 
       * domain on this rank.   If not then nothing 
       * was in domain.
       */

      // Free temporary space allocated in GrGeomInLoop if needed since we did a goto.
      if (breaking_out_PV_visiting)
      {
        tfree(breaking_out_PV_visiting - 1);
      }

      // GrGeomInLoop didnt' find any points in domain
      if (k - iz < nz)
      {
        z[is] += 0.5 * RealSpaceDZ(r) * dz_data[ips];
        zz[k - iz] = z[is];
        z[is] += 0.5 * RealSpaceDZ(r) * dz_data[ips];
      }
    }
    if (USE_P4EST)
    {
#ifdef HAVE_P4EST
      BeginTiming(P4ESTSolveTimingIndex);
      sidx = SubgridPlusZneigh(subgrid);

      /** We have a neighboring subgrid above, send our partial sum */
      if (sidx >= 0)
      {
        neigh_subgrid = SubgridArraySubgrid(all_subgrids, sidx);
        dstRank = SubgridProcess(neigh_subgrid);
        tag = ComputeTag(subgrid, neigh_subgrid);
        outreq = (sc_MPI_Request*)sc_array_push(new_requests);
        zbuf = (double*)sc_mempool_alloc(sendbuf);
        zbuf = &z[is];
        mpiret = sc_MPI_Isend(zbuf, 1, sc_MPI_DOUBLE, dstRank,
                              tag, amps_CommWorld, outreq);
        SC_CHECK_MPI(mpiret);
      }
      EndTiming(P4ESTSolveTimingIndex);
#endif
    }
    else
    {
      /* Send partial sum to rank above current rank */
      if (GlobalsR < GlobalsNumProcsZ - 1)
      {
        invoice = amps_NewInvoice("%d", &z[is]);

        dstRank = pqr_to_process(GlobalsP,
                                 GlobalsQ,
                                 GlobalsR + 1,
                                 GlobalsNumProcsX,
                                 GlobalsNumProcsY,
                                 GlobalsNumProcsZ);

        amps_Send(amps_CommWorld, dstRank, invoice);
        amps_FreeInvoice(invoice);
      }
    }

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      ips = SubvectorEltIndex(rsz_sub, i, j, k);
      rsz_data[ips] = zz[k - iz];
    });
    tfree(zz);
  }

  if (USE_P4EST)
  {
#ifdef HAVE_P4EST
    BeginTiming(P4ESTSolveTimingIndex);

    /** There are no send requests for this z_level,
     * free request and buffer array*/
    if (!new_requests->elem_count)
    {
      sc_array_destroy(new_requests);
      sc_mempool_destroy(sendbuf);
      sendbuf = NULL;
      new_requests = NULL;
    }

    if (old_requests != NULL)
    {
      /** Complete sends from previous z_level */
      mpiret = sc_MPI_Waitall((int)old_requests->elem_count,
                              (sc_MPI_Request*)old_requests->array,
                              sc_MPI_STATUSES_IGNORE);
      SC_CHECK_MPI(mpiret);

      /** Free requests and buffer from previous level */
      sc_array_destroy(old_requests);
      sc_mempool_destroy(old_sendbuf);
    }

    /** Save current requests and buffer to be completed
     *  in the next z_level */
    old_requests = new_requests;
    old_sendbuf = sendbuf;

    EndTiming(P4ESTSolveTimingIndex);
#endif
  }

#ifdef HAVE_P4EST
}
#endif

  handle = InitVectorUpdate(rsz, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
}


/*--------------------------------------------------------------------------
 * realSpaceZInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *realSpaceZInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * realSpaceZFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  realSpaceZFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * realSpaceZNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *realSpaceZNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;


  public_xtra = ctalloc(PublicXtra, 1);


  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * realSpaceZFreePublicXtra
 *--------------------------------------------------------------------------*/

void  realSpaceZFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);



  tfree(public_xtra);
}

/*--------------------------------------------------------------------------
 * realSpaceZSizeOfTempData
 *--------------------------------------------------------------------------*/

int  realSpaceZSizeOfTempData()
{
  return 0;
}
