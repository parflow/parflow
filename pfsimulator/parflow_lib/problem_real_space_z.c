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

#include "parflow.h"
#include "globals.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
} PublicXtra;

typedef void InstanceXtra;


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

  VectorUpdateCommHandle       *handle;

  Vector      *dz_mult = ProblemDataZmult(problem_data);

  double         *rsz_data, *dz_data, z, *zz;

  int ix, iy, iz;
  int nx, ny, nz;
  int r;
  int is, i, j, k, l, ips = 0;

  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

  k = 0; //@RMM bug fix for processor outside active domain

  /*-----------------------------------------------------------------------
   * real_space_z
   *-----------------------------------------------------------------------*/

  InitVectorAll(rsz, 1.0);

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);
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

    z = BackgroundZ(GlobalsBackground);

    /* Receive partial sum from rank below current rank.  This is lower z value for this rank. */
    if (GlobalsR > 0)
    {
      amps_Invoice invoice = amps_NewInvoice("%d", &z);
      int srcRank = pqr_to_process(GlobalsP,
                                   GlobalsQ,
                                   GlobalsR - 1,
                                   GlobalsNumProcsX,
                                   GlobalsNumProcsY,
                                   GlobalsNumProcsZ);

      amps_Recv(amps_CommWorld, srcRank, invoice);
      amps_FreeInvoice(invoice);
    }
    else
    {
      z = BackgroundZ(GlobalsBackground);
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
        z += 0.5 * RealSpaceDZ(SubgridRZ(subgrid)) * dz_data[ips];
        zz[k - iz] = z;
        z += 0.5 * RealSpaceDZ(SubgridRZ(subgrid)) * dz_data[ips];
      }
    }

    /* Send partial sum to rank above current rank */
    if (GlobalsR < GlobalsNumProcsZ - 1)
    {
      amps_Invoice invoice = amps_NewInvoice("%d", &z);

      int dstRank = pqr_to_process(GlobalsP,
                                   GlobalsQ,
                                   GlobalsR + 1,
                                   GlobalsNumProcsX,
                                   GlobalsNumProcsY,
                                   GlobalsNumProcsZ);

      amps_Send(amps_CommWorld, dstRank, invoice);
      amps_FreeInvoice(invoice);
    }

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      ips = SubvectorEltIndex(rsz_sub, i, j, k);

      rsz_data[ips] = zz[k - iz];
    });
    tfree(zz);
  }



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
