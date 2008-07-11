/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * NewComputePkg
 *--------------------------------------------------------------------------*/

ComputePkg  *NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg)
Region      *send_reg;
Region      *recv_reg;
Region      *dep_reg;
Region      *ind_reg;
{
   ComputePkg      *new;


   new = talloc(ComputePkg, 1);

   ComputePkgSendRegion(new) = send_reg;
   ComputePkgRecvRegion(new) = recv_reg;

   ComputePkgDepRegion(new) = dep_reg;
   ComputePkgIndRegion(new) = ind_reg;

   return new;
}


/*--------------------------------------------------------------------------
 * FreeComputePkg
 *--------------------------------------------------------------------------*/

void         FreeComputePkg(compute_pkg)
ComputePkg  *compute_pkg;
{
   if (ComputePkgSendRegion(compute_pkg))
      FreeRegion(ComputePkgSendRegion(compute_pkg));
   if (ComputePkgRecvRegion(compute_pkg))
      FreeRegion(ComputePkgRecvRegion(compute_pkg));

   if (ComputePkgIndRegion(compute_pkg))
      FreeRegion(ComputePkgIndRegion(compute_pkg));
   if (ComputePkgDepRegion(compute_pkg))
      FreeRegion(ComputePkgDepRegion(compute_pkg));

   tfree(compute_pkg);
}


