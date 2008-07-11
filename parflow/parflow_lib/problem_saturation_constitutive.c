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
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int     num_phases;
   double  satconstitutive;
} PublicXtra;

typedef struct
{
   Grid    *grid;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * SaturationConstitutive
 *--------------------------------------------------------------------------*/

void     SaturationConstitutive( phase_saturations )
Vector **phase_saturations;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   int            num_phases      = (public_xtra -> num_phases);
   double         satconstitutive = (public_xtra -> satconstitutive);

   Grid          *grid            = (instance_xtra -> grid);

   SubgridArray  *subgrids;

   Subvector     *subvector_ps, *subvector_psi;

   double        *ps, *psi;

   int            i,  j,  k;
   int            ix, iy, iz;
   int            nx, ny, nz;

   int            nx_ps,  ny_ps,  nz_ps;
   int            nx_psi, ny_psi, nz_psi;

   int            sg, ips, ipsi;


   subgrids = GridSubgrids( grid );


   ForSubgridI(sg, subgrids)
   {
      subvector_ps = VectorSubvector(phase_saturations[num_phases-1], sg);

      ix = SubvectorIX(subvector_ps);
      iy = SubvectorIY(subvector_ps);
      iz = SubvectorIZ(subvector_ps);

      nx = SubvectorNX(subvector_ps);
      ny = SubvectorNY(subvector_ps);
      nz = SubvectorNZ(subvector_ps);

      nx_ps = SubvectorNX(subvector_ps);
      ny_ps = SubvectorNY(subvector_ps);
      nz_ps = SubvectorNZ(subvector_ps);

      ps = SubvectorElt(subvector_ps, ix, iy, iz);

      ips = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                ips,  nx_ps,  ny_ps,  nz_ps,  1, 1, 1,
      {
         ps[ips] = satconstitutive;
      });
   }


   for( i = 0; i < num_phases - 1; i++)
   {
      ForSubgridI(sg, subgrids)
      {
         subvector_ps  = VectorSubvector(phase_saturations[num_phases-1], sg);
         subvector_psi = VectorSubvector(phase_saturations[i], sg);

         ix = SubvectorIX(subvector_ps);
         iy = SubvectorIY(subvector_ps);
         iz = SubvectorIZ(subvector_ps);

         nx = SubvectorNX(subvector_ps);
         ny = SubvectorNY(subvector_ps);
         nz = SubvectorNZ(subvector_ps);

         nx_ps = SubvectorNX(subvector_ps);
         ny_ps = SubvectorNY(subvector_ps);
         nz_ps = SubvectorNZ(subvector_ps);

         nx_psi = SubvectorNX(subvector_psi);
         ny_psi = SubvectorNY(subvector_psi);
         nz_psi = SubvectorNZ(subvector_psi);

         ps  = SubvectorElt(subvector_ps,  ix, iy, iz);
         psi = SubvectorElt(subvector_psi, ix, iy, iz);

         ips = 0;
         ipsi = 0;
         BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                   ips,  nx_ps,  ny_ps,  nz_ps,  1, 1, 1,
                   ipsi, nx_psi, ny_psi, nz_psi, 1, 1, 1,
         {
            ps[ips] -= psi[ipsi];
         });
      }
   }

}

/*--------------------------------------------------------------------------
 * SaturationConstitutiveInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SaturationConstitutiveInitInstanceXtra(grid)
Grid *grid;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      (instance_xtra -> grid) = grid;
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;

   return this_module;
}


/*--------------------------------------------------------------------------
 * SaturationConstitutiveFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SaturationConstitutiveFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      tfree(instance_xtra);
   }

}


/*--------------------------------------------------------------------------
 * SaturationConstitutiveNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *SaturationConstitutiveNewPublicXtra(num_phases)
int        num_phases;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   /*-----------------------------------------------------------------------
    * Setup the PublicXtra structure
    *-----------------------------------------------------------------------*/

   public_xtra = ctalloc(PublicXtra, 1);

   /*-------------------------------------------------------------*/
   /*                     setup parameters                        */

   (public_xtra -> num_phases) = num_phases;

   (public_xtra -> satconstitutive) = 1.0;

   /*-------------------------------------------------------------*/

   PFModulePublicXtra(this_module) = public_xtra;

   return this_module;
}

/*-------------------------------------------------------------------------
 * SaturationConstitutiveFreePublicXtra
 *-------------------------------------------------------------------------*/

void  SaturationConstitutiveFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * SaturationConstitutiveSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SaturationConstitutiveSizeOfTempData()
{
   return 0;
}
