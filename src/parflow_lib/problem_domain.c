/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    geom_index;

} PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid  *grid;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * Domain
 *--------------------------------------------------------------------------*/

void           Domain(problem_data)
ProblemData   *problem_data;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   int            geom_index    = (public_xtra -> geom_index);


   ProblemDataDomain(problem_data) =
      ProblemDataSolid(problem_data, geom_index);
   ProblemDataGrDomain(problem_data) =
      ProblemDataGrSolid(problem_data, geom_index);

   return;
}  


/*--------------------------------------------------------------------------
 * DomainInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *DomainInitInstanceXtra(grid)
Grid      *grid;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;


   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `grid'
    *-----------------------------------------------------------------------*/

   if ( grid != NULL)
   {
      (instance_xtra -> grid) = grid;
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}  


/*--------------------------------------------------------------------------
 * DomainFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  DomainFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if(instance_xtra)
   {
      tfree(instance_xtra);
   }
}  


/*--------------------------------------------------------------------------
 * DomainNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *DomainNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char *geom_name;

   public_xtra = ctalloc(PublicXtra, 1);

   geom_name = GetString("Domain.GeomName");
   public_xtra -> geom_index = NA_NameToIndex(GlobalsGeomNames, geom_name);

   if( public_xtra -> geom_index < 0 )
   {
      InputError("Error: invalid geometry name <%s> for key <%s>\n",
		  geom_name, "Domain.GeomName");
   }


   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}  


/*--------------------------------------------------------------------------
 * DomainFreePublicXtra
 *--------------------------------------------------------------------------*/

void  DomainFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if (public_xtra)
   {
      tfree(public_xtra);
   }
}  

/*--------------------------------------------------------------------------
 * DomainSizeOfTempData
 *--------------------------------------------------------------------------*/

int  DomainSizeOfTempData()
{
   return 0;
}

