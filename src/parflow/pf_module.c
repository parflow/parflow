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
 * Member functions for the pf_module class.
 *
 *****************************************************************************/

#include "parflow.h"
#include "pf_module.h"


/*--------------------------------------------------------------------------
 * NewPFModule
 *--------------------------------------------------------------------------*/

PFModule  *NewPFModule(call,
		       init_instance_xtra, free_instance_xtra,
		       new_public_xtra, free_public_xtra,
		       sizeof_temp_data,
		       instance_xtra, public_xtra)
void      *call;
void      *init_instance_xtra;
void      *free_instance_xtra;
void      *new_public_xtra;
void      *free_public_xtra;
void      *sizeof_temp_data;
void      *instance_xtra;
void      *public_xtra;
{
    PFModule         *new;

    new = talloc(PFModule, 1);

    (new -> call)               = (void (*)())call;
    (new -> init_instance_xtra) = (void (*)())init_instance_xtra;
    (new -> free_instance_xtra) = (void (*)())free_instance_xtra;
    (new -> new_public_xtra)    = (void (*)())new_public_xtra;
    (new -> free_public_xtra)   = (void (*)())free_public_xtra;
    (new -> sizeof_temp_data)   = (int  (*)())sizeof_temp_data;

    PFModuleInstanceXtra(new)     = instance_xtra;
    PFModulePublicXtra(new)       = public_xtra;

    return new;
}


/*--------------------------------------------------------------------------
 * DupPFModule
 *--------------------------------------------------------------------------*/

PFModule  *DupPFModule(pf_module)
PFModule  *pf_module;
{
    return  NewPFModule((void *)(pf_module -> call),
			(void *)(pf_module -> init_instance_xtra),
			(void *)(pf_module -> free_instance_xtra),
			(void *)(pf_module -> new_public_xtra),
			(void *)(pf_module -> free_public_xtra),
			(void *)(pf_module -> sizeof_temp_data),
			PFModuleInstanceXtra(pf_module),
			PFModulePublicXtra(pf_module));
}


/*--------------------------------------------------------------------------
 * FreePFModule
 *--------------------------------------------------------------------------*/

void            FreePFModule(pf_module)
PFModule       *pf_module;
{
   tfree(pf_module);
}

