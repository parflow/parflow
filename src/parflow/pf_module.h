/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * PFModule structure and accessor macros
 *
 *****************************************************************************/

#ifndef _PFMODULE_HEADER
#define _PFMODULE_HEADER


/*--------------------------------------------------------------------------
 * PFModule structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   void  (*call)();
   void  (*init_instance_xtra)();
   void  (*free_instance_xtra)();
   void  (*new_public_xtra)();
   void  (*free_public_xtra)();
   int   (*sizeof_temp_data)();

   void  *instance_xtra;
   void  *public_xtra;

} PFModule;

/*--------------------------------------------------------------------------
 * Global xtra pointer
 *   used by module macros below to pass module xtra data to module routines
 *--------------------------------------------------------------------------*/

#ifdef PARFLOW_GLOBALS
amps_ThreadLocalDcl(PFModule  *, global_ptr_this_pf_module);
#else
amps_ThreadLocalDcl(extern PFModule *, global_ptr_this_pf_module);
#endif

#define global_this_pf_module amps_ThreadLocal(global_ptr_this_pf_module)

/*--------------------------------------------------------------------------
 * Accessor macros
 *--------------------------------------------------------------------------*/

#define PFModuleInstanceXtra(pf_module)      (pf_module -> instance_xtra)
#define PFModulePublicXtra(pf_module) 	     (pf_module -> public_xtra)

#define ThisPFModule  global_this_pf_module

/*--------------------------------------------------------------------------
 * PFModule interface macros
 *--------------------------------------------------------------------------*/

#define PFModuleInvoke(type, pf_module, args)\
(\
 ThisPFModule = pf_module,\
 (*(type (*)())(ThisPFModule -> call)) args\
)

#define PFModuleNewInstance(pf_module, args)\
(\
 ThisPFModule = DupPFModule(pf_module),\
 (*(PFModule * (*)())(ThisPFModule -> init_instance_xtra)) args\
)

#define PFModuleReNewInstance(pf_module, args)\
(\
 ThisPFModule = pf_module,\
 (*(PFModule * (*)())(ThisPFModule -> init_instance_xtra)) args\
)

#define PFModuleFreeInstance(pf_module)\
(\
 ThisPFModule = pf_module,\
 (*(void (*)())(ThisPFModule -> free_instance_xtra))(),\
 FreePFModule(pf_module)\
)

#define PFModuleNewModule(name, args)\
(\
 ThisPFModule = NewPFModule((void *)name,\
			    (void *)name##InitInstanceXtra,\
			    (void *)name##FreeInstanceXtra,\
			    (void *)name##NewPublicXtra,\
			    (void *)name##FreePublicXtra,\
			    (void *)name##SizeOfTempData,\
			    NULL, NULL),\
 (*(PFModule * (*)())(ThisPFModule -> new_public_xtra)) args\
)

#define PFModuleFreeModule(pf_module)\
(\
 ThisPFModule = pf_module,\
 (*(void (*)())(ThisPFModule -> free_public_xtra))(),\
 FreePFModule(pf_module)\
)

#define PFModuleSizeOfTempData(pf_module)\
(\
 ThisPFModule = pf_module,\
 (*(int (*)())(ThisPFModule -> sizeof_temp_data)) ()\
)


#endif
