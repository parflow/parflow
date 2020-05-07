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
* PFModule structure and accessor macros
*
*****************************************************************************/

#ifndef _PFMODULE_HEADER
#define _PFMODULE_HEADER


/*--------------------------------------------------------------------------
 * PFModule structure
 *--------------------------------------------------------------------------*/

typedef struct {
  void (*call)();
  void (*init_instance_xtra)();
  void (*free_instance_xtra)();
  void (*new_public_xtra)();
  void (*free_public_xtra)();
  int (*sizeof_temp_data)();

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

/*--------------------------------------------------------------------------
 * Define __device__ pointer for global_ptr_this_pf_module (CUDA)  
 *--------------------------------------------------------------------------*/

#if (PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA) && defined(__CUDACC__)
#ifdef PARFLOW_GLOBALS
__device__ PFModule *dev_global_ptr_this_pf_module;
#else
/* This extern requires CUDA separate compilation, otherwise nvcc compiler 
   treats the pointer as static variable for each compilation unit          */
extern __device__ PFModule *dev_global_ptr_this_pf_module;
#endif // PARFLOW_GLOBALS
#endif // PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA && __CUDACC__

/*--------------------------------------------------------------------------
 * Accessor macros
 *--------------------------------------------------------------------------*/

#define PFModuleInstanceXtra(pf_module)      (pf_module->instance_xtra)
#define PFModulePublicXtra(pf_module)        (pf_module->public_xtra)

/* These accessor macros depends on compilation trajectory (host/device)    */
#ifdef __CUDA_ARCH__
#define dev_global_this_pf_module amps_ThreadLocal(dev_global_ptr_this_pf_module)
#define ThisPFModule dev_global_this_pf_module
#else
#define global_this_pf_module amps_ThreadLocal(global_ptr_this_pf_module)
#define ThisPFModule global_this_pf_module
#endif

/*--------------------------------------------------------------------------
 * PFModule interface macros
 *--------------------------------------------------------------------------*/

#define PFModuleInvoke(type, pf_module, args) \
  (                                           \
   ThisPFModule = pf_module,                  \
   (*(type (*)())(ThisPFModule->call))args    \
  )

#define PFModuleInvokeType(type, pf_module, args) \
  (                                               \
   ThisPFModule = pf_module,                      \
   (*(type)(ThisPFModule->call))args              \
  )

#define PFModuleNewInstance(pf_module, args)                   \
  (                                                            \
   ThisPFModule = DupPFModule(pf_module),                      \
   (*(PFModule * (*)())(ThisPFModule->init_instance_xtra))args \
  )

#define PFModuleNewInstanceType(type, pf_module, args) \
  (                                                    \
   ThisPFModule = DupPFModule(pf_module),              \
   (*(type)(ThisPFModule->init_instance_xtra))args     \
  )

#define PFModuleReNewInstance(pf_module, args)                 \
  (                                                            \
   ThisPFModule = pf_module,                                   \
   (*(PFModule * (*)())(ThisPFModule->init_instance_xtra))args \
  )

#define PFModuleReNewInstanceType(type, pf_module, args) \
  (                                                      \
   ThisPFModule = pf_module,                             \
   (*(type)(ThisPFModule->init_instance_xtra))args       \
  )

#define PFModuleFreeInstance(pf_module)                 \
  (                                                     \
   ThisPFModule = pf_module,                            \
   (*(void (*)())(ThisPFModule->free_instance_xtra))(), \
   FreePFModule(pf_module)                              \
  )

#define PFModuleNewModule(name, args)                          \
  (                                                            \
   ThisPFModule = NewPFModule((void*)name,                     \
                              (void*)name ## InitInstanceXtra, \
                              (void*)name ## FreeInstanceXtra, \
                              (void*)name ## NewPublicXtra,    \
                              (void*)name ## FreePublicXtra,   \
                              (void*)name ## SizeOfTempData,   \
                              NULL, NULL),                     \
   (*(PFModule * (*)())(ThisPFModule->new_public_xtra))args    \
  )

#define PFModuleNewModuleType(type, name, args)                \
  (                                                            \
   ThisPFModule = NewPFModule((void*)name,                     \
                              (void*)name ## InitInstanceXtra, \
                              (void*)name ## FreeInstanceXtra, \
                              (void*)name ## NewPublicXtra,    \
                              (void*)name ## FreePublicXtra,   \
                              (void*)name ## SizeOfTempData,   \
                              NULL, NULL),                     \
   (*(type)(ThisPFModule->new_public_xtra))args                \
  )

#define PFModuleFreeModule(pf_module)                 \
  (                                                   \
   ThisPFModule = pf_module,                          \
   (*(void (*)())(ThisPFModule->free_public_xtra))(), \
   FreePFModule(pf_module)                            \
  )

#define PFModuleSizeOfTempData(pf_module)           \
  (                                                 \
   ThisPFModule = pf_module,                        \
   (*(int (*)())(ThisPFModule->sizeof_temp_data))() \
  )


#endif
