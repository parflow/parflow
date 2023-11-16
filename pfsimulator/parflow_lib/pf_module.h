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

/**
 * Object like structure to achieve polymorphism.
 *
 * ParFlow existed before C++ was in widespread use.  Polymorphism is
 * achieved by using function pointers stored in a standard C
 * structure. This pattern is used across ParFlow to enable the use of
 * multiple algorithms without need for switch like statements at the
 * invocation site.
 *
 * The Parflow Module pattern has a single main "call" method which
 * invokes the algorithm.  It is a little like a functor in C++.  This
 * is limited so two Output methods have been added in Extended
 * modules.  And yes, moving to C++ would make this cleaner.
 * 
 * Construction is broken down into two calls.   PublicExtra is data that is 
 * shared by all instances of that type of module.   The Extra concept is
 * much like a static member of a C++ class.   InstanceExtra is the 
 * data that is unique to each instance of an algorithm.
 *
 * The output methods can be called to output state that is 
 * time variant (called at each timestep as indicated by the user)
 * or time invariant (called only once).
 */
typedef struct {
  /**
   * Virtual method invoked to do the function of the algorithm.
   */
  void (*call)();

  /**
   * Virtual method invoked to construct instance data.
   * 
   * Instance data is per instantiation the module type; it is 
   * not shared between instances.
   */
  void (*init_instance_xtra)();
  
  /**
   * Virtual method invoked to free the instance data.
   */
  void (*free_instance_xtra)();

  /**
   * Virtual method invoked to construct the class data.
   *
   * Public data is allocated once per module type.   It is
   * similar to class data in C++.
   */
  void (*new_public_xtra)();

  /**
   * Virtual method invoked to free the class data.
   */
  void (*free_public_xtra)();
  
  /**
   * Virtual method to compute the size of the temporary data.
   *
   * This returns the size of the temporary data used by 
   * the instance.
   */
  int (*sizeof_temp_data)();

  /**
   * Output time variant state.
   *
   * Output data from the module instance, once per time
   * interval specified by the user.
   */
  void (*output)();

  /**
   * Output static (time invariant) state.
   *
   * Output data from the module instance, once per simulation.
   * Time invariant data should be outputted in this method.
   */
  void (*output_static)();

  /**
   * The instance data for each instance of a module.
   * This data is unique to each instance; it is not shared.
   * Like object data members in C++.
   */
  void  *instance_xtra;
  /**
   * The class data for each type of module.
   * This data is shared by all instances.
   * Like class data members in C++.
   */
  void  *public_xtra;
} PFModule;

/**
 * A 'this' pointer for module methods.
 * 
 * Used by module macros below to pass module xtra data to module
 * routines This is similar in purpose to the 'this' pointer in C++.
 * Rather than passing this in each function it is done through a
 * thread pointer.  For non-threaded impls this is just a global
 * value.
 *
 * This value is set before the by the PFModule macros before the
 * function pointer is extracted from the module structure and
 * invoked.
 */
#ifdef PARFLOW_GLOBALS
amps_ThreadLocalDcl(PFModule  *, global_ptr_this_pf_module);
#else
amps_ThreadLocalDcl(extern PFModule *, global_ptr_this_pf_module);
#endif

/**
 * Device copy of the 'this' module pointer.
 */ 
#if (PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA) && defined(__CUDACC__)
#ifdef PARFLOW_GLOBALS
__device__ PFModule *dev_global_ptr_this_pf_module;
#else
/* This extern requires CUDA separate compilation, otherwise nvcc compiler 
   treats the pointer as static variable for each compilation unit          */
extern __device__ PFModule *dev_global_ptr_this_pf_module;
#endif // PARFLOW_GLOBALS
#endif // PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA && __CUDACC__

/**
 * Return the module instance data.
 *
 * @param pf_module The module instance.
 * @return Pointer to the instance data
 */
#define PFModuleInstanceXtra(pf_module)      (pf_module->instance_xtra)

/**
 * Return the module class (public) data.
 *
 * @param pf_module The module instance
 * @return Pointer to the class data.
 */
#define PFModulePublicXtra(pf_module)        (pf_module->public_xtra)

/**
 * Accessor macro for the global/thread variable holding the 'this' pointer
 *
 * These accessor macros depends on host/device context.
 */
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

/**
 * Invoke the main algorithm method in the module.
 *
 * Modules are based on the idea the module has single method that
 * does the work of the algorithm.  A bit like the functor pattern in
 * C++.   
 *
 * @note
 * Rational: This was done so all PFModules would have the same
 * calling convention.
 *
 * @param type Function type to invoke.
 * @param pf_module The module instance
 * @param args Arguments for the method to invoke.
 * @return The invoked method return
 */
#define PFModuleInvokeType(type, pf_module, args) \
  (                                               \
   ThisPFModule = pf_module,                      \
   (*(type)(ThisPFModule->call))args              \
  )

/**
 * Create a new instance of a module.
 *
 * Note we have done some renaming to better match OO (C++) naming
 * schemes.  This invokes the "init" method in the method which is a
 * ctor for the method instance.
 * 
 * The method invoked is assumed to be of type 'PFModule * (*)()'.
 * The PFModuleNewInstanceType can be used to specify the actual type
 * of the method to invoke.  It is better to specify the type; PF
 * started with KR style C with less type safety.
 * 
 * @param pf_module The module instance
 * @param args Arguments for the method to invoke
 * @return The new module instance pointer
 */
#define PFModuleNewInstance(pf_module, args)                   \
  (                                                            \
   ThisPFModule = DupPFModule(pf_module),                      \
   (*(PFModule * (*)())(ThisPFModule->init_instance_xtra))args \
  )

/**
 * Create a new instance of a module.
 *
 * Note we have done some renaming to better match OO (C++) naming
 * schemes.  This invokes the "init" method in the method which is a
 * ctor for the method instance.
 *
 * @param type Function type to invoke
 * @param pf_module The module instance
 * @param args Arguments for the method to invoke
 * @return The new module instance pointer
 */
#define PFModuleNewInstanceType(type, pf_module, args) \
  (                                                    \
   ThisPFModule = DupPFModule(pf_module),              \
   (*(type)(ThisPFModule->init_instance_xtra))args     \
  )

/**
 * 'ReNew' the instance of a module.
 * 
 * 'ReNew'ing a module is done when the module has already been
 * constructed but some state may have changed.
 *
 * The method invoked is assumed to be of type 'PFModule * (*)()'.
 * The PFModuleReNewInstanceType can be used to specify the actual
 * type of the method to invoke.  It is better to specify the type; PF
 * started with KR style C with less type safety.
 * 
 * \TODO this should be better documented.  What do we mean by rewnew
 * and when/how is it used.
 *
 * @param pf_module The module instance
 * @param args Arguments for the method to invoke
 * @return The new module instance pointer
 */
#define PFModuleReNewInstance(pf_module, args)                 \
  (                                                            \
   ThisPFModule = pf_module,                                   \
   (*(PFModule * (*)())(ThisPFModule->init_instance_xtra))args \
  )

/**
 * 'ReNew' the instance of a module.
 * 
 * 'ReNew'ing a module is done when the module has already been
 * constructed but some state may have changed.
 * 
 * \TODO this should be better documented.  What do we mean by rewnew
 * and when/how is it used.
 *
 * @param pf_module The module instance
 * @param args Arguments for the method to invoke.
 * @return The new module instance pointer.
 */
#define PFModuleReNewInstanceType(type, pf_module, args) \
  (                                                      \
   ThisPFModule = pf_module,                             \
   (*(type)(ThisPFModule->init_instance_xtra))args       \
  )

/**
 * Free the module.
 *
 * Similiar to a C++ destructor.
 *
 * @param pf_module The module instance
 */
#define PFModuleFreeInstance(pf_module)                 \
  (                                                     \
   ThisPFModule = pf_module,                            \
   (*(void (*)())(ThisPFModule->free_instance_xtra))(), \
   FreePFModule(pf_module)                              \
  )

/**
 * Output time variant state associated with this module instance.
 * 
 * This method is invoked at the output time intervals specified by
 * the user input.  The method should write any vectors or other state
 * out for the current timestep.
 *
 * @param type Type of the method to invoke for outputing
 * @param pf_module The module instance
 * @param args Arguments for the output method
 */
#define PFModuleOutputType(type, pf_module, args) \
  (                                           \
   ThisPFModule = pf_module,                  \
   (*(type (*)())(ThisPFModule->output))args  \
  )


/**
 * Output static (time invariant) state associated with this module instance.
 * 
 * This method is invoked at the start of a run and should write any
 * vectors or other state for the module that does not require a
 * timestamp.
 *
 * @param type Type of method to invoke for outputing
 * @param pf_module The module instance
 * @param args Arguments for the output method
 */
#define PFModuleOutputStaticType(type, pf_module, args) \
  (                                           \
   ThisPFModule = pf_module,                  \
   (*(type (*)(char file_prefix[2048],ProblemData *))(ThisPFModule->output_static))args	\
  )

/**
 * Create a class of module.
 *
 * This allocates the public/class data for the module
 * class.
 *
 * @param name PFModule name
 * @param args Arguments for the module class constructor
 * @return The new module instance
 */
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

/**
 * Create a class of module.
 *
 * This allocates the public/class data for the module
 * class.
 *
 * @param type Type of the new public xtra method
 * @param name PFModule name
 * @param args Arguments for the module class constructor
 * @return The new module instance
 */
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

/**
 * Create a class of extended module.
 * 
 * For use with modules that implement the Extended module API.
 * Currently this is the Output methods.
 *
 * This allocates the public/class data for the module
 * class.
 *
 * @param type Type of the new public xtra method
 * @param name PFModule name
 * @param args Arguments for the module class constructor
 * @return The new module instance
 */
#define PFModuleNewModuleExtendedType(type, name, args)        \
  (                                                            \
   ThisPFModule = NewPFModuleExtended((void*)name,                     \
                              (void*)name ## InitInstanceXtra, \
                              (void*)name ## FreeInstanceXtra, \
                              (void*)name ## NewPublicXtra,    \
                              (void*)name ## FreePublicXtra,   \
                              (void*)name ## SizeOfTempData,   \
                              (void*)name ## Output,	       \
                              (void*)name ## OutputStatic,     \
                              NULL, NULL),                     \
   (*(type)(ThisPFModule->new_public_xtra))args                \
  )


/**
 * Invoke the destructor of the module.
 * 
 * @param pf_module The module instance
 */
#define PFModuleFreeModule(pf_module)                 \
  (                                                   \
   ThisPFModule = pf_module,                          \
   (*(void (*)())(ThisPFModule->free_public_xtra))(), \
   FreePFModule(pf_module)                            \
  )

/**
 * Return this size of the temporary data needed by this module instance.
 *
 * Temp date is a block of memory allocated in the solver and modules
 * use this block.  Data use may be overlayed in this block.  The
 * size of the block is the high water mark.
 * 
 * Use of temp data has mostly been removed but some modules
 * (e.g. advection_godunov) still use temp_data.  Modern memory
 * allocators have removed much of the need for this mechanism.
 *
 * @note
 * Rational: PF originally operated on very low memory compute nodes
 * (4MB-16MB) so had to carefully manage space for vectors.   This 
 * method is part of that mechanism.
 *
 * @param pf_module The module instance
 */
#define PFModuleSizeOfTempData(pf_module)           \
  (                                                 \
   ThisPFModule = pf_module,                        \
   (*(int (*)())(ThisPFModule->sizeof_temp_data))() \
  )

#endif
