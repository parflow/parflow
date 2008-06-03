/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"
#include "kinsol_dependences.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int        pc_matrix_type;

   PFModule  *precond;
   PFModule  *discretization;

} PublicXtra;

typedef struct
{
   PFModule  *precond;
   PFModule  *discretization;

   Matrix    *PC;

   Vector    *soln;

   Grid      *grid;

   double    *temp_data;

} InstanceXtra;

/*--------------------------------------------------------------------------
 * KinsolPC
 *--------------------------------------------------------------------------*/

void         KinsolPC(rhs)
Vector      *rhs;
{
   PFModule     *this_module      = ThisPFModule;
   InstanceXtra *instance_xtra    = PFModuleInstanceXtra(this_module);

   PFModule     *precond          = instance_xtra -> precond;
   Vector       *soln             = instance_xtra -> soln;
   double        tol              = 0.0;
   int           zero             = 1;
     
   /* Invoke the preconditioner using a zero initial guess */
   PFModuleInvoke(void, precond, (soln, rhs, tol, zero));

   /* Copy solution from soln to the rhs vector. */
   PFVCopy(soln, rhs);
}

/*--------------------------------------------------------------------------
 * KinsolPCInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolPCInitInstanceXtra(problem, grid, problem_data,  
				    temp_data, specie, pressure, temperature, saturation, 
				    density, viscosity, dt, time, n, nspecies)
Problem      *problem;
Grid         *grid;
ProblemData  *problem_data;
double       *temp_data;
Vector       *specie;
Vector       *pressure;
Vector       *temperature;
Vector       *saturation;
Vector       *density;
Vector       *viscosity;
double        dt;
double        time;
int           n;
int           nspecies;
{
   PFModule      *this_module        = ThisPFModule;
   PublicXtra    *public_xtra        = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   int            pc_matrix_type     = public_xtra -> pc_matrix_type;

   PFModule      *discretization     = public_xtra -> discretization;
   PFModule      *precond            = public_xtra -> precond;

   Matrix        *PC;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
         FreeTempVector(instance_xtra -> soln);
      }

      /* set new data */
      instance_xtra -> grid = grid;

      instance_xtra -> soln     = NewTempVector(grid, 1, 1);
   }

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;
      SetTempVectorData((instance_xtra -> soln), temp_data);
      temp_data += SizeOfVector(instance_xtra -> soln);
   }

   /*-----------------------------------------------------------------------
    * Initialize module instances
    *-----------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> discretization) = 
	 PFModuleNewInstance(discretization, (problem, grid, temp_data, 
					      pc_matrix_type, n, nspecies));
      (instance_xtra -> precond) =
         PFModuleNewInstance(precond,(problem, grid, problem_data, NULL, 
				      temp_data));
   }
   else if (specie!= NULL)
   {
      PFModuleInvoke(void, (instance_xtra -> discretization),
		  (specie, &PC, pressure, temperature, saturation, density, viscosity, problem_data, dt, 
		   time, pc_matrix_type, n, nspecies));
      PFModuleReNewInstance((instance_xtra -> precond),
			    (NULL, NULL, problem_data, PC, temp_data));
   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> discretization), 
			    (problem, grid, temp_data, pc_matrix_type, n, nspecies));
      PFModuleReNewInstance((instance_xtra -> precond),
			    (NULL, NULL, problem_data, NULL, temp_data));
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * KinsolPCFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  KinsolPCFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      PFModuleFreeInstance((instance_xtra -> precond));
      PFModuleFreeInstance((instance_xtra -> discretization));

      FreeTempVector(instance_xtra -> soln);

      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * KinsolPCNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *KinsolPCNewPublicXtra(char *name, char *pc_name)
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char          *switch_name;
   int            switch_value;
   char           key[IDB_MAX_KEY_LEN];
   NameArray      precond_na, precond_switch_na;

   public_xtra = ctalloc(PublicXtra, 1);

   precond_na = NA_NewNameArray("FullJacobian PFSymmetric SymmetricPart Picard"
				);
   sprintf(key, "%s.PCMatrixType", name);
   switch_name = GetStringDefault(key,"PFSymmetric");
   switch_value  = NA_NameToIndex(precond_na, switch_name);
   switch (switch_value)
   {
      case 0:
      {
	 public_xtra -> pc_matrix_type = 0;
	 break;
      }
      case 1:
      {
	 public_xtra -> pc_matrix_type = 1;
	 break;
      }
      case 2:
      {
	 public_xtra -> pc_matrix_type = 2;
	 break;
      }
      case 3:
      {
	 public_xtra -> pc_matrix_type = 3;
	 break;
      }
      default:
      {
	 InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		     key);
      }
   }
   NA_FreeNameArray(precond_na);

   precond_switch_na = NA_NewNameArray("NoPC MGSemi SMG PFMG");
   switch_value = NA_NameToIndex(precond_switch_na, pc_name);
   sprintf(key, "%s.%s", name, pc_name);
   switch (switch_value)
   {
      case 0:
      {
	 public_xtra -> precond = NULL;
	 break;
      }
      case 1:
      {
	 public_xtra -> precond = PFModuleNewModule(MGSemi, (key));
	 break;
      }
      case 2:
      {
#ifdef PARFLOW_USE_HYPRE
	 public_xtra -> precond = PFModuleNewModule(SMG, (key));
#else
	 InputError("Error: Invalid value <%s> for key <%s>.\n"
		   "SMG code not compiled in.\n", switch_name, key);
#endif
	 break;
      }
      case 3:
      {
#ifdef PARFLOW_USE_HYPRE
	 public_xtra -> precond = PFModuleNewModule(PFMG, (key));
#else
	 InputError("Error: Invalid value <%s> for key <%s>.\n"
		    "PFMG code not compiled in.\n", switch_name, key);
#endif
	 break;
      }
   }
   NA_FreeNameArray(precond_switch_na);

   public_xtra -> discretization = PFModuleNewModule(RichardsJacobianEval, ());

   PFModulePublicXtra(this_module) = public_xtra;

   return this_module;
}

/*-------------------------------------------------------------------------
 * KinsolPCFreePublicXtra
 *-------------------------------------------------------------------------*/

void  KinsolPCFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      PFModuleFreeModule(public_xtra -> precond);
      PFModuleFreeModule(public_xtra -> discretization);

      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * KinsolPCSizeOfTempData
 *--------------------------------------------------------------------------*/

int  KinsolPCSizeOfTempData()
{
   PFModule             *this_module   = ThisPFModule;
   InstanceXtra         *instance_xtra = PFModuleInstanceXtra(this_module);
   PFModule             *precond       = (instance_xtra -> precond);

   int sz = 0;

   sz += PFModuleSizeOfTempData(precond);
   sz += SizeOfVector(instance_xtra -> soln);
   
   return sz;
}
