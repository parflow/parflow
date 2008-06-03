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
 * Hestenes-Stiefel conjugate gradient solver (Omin).
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int        max_iter;

   int        time_index;

} PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid     *grid;
   Matrix   *A;
   double   *temp_data;

   /* instance data */
   Vector   *p;
   Vector   *s;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * CGHS
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test (see Ashby, Holst, Manteuffel, and
 * Saylor):
 *
 *       ||e||_A                       ||r||
 *       -------  <=  [kappa(A)]^(1/2) -----  < tol
 *       ||x||_A                       ||b||
 *
 * where we let (for the time being) kappa(A) = 1.
 * We implement the test as:
 *
 *       gamma = <r,r>  <  (tol^2)*<b,b> = eps
 *
 *--------------------------------------------------------------------------*/

void   	 CGHS(x, b, tol, zero)
Vector 	*x;
Vector 	*b;
double 	 tol;
int    	 zero;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   int        max_iter  = (public_xtra -> max_iter);

   Matrix    *A         = (instance_xtra -> A);

   Vector    *r;
   Vector    *p         = (instance_xtra -> p);
   Vector    *s         = (instance_xtra -> s);

   double     alpha, beta;
   double     gamma, gamma_old;
   double     eps;

   double     b_dot_b;
   
   int        i = 0;

   double    *norm_log;
   double    *rel_norm_log;


   /*-----------------------------------------------------------------------
    * Initialize some logging variables
    *-----------------------------------------------------------------------*/

   IfLogging(1)
   {
      norm_log     = talloc(double, max_iter);
      rel_norm_log = talloc(double, max_iter);
   }

   /*-----------------------------------------------------------------------
    * Begin timing
    *-----------------------------------------------------------------------*/

   BeginTiming(public_xtra -> time_index);

   /*-----------------------------------------------------------------------
    * Start cghs solve
    *-----------------------------------------------------------------------*/

   if (zero)
      InitVector(x, 0.0);

   /* eps = (tol^2)*<b,b> */
   b_dot_b = InnerProd(b, b);
   eps = (tol*tol)*b_dot_b;

   /* r = b - Ax,  (overwrite b with r) */
   Matvec(-1.0, A, x, 1.0, (r = b));

   /* p = r */
   Copy(r, p);

   /* gamma = <r,r> */
   gamma = InnerProd(r,r);

   while (((i+1) <= max_iter) && (gamma > 0))
   {
      i++;

      /* s = A*p */
      Matvec(1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      alpha = gamma / InnerProd(s, p);

      gamma_old = gamma;

      /* x = x + alpha*p */
      Axpy(alpha, p, x);
 
      /* r = r - alpha*s */
      Axpy(-alpha, s, r);
	 
      /* gamma = <r,r> */
      gamma = InnerProd(r, r);

#if 0
      if(!amps_Rank(amps_CommWorld))
         amps_Printf("Iteration (%d): ||r|| = %e, ||r||/||b|| = %e\n",
		    i, sqrt(gamma), (b_dot_b ? sqrt(gamma/b_dot_b) : 0));
#endif

      /* log norm info */
      IfLogging(1)
      {
         norm_log[i-1]     = sqrt(gamma);
         rel_norm_log[i-1] = b_dot_b ? sqrt(gamma/b_dot_b) : 0;
      }
 
      /* check for convergence */
      if (gamma < eps)
	 break;

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = r + beta p */
      Scale(beta, p);   
      Axpy(1.0, r, p);
   }

   if(!amps_Rank(amps_CommWorld))
      amps_Printf("Iterations = %d, ||r|| = %e, ||r||/||b|| = %e\n",
		  i, sqrt(gamma), (b_dot_b ? sqrt(gamma/b_dot_b) : 0));

   /*-----------------------------------------------------------------------
    * end timing
    *-----------------------------------------------------------------------*/

   IncFLOPCount(i*2 - 1);
   EndTiming(public_xtra -> time_index);

   /*-----------------------------------------------------------------------
    * Print log.
    *-----------------------------------------------------------------------*/

   IfLogging(1)
   {
      FILE  *log_file;
      int        j;

      log_file = OpenLogFile("CGHS");

      fprintf(log_file, "Iters       ||r||_2    ||r||_2/||b||_2\n");
      fprintf(log_file, "-----    ------------    ------------\n");
    
      for (j = 0; j < i; j++)
      {
         fprintf(log_file, "% 5d    %e    %e\n",
                      (j+1), norm_log[j], rel_norm_log[j]);
      }

      CloseLogFile(log_file);

      tfree(norm_log);
      tfree(rel_norm_log);
   }
}


/*--------------------------------------------------------------------------
 * CGHSInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule     *CGHSInitInstanceXtra(problem, grid, problem_data, A, temp_data)
Problem      *problem;
Grid         *grid;
ProblemData  *problem_data;
Matrix       *A;
double       *temp_data;
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
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
         FreeTempVector(instance_xtra -> s);
         FreeTempVector(instance_xtra -> p);
      }

      /* set new data */
      (instance_xtra -> grid) = grid;

      (instance_xtra -> p) = NewTempVector(grid, 1, 1);
      (instance_xtra -> s) = NewTempVector(grid, 1, 1);
   }

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `A'
    *-----------------------------------------------------------------------*/

   if ( A != NULL)
      (instance_xtra -> A) = A;

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;

      SetTempVectorData((instance_xtra -> p), temp_data);
      temp_data += SizeOfVector(instance_xtra -> p);
      SetTempVectorData((instance_xtra -> s), temp_data);
      temp_data += SizeOfVector(instance_xtra -> s);
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * CGHSFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   CGHSFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if(instance_xtra)
   {
      FreeTempVector(instance_xtra -> s);
      FreeTempVector(instance_xtra -> p);

      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * CGHSNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *CGHSNewPublicXtra(char *name)
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char key[IDB_MAX_KEY_LEN];

   public_xtra = ctalloc(PublicXtra, 1);

   sprintf(key, "%s.MaxIter", name);
   public_xtra -> max_iter = GetIntDefault(key, 1000);

   (public_xtra -> time_index) = RegisterTiming("CGHS");

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * CGHSFreePublicXtra
 *--------------------------------------------------------------------------*/

void   CGHSFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if(public_xtra)
   {
      tfree(public_xtra);
   }
   
}


/*--------------------------------------------------------------------------
 * CGHSSizeOfTempData
 *--------------------------------------------------------------------------*/

int  CGHSSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);

   int  sz = 0;


   /* add local TempData size to `sz' */
   sz += SizeOfVector(instance_xtra -> p);
   sz += SizeOfVector(instance_xtra -> s);

   return sz;
}
