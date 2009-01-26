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
 * Chebyshev iteration.
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid     *grid;
   Matrix   *A;
   double    *temp_data;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * Chebyshev:
 *   Solves A x = b with interval [ia, ib].
 *   RDF Assumes initial guess of 0.
 *--------------------------------------------------------------------------*/

void   	 Chebyshev(x, b, tol, zero, ia, ib, num_iter)
Vector 	*x;
Vector 	*b;
double 	 tol;
int    	 zero;
double 	 ia, ib;
int    	 num_iter;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   Matrix    *A         = (instance_xtra -> A);

   Vector *r   = NULL;
   Vector *del = NULL;

   double  d, c22;
   double  alpha, beta;

   int     i = 0;


   /*-----------------------------------------------------------------------
    * Begin timing
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    * Allocate temp vectors
    *-----------------------------------------------------------------------*/
   r   = NewVector(instance_xtra -> grid, 1, 1);
   del = NewVector(instance_xtra -> grid, 1, 1);

   /*-----------------------------------------------------------------------
    * Start Chebyshev
    *-----------------------------------------------------------------------*/

   /* x = b */
   Copy(b, x);

   if ((i+1) > num_iter)
   {
      return;
   }

   i++;

   d = 0.5*(ib + ia);

   /* x = (1/d)*x */
   Scale((alpha = (1.0/d)), x);

   if ((i+1) > num_iter)
   {
      IncFLOPCount(3);
      return;
   }

   c22  = 0.25*(ib - ia);
   c22 *= c22;

   /* alpha = 2 / d */
   alpha *= 2.0;

   /* del = x */
   Copy(x, del);

   while ((i+1) <= num_iter)
   {
      i++;

      alpha = 1.0 / (d - c22*alpha);

      beta = d*alpha - 1.0;

      /* r = b - A*x */
      Matvec(-1.0, A, x, 0.0, r);
      Axpy(1.0, b, r);

      /* del = alpha*r + beta*del */
      Scale(beta, del);
      Axpy(alpha, r, del);

      /* x = x + del */
      Axpy(1.0, del, x);
   }

   /*-----------------------------------------------------------------------
    * Free temp vectors
    *-----------------------------------------------------------------------*/
   FreeVector(del);
   FreeVector(r);

   /*-----------------------------------------------------------------------
    * end Chebyshev
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    * end timing
    *-----------------------------------------------------------------------*/

   IncFLOPCount(i*5 + 7);
}


/*--------------------------------------------------------------------------
 * ChebyshevInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule     *ChebyshevInitInstanceXtra(problem, grid, problem_data, A,
					temp_data)
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
      /* set new data */
      (instance_xtra -> grid) = grid;
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
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * ChebyshevFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  ChebyshevFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if(instance_xtra)
   {
      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * ChebyshevNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ChebyshevNewPublicXtra(char *name)
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   public_xtra = NULL;

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * ChebyshevFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ChebyshevFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   if(public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * ChebyshevSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ChebyshevSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);

   int  sz = 0;

   return sz;
}
