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
 * Weighted Jacobi iteration.
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   double    weight;
   int       max_iter;

} PublicXtra;

typedef struct
{
   /* InitInstanceXtra arguments */
   Grid      *grid;
   Matrix    *A;
   double    *temp_data;

   /* instance data */
   Vector    *t;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * WJacobi:
 *   Solves A x = b.
 *--------------------------------------------------------------------------*/

void   	 WJacobi(x, b, tol, zero)
Vector 	*x;
Vector 	*b;
double 	 tol;
int    	 zero;
{
   PFModule       *this_module   = ThisPFModule;
   PublicXtra     *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra   *instance_xtra = PFModuleInstanceXtra(this_module);

   double          weight   = (public_xtra -> weight);
   int         	   max_iter = (public_xtra -> max_iter);

   Matrix         *A        = (instance_xtra -> A);
	       	
   Vector      	  *t        = (instance_xtra -> t);

   SubregionArray *subregion_array;
   Subregion      *subregion;

   ComputePkg     *compute_pkg;
   Region         *compute_reg;

   Submatrix      *A_sub;
   Subvector      *x_sub;
   Subvector      *b_sub;
   Subvector      *t_sub;

   Stencil        *stencil;
   int             stencil_size;
   StencilElt     *s;

   double         *ap;
   double         *xp;
   double         *bp, *tp;

   int       	   ix,   iy,   iz;
   int       	   nx,   ny,   nz;
   int       	   sx,   sy,   sz;
	      
   int       	   nx_v, ny_v, nz_v;
   int       	   nx_m, ny_m, nz_m;

   int             compute_i, i_sa, i_s, si, i, j, k;
   int             im, iv;

   int             iter = 0;

   CommHandle     *handle;


   /*-----------------------------------------------------------------------
    * Begin timing
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    * Start WJacobi
    *-----------------------------------------------------------------------*/

   compute_pkg = GridComputePkg(VectorGrid(x), VectorUpdateAll);

   /*-----------------------------------------------------------------------
    * If (zero) optimize iteration
    *-----------------------------------------------------------------------*/

   if (zero)
   {
      if ((iter+1) > max_iter)
      {
         InitVector(x, 0.0);
         return;
      }

      iter++;

      /* RDF get from ComputePkg instead */
      subregion_array = GridSubgrids(VectorGrid(x));
      
      ForSubregionI(i_s, subregion_array)
      {
	 subregion = SubregionArraySubregion(subregion_array, i_s);
	 
	 A_sub = MatrixSubmatrix(A, i_s);
	 x_sub = VectorSubvector(x, i_s);
	 b_sub = VectorSubvector(b, i_s);

	 nx_m = SubmatrixNX(A_sub);
	 ny_m = SubmatrixNY(A_sub);
	 nz_m = SubmatrixNZ(A_sub);

	 nx_v = SubvectorNX(x_sub);
	 ny_v = SubvectorNY(x_sub);
	 nz_v = SubvectorNZ(x_sub);
      
	 ix = SubregionIX(subregion);
	 iy = SubregionIY(subregion);
	 iz = SubregionIZ(subregion);
	 
	 nx = SubregionNX(subregion);
	 ny = SubregionNY(subregion);
	 nz = SubregionNZ(subregion);
	 
	 sx = SubregionSX(subregion);
	 sy = SubregionSY(subregion);
	 sz = SubregionSZ(subregion);
	 
	 ap = SubmatrixElt(A_sub, 0, ix, iy, iz);
	 xp = SubvectorElt(x_sub, ix, iy, iz);
	 bp = SubvectorElt(b_sub, ix, iy, iz);
	    
	 iv = im = 0;
	 BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
		   iv, nx_v, ny_v, nz_v, sx, sy, sz,
		   im, nx_m, ny_m, nz_m, sx, sy, sz,
		   {
		      xp[iv] = bp[iv] / ap[im];
		   });
      }

      if (weight != 1.0)
	 Scale(weight, x);

      IncFLOPCount( -12*(iter*VectorSize(x)) );
   }

   /*-----------------------------------------------------------------------
    * Do regular iterations
    *-----------------------------------------------------------------------*/

   while ((iter+1) <= max_iter)
   {
      iter++;

      Copy(b, t);

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
         case 0:
            handle = InitVectorUpdate(x, VectorUpdateAll);
            compute_reg = ComputePkgIndRegion(compute_pkg);
            break;

         case 1:
            FinalizeVectorUpdate(handle);
            compute_reg = ComputePkgDepRegion(compute_pkg);
            break;
         }

         ForSubregionArrayI(i_sa, compute_reg)
         {
            subregion_array = RegionSubregionArray(compute_reg, i_sa);

            if (SubregionArraySize(subregion_array))
            {
               A_sub = MatrixSubmatrix(A, i_sa);
               x_sub = VectorSubvector(x, i_sa);
               t_sub = VectorSubvector(t, i_sa);

               nx_m = SubmatrixNX(A_sub);
               ny_m = SubmatrixNY(A_sub);
               nz_m = SubmatrixNZ(A_sub);

               nx_v = SubvectorNX(x_sub);
               ny_v = SubvectorNY(x_sub);
               nz_v = SubvectorNZ(x_sub);
            }

            ForSubregionI(i_s, subregion_array)
            {
               subregion = SubregionArraySubregion(subregion_array, i_s);

               ix = SubregionIX(subregion);
               iy = SubregionIY(subregion);
               iz = SubregionIZ(subregion);

               nx = SubregionNX(subregion);
               ny = SubregionNY(subregion);
               nz = SubregionNZ(subregion);

               sx = SubregionSX(subregion);
               sy = SubregionSY(subregion);
               sz = SubregionSZ(subregion);

	       stencil = MatrixStencil(A);
	       stencil_size = StencilSize(stencil);
	       s = StencilShape(stencil);

	       tp = SubvectorElt(t_sub, ix, iy, iz);

	       for (si = 1; si < stencil_size; si++)
	       {
		  xp = SubvectorElt(x_sub,
				    (ix + s[si][0]),
				    (iy + s[si][1]),
				    (iz + s[si][2]));
		  ap = SubmatrixElt(A_sub, si, ix, iy, iz);
 
		  iv = im = 0;
		  BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
			    iv, nx_v, ny_v, nz_v, sx, sy, sz,
			    im, nx_m, ny_m, nz_m, sx, sy, sz,
			    {
			       tp[iv] -= ap[im] * xp[iv];
			    });
	       }

	       ap = SubmatrixElt(A_sub, 0, ix, iy, iz);
	       iv = im = 0;
	       BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
			 iv, nx_v, ny_v, nz_v, sx, sy, sz,
			 im, nx_m, ny_m, nz_m, sx, sy, sz,
			 {
			    tp[iv] /= ap[im];
			 });
	    }
         }
      }

      if (weight != 1.0)
      {
	 Scale((1.0 - weight), x);
	 Axpy(weight, t, x);
      }
      else
      {
	 Copy(t, x);
      }
   }

   /*-----------------------------------------------------------------------
    * end timing
    *-----------------------------------------------------------------------*/

   IncFLOPCount( 13*(iter*VectorSize(x)) );
}


/*--------------------------------------------------------------------------
 * WJacobiInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule     *WJacobiInitInstanceXtra(problem, grid, problem_data, A,
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
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
         FreeTempVector(instance_xtra -> t);
      }

      /* set new data */
      (instance_xtra -> grid) = grid;

      (instance_xtra -> t) = NewTempVector(grid, 1, 1);
   }

   /*-----------------------------------------------------------------------
    * Initialize data associated with arguments `A' and `b'
    *-----------------------------------------------------------------------*/

   if ( A != NULL)
      (instance_xtra -> A) = A;

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;

      SetTempVectorData((instance_xtra -> t), temp_data);
      temp_data += SizeOfVector(instance_xtra -> t);
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * WJacobiFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  WJacobiFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if(instance_xtra)
   {
      FreeTempVector((instance_xtra -> t));

      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * WJacobiNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *WJacobiNewPublicXtra(char *name)
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char key[IDB_MAX_KEY_LEN];

   public_xtra = ctalloc(PublicXtra, 1);

   sprintf(key, "%s.Weight", name);
   public_xtra -> weight = GetDoubleDefault(key, 1.0);

   sprintf(key, "%s.MaxIter", name);
   public_xtra -> max_iter = GetIntDefault(key, 1);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * WJacobiFreePublicXtra
 *--------------------------------------------------------------------------*/

void  WJacobiFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if(public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * WJacobiSizeOfTempData
 *--------------------------------------------------------------------------*/

int  WJacobiSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);

   int  sz = 0;


   /* add local TempData size to `sz' */
   sz += SizeOfVector(instance_xtra -> t);

   return sz;
}
