/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifdef PARFLOW_USE_HYPRE

#include "parflow.h"
#include "hypre_dependences.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{

   int  max_iter;
   int  num_pre_relax;
   int  num_post_relax;
   int  smoother;

   int  time_index_pfmg;
   int  time_index_copy_hypre;

} PublicXtra;

typedef struct
{
   double               dxyz[3];

   HYPRE_StructGrid     hypre_grid;
   HYPRE_StructMatrix   hypre_mat;
   HYPRE_StructVector   hypre_b, hypre_x;
   HYPRE_StructStencil  hypre_stencil;
   
   HYPRE_StructSolver   hypre_pfmg_data;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * PFMG
 *--------------------------------------------------------------------------*/

void         PFMG(soln, rhs, tol, zero)
Vector      *soln;
Vector      *rhs;
double       tol;
int          zero;
{
   PFModule           *this_module    = ThisPFModule;
   InstanceXtra       *instance_xtra  = PFModuleInstanceXtra(this_module);
   PublicXtra         *public_xtra    = PFModulePublicXtra(this_module);

   HYPRE_StructMatrix  hypre_mat      = instance_xtra -> hypre_mat;
   HYPRE_StructVector  hypre_b        = instance_xtra -> hypre_b;
   HYPRE_StructVector  hypre_x        = instance_xtra -> hypre_x;
   
   HYPRE_StructSolver  hypre_pfmg_data = instance_xtra -> hypre_pfmg_data;

   Grid               *grid           = VectorGrid(rhs);
   Subgrid            *subgrid;
   int                 sg;

   Subvector          *rhs_sub;
   Subvector          *soln_sub;

   double             *rhs_ptr;
   double             *soln_ptr;
   double              value;

   int                 index[3];

   int                 ix,   iy,   iz;
   int                 nx,   ny,   nz;
   int                 nx_v, ny_v, nz_v;
   int                 i, j, k;
   int                 iv;

   int                 num_iterations;
   double              rel_norm;

   /* Copy rhs to hypre_b vector. */
   BeginTiming(public_xtra->time_index_copy_hypre);

   ForSubgridI(sg, GridSubgrids(grid))
   {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
      rhs_sub = VectorSubvector(rhs, sg);

      rhs_ptr = SubvectorData(rhs_sub);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_v = SubvectorNX(rhs_sub);
      ny_v = SubvectorNY(rhs_sub);
      nz_v = SubvectorNZ(rhs_sub);

      iv  = SubvectorEltIndex(rhs_sub, ix, iy, iz);

      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		iv,  nx_v,  ny_v,  nz_v,  1, 1, 1,
		{
		   index[0] = i;
		   index[1] = j;
		   index[2] = k;

		   HYPRE_StructVectorSetValues(hypre_b, index, rhs_ptr[iv]);
		});
   }
   HYPRE_StructVectorAssemble(hypre_b);

   EndTiming(public_xtra->time_index_copy_hypre);

   if (tol > 0.0)
   {
      IfLogging(1)
      {
	 HYPRE_StructPFMGSetLogging(instance_xtra->hypre_pfmg_data, 1);
      }
   }

   /* Invoke the preconditioner using a zero initial guess */
   HYPRE_StructPFMGSetZeroGuess(hypre_pfmg_data);
   HYPRE_StructPFMGSetTol(hypre_pfmg_data, tol);

#if 0
   HYPRE_StructVectorPrint("hypre_b", hypre_b, 0);
   PrintVector("parflow_b", rhs);
#endif

   BeginTiming(public_xtra->time_index_pfmg);

   HYPRE_StructPFMGSolve(hypre_pfmg_data, hypre_mat, hypre_b, hypre_x);

   EndTiming(public_xtra->time_index_pfmg);

   if (tol > 0.0)
   {
      IfLogging(1)
      {
         FILE  *log_file;
	 
	 HYPRE_StructPFMGGetNumIterations(hypre_pfmg_data, &num_iterations);
	 HYPRE_StructPFMGGetFinalRelativeResidualNorm(hypre_pfmg_data, 
						     &rel_norm);

	 log_file = OpenLogFile("PFMG");
	 fprintf(log_file, "PFMG num. its: %i  PFMG Final norm: %12.4e\n", 
		 num_iterations, rel_norm);
	 CloseLogFile(log_file);
      }
   }

   /* Copy solution from hypre_x vector to the soln vector. */
   BeginTiming(public_xtra->time_index_copy_hypre);

   ForSubgridI(sg, GridSubgrids(grid))
   {
      subgrid = SubgridArraySubgrid(GridSubgrids(grid), sg);
      soln_sub = VectorSubvector(soln, sg);

      soln_ptr = SubvectorData(soln_sub);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_v = SubvectorNX(soln_sub);
      ny_v = SubvectorNY(soln_sub);
      nz_v = SubvectorNZ(soln_sub);

      iv  = SubvectorEltIndex(soln_sub, ix, iy, iz);

      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		iv, nx_v, ny_v, nz_v, 1, 1, 1,
		{
		   index[0] = i;
		   index[1] = j;
		   index[2] = k;

		   HYPRE_StructVectorGetValues(hypre_x, index, &value);
		   soln_ptr[iv] = value;
		});
   }
   EndTiming(public_xtra->time_index_copy_hypre);

#if 0
   HYPRE_StructVectorPrint("hypre_x", hypre_x, 0);
   PrintVector("parflow_x", soln);
   exit(1);
#endif
}

/*--------------------------------------------------------------------------
 * PFMGInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PFMGInitInstanceXtra(problem, grid, problem_data,  
				 pf_matrix, temp_data)
Problem      *problem;
Grid         *grid;
ProblemData  *problem_data;
Matrix       *pf_matrix;
double       *temp_data;
{
   PFModule      *this_module        = ThisPFModule;
   PublicXtra    *public_xtra        = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   int                 max_iter      = public_xtra -> max_iter;
   int                 num_pre_relax = public_xtra -> num_pre_relax;
   int                 num_post_relax= public_xtra -> num_post_relax;
   int                 smoother      = public_xtra -> smoother;

   Grid               *mat_grid;
   Subgrid            *subgrid;
   int                 sg;

   Submatrix          *pf_sub;
   double             *cp, *wp, *ep, *sop, *np, *lp, *up;

   double              coeffs[7];
   double              coeffs_symm[4];
   
   int                 i, j, k;
   int                 ix, iy, iz;
   int                 nx, ny, nz;
   int                 nx_m, ny_m, nz_m;
   int                 im;
   int                 stencil_size;
   int                 symmetric;

   int                 full_ghosts[6]          = {1, 1, 1, 1, 1, 1};
   int                 no_ghosts[6]            = {0, 0, 0, 0, 0, 0};
   int                 stencil_indices[7]      = {0, 1, 2, 3, 4, 5, 6};
   int                 stencil_indices_symm[4] = {0, 1, 2, 3};
   int                 index[3];
   int                 ilo[3];
   int                 ihi[3];

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      /* Free the HYPRE grid */
      if (instance_xtra -> hypre_grid)
	 HYPRE_StructGridDestroy(instance_xtra->hypre_grid);

      /* Set the HYPRE grid */
      HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &(instance_xtra->hypre_grid) );

      /* Set local grid extents as global grid values */
      ForSubgridI(sg, GridSubgrids(grid))
      {
	 subgrid = GridSubgrid(grid, sg);

	 ilo[0] = SubgridIX(subgrid);
	 ilo[1] = SubgridIY(subgrid);
	 ilo[2] = SubgridIZ(subgrid);
	 ihi[0] = ilo[0] + SubgridNX(subgrid) - 1;
	 ihi[1] = ilo[1] + SubgridNY(subgrid) - 1;
	 ihi[2] = ilo[2] + SubgridNZ(subgrid) - 1;

	 instance_xtra->dxyz[0] = SubgridDX(subgrid);
	 instance_xtra->dxyz[1] = SubgridDY(subgrid);
	 instance_xtra->dxyz[2] = SubgridDZ(subgrid);
      }		
      HYPRE_StructGridSetExtents(instance_xtra->hypre_grid, ilo, ihi); 
      HYPRE_StructGridAssemble(instance_xtra->hypre_grid);
   }

   /* Reset the HYPRE solver for each recompute of the PC matrix.  
      This reset will require a matrix copy from PF format to HYPRE format. */
   if ( pf_matrix != NULL )
   {
      /* Free old solver data because HYPRE requires a new solver if 
         matrix values change */
      if (instance_xtra->hypre_pfmg_data)
	 HYPRE_StructPFMGDestroy(instance_xtra->hypre_pfmg_data);

      /* For remainder of routine, assume matrix is structured the same for
	 entire nonlinear solve process */
      /* Set stencil parameters */
      stencil_size = MatrixDataStencilSize(pf_matrix);
      if ( !(instance_xtra->hypre_stencil) )
      {
         HYPRE_StructStencilCreate(3, stencil_size, 
				&(instance_xtra->hypre_stencil) );

         for (i = 0; i < stencil_size; i++) 
         {
            HYPRE_StructStencilSetElement(instance_xtra->hypre_stencil, i,
                                         &(MatrixDataStencil(pf_matrix))[i*3]);
         }
      }

      /* Set up new matrix */
      symmetric = MatrixSymmetric(pf_matrix);
      if ( !(instance_xtra->hypre_mat) )
      {
         HYPRE_StructMatrixCreate(MPI_COMM_WORLD, instance_xtra->hypre_grid, 
			       instance_xtra->hypre_stencil,
			       &(instance_xtra->hypre_mat) );
	 HYPRE_StructMatrixSetNumGhost(instance_xtra->hypre_mat, full_ghosts);
         HYPRE_StructMatrixSetSymmetric(instance_xtra->hypre_mat, symmetric);
         HYPRE_StructMatrixInitialize(instance_xtra->hypre_mat);
      }

      /* Set up new right-hand-side vector */
      if ( !(instance_xtra->hypre_b) )
      {
         HYPRE_StructVectorCreate(MPI_COMM_WORLD, 
			       instance_xtra->hypre_grid, 
			       &(instance_xtra->hypre_b) );
	 HYPRE_StructVectorSetNumGhost(instance_xtra->hypre_b, no_ghosts);
	 HYPRE_StructVectorInitialize(instance_xtra->hypre_b);
      }

      /* Set up new solution vector */
      if ( !(instance_xtra->hypre_x) )
      {
         HYPRE_StructVectorCreate(MPI_COMM_WORLD, 
			       instance_xtra->hypre_grid, 
			       &(instance_xtra->hypre_x) );
	 HYPRE_StructVectorSetNumGhost(instance_xtra->hypre_x, full_ghosts);
	 HYPRE_StructVectorInitialize(instance_xtra->hypre_x);
      }
      HYPRE_StructVectorSetConstantValues(instance_xtra->hypre_x, 0.0e0);
      HYPRE_StructVectorAssemble(instance_xtra->hypre_x);

      /* Copy the matrix entries */
      BeginTiming(public_xtra->time_index_copy_hypre);

      mat_grid = MatrixGrid(pf_matrix);
      ForSubgridI(sg, GridSubgrids(mat_grid))
      {
	 subgrid = GridSubgrid(mat_grid, sg);

	 pf_sub  = MatrixSubmatrix(pf_matrix, sg);

	 if (symmetric)
	 {
	    /* Pull off upper diagonal coeffs here for symmetric part */
	    cp      = SubmatrixStencilData(pf_sub, 0);
	    ep      = SubmatrixStencilData(pf_sub, 2);
	    np      = SubmatrixStencilData(pf_sub, 4);
	    up      = SubmatrixStencilData(pf_sub, 6);
	 }
	 else
	 {
	    cp      = SubmatrixStencilData(pf_sub, 0);
	    wp      = SubmatrixStencilData(pf_sub, 1);
	    ep      = SubmatrixStencilData(pf_sub, 2);
	    sop     = SubmatrixStencilData(pf_sub, 3);
	    np      = SubmatrixStencilData(pf_sub, 4);
	    lp      = SubmatrixStencilData(pf_sub, 5);
	    up      = SubmatrixStencilData(pf_sub, 6);
	 }

	 ix = SubgridIX(subgrid);
	 iy = SubgridIY(subgrid);
	 iz = SubgridIZ(subgrid);
	 
	 nx = SubgridNX(subgrid);
	 ny = SubgridNY(subgrid);
	 nz = SubgridNZ(subgrid);
	 
	 nx_m  = SubmatrixNX(pf_sub);
	 ny_m  = SubmatrixNY(pf_sub);
	 nz_m  = SubmatrixNZ(pf_sub);

	 im  = SubmatrixEltIndex(pf_sub,  ix, iy, iz);

	 if (symmetric)
	 {
            BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		      im,  nx_m,  ny_m,  nz_m,  1, 1, 1,
		      {
		         coeffs_symm[0] = cp[im];
			 coeffs_symm[1] = ep[im];
			 coeffs_symm[2] = np[im];
			 coeffs_symm[3] = up[im];
			 index[0] = i;
			 index[1] = j;
			 index[2] = k;
			 HYPRE_StructMatrixSetValues(instance_xtra->hypre_mat, 
						     index, 
						     stencil_size, 
						     stencil_indices_symm, 
						     coeffs_symm);
		      });
	 }
	 else
	 {
            BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		      im,  nx_m,  ny_m,  nz_m,  1, 1, 1,
		      {
		         coeffs[0] = cp[im];
			 coeffs[1] = wp[im];
			 coeffs[2] = ep[im];
			 coeffs[3] = sop[im];
			 coeffs[4] = np[im];
			 coeffs[5] = lp[im];
			 coeffs[6] = up[im];
			 index[0] = i;
			 index[1] = j;
			 index[2] = k;
			 HYPRE_StructMatrixSetValues(instance_xtra->hypre_mat, 
						     index, 
						     stencil_size, 
						     stencil_indices, coeffs);
		      });
	 }
      }   /* End subgrid loop */
      HYPRE_StructMatrixAssemble(instance_xtra->hypre_mat);

      EndTiming(public_xtra->time_index_copy_hypre);

#if 0
      PrintMatrix("parflow_matrix", pf_matrix);
      HYPRE_StructMatrixPrint("hypre_mat", instance_xtra->hypre_mat, 0);
#endif

      /* Set up the PFMG preconditioner */
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD,
				&(instance_xtra->hypre_pfmg_data) );

      HYPRE_StructPFMGSetTol(instance_xtra->hypre_pfmg_data, 1.0e-40);
      /* Set user parameters for PFMG */
      HYPRE_StructPFMGSetMaxIter(instance_xtra->hypre_pfmg_data, max_iter);
      HYPRE_StructPFMGSetNumPreRelax(instance_xtra->hypre_pfmg_data, 
				     num_pre_relax);
      HYPRE_StructPFMGSetNumPostRelax(instance_xtra->hypre_pfmg_data, 
				      num_post_relax);
      /* weighted Jacobi = 1; red-black GS = 2 */
      HYPRE_StructPFMGSetRelaxType(instance_xtra->hypre_pfmg_data, smoother);
      HYPRE_StructPFMGSetDxyz(instance_xtra->hypre_pfmg_data, 	
			      instance_xtra->dxyz);

      HYPRE_StructPFMGSetup(instance_xtra->hypre_pfmg_data, 
			    instance_xtra->hypre_mat, 
			    instance_xtra->hypre_b, instance_xtra->hypre_x);
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * PFMGFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PFMGFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      if (instance_xtra->hypre_pfmg_data)
	 HYPRE_StructPFMGDestroy(instance_xtra->hypre_pfmg_data);
      if (instance_xtra->hypre_mat)
	 HYPRE_StructMatrixDestroy(instance_xtra->hypre_mat);
      if (instance_xtra->hypre_b)
	 HYPRE_StructVectorDestroy(instance_xtra->hypre_b);
      if (instance_xtra->hypre_x)
	 HYPRE_StructVectorDestroy(instance_xtra->hypre_x);
      if (instance_xtra->hypre_stencil)
	 HYPRE_StructStencilDestroy(instance_xtra->hypre_stencil);
      if (instance_xtra->hypre_grid)
	 HYPRE_StructGridDestroy(instance_xtra->hypre_grid);

      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * PFMGNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PFMGNewPublicXtra(char *name)
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char           key[IDB_MAX_KEY_LEN];
   char          *smoother_name;

   NameArray      smoother_switch_na;

   int            smoother;

   public_xtra = ctalloc(PublicXtra, 1);

   sprintf(key, "%s.MaxIter", name);
   public_xtra -> max_iter = GetIntDefault(key, 1);

   sprintf(key, "%s.NumPreRelax", name);
   public_xtra -> num_pre_relax = GetIntDefault(key, 1);

   sprintf(key, "%s.NumPostRelax", name);
   public_xtra -> num_post_relax = GetIntDefault(key, 0);

   /* Use a dummy place holder so that cardinalities match 
      with what HYPRE expects */
   smoother_switch_na = NA_NewNameArray("Dummy WJacobi RBGaussSeidel");
   sprintf(key, "%s.Smoother", name);
   smoother_name = GetStringDefault(key, "WJacobi");
   smoother = NA_NameToIndex(smoother_switch_na, smoother_name);
   if (smoother != 0)
   {
      public_xtra->smoother = NA_NameToIndex(smoother_switch_na, 
					     smoother_name);
   }
   else
   {
      InputError("Error: Invalid value <%s> for key <%s>.\n", 
		 smoother_name, key);
   }

   public_xtra -> time_index_pfmg = RegisterTiming("PFMG");
   public_xtra -> time_index_copy_hypre = RegisterTiming("HYPRE_Copies");

   PFModulePublicXtra(this_module) = public_xtra;

   return this_module;
}

/*-------------------------------------------------------------------------
 * PFMGFreePublicXtra
 *-------------------------------------------------------------------------*/

void  PFMGFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * PFMGSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PFMGSizeOfTempData()
{
   int sz = 0;

   return sz;
}

#endif
