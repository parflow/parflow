/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
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
   NameArray regions;

   int    type; /* input type */
   void  *data; /* pointer to Type structure */
} PublicXtra;

typedef struct
{
 
  Grid   *grid;
  double *temp_data;

} InstanceXtra;

typedef struct
{
   int     num_regions;
   int    *region_indices;
   double *values;
} Type0;

typedef struct
{
   int     num_regions;
   int    *region_indices;
   int     data_from_file;
   double *alphas;
   double *ns;
   char   *alpha_file;
   char   *n_file;
   Vector *alpha_values;
   Vector *n_values;
} Type1;                      /* Van Genuchten Rel. Perm. */

typedef struct
{
   int     num_regions;
   int    *region_indices;
   double *As;
   double *gammas;
} Type2;                      /* Haverkamp, et.al. Rel. Perm. */

typedef struct
{
   int     num_regions;
   int    *region_indices;
} Type3;                      /* Data points for Rel. Perm. */

typedef struct
{
   int     num_regions;
   int    *region_indices;
   int    *degrees;
   double **coefficients;
} Type4;                      /* Polynomial Function for Rel. Perm. */


/*--------------------------------------------------------------------------
 * PhaseRelPerm:
 *    This routine calculates relative permeabilities given a set of
 *    pressures.
 *--------------------------------------------------------------------------*/

void         PhaseRelPerm(phase_rel_perm, phase_pressure, phase_density, 
			  gravity, problem_data, fcn)

Vector      *phase_rel_perm; /* Vector of return rel. perms. at each block */
Vector      *phase_pressure; /* Vector of pressures at each block */
Vector      *phase_density;  /* Vector of densities at each block */
double       gravity;        /* Magnitude of gravity in neg. z direction */
ProblemData *problem_data;   /* Contains geometry info for the problem */
int          fcn;            /* Flag determining what to calculate 
                              * fcn = CALCFCN => calculate the function value
                              * fcn = CALCDER => calculate the function 
                              *                  derivative */

{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Grid          *grid = VectorGrid(phase_rel_perm);

   GrGeomSolid   *gr_solid;

   Type0         *dummy0;
   Type1         *dummy1;
   Type2         *dummy2;
   Type3         *dummy3;
   Type4         *dummy4;

   Subvector     *pr_sub;
   Subvector     *pp_sub;
   Subvector     *pd_sub;
   Subvector     *n_values_sub;
   Subvector     *alpha_values_sub;

   double        *prdat, *ppdat, *pddat;
   double        *n_values_dat, *alpha_values_dat;

   SubgridArray  *subgrids = GridSubgrids(grid);

   Subgrid       *subgrid;

   int            sg;

   int            ix,   iy,   iz;
   int            nx,   ny,   nz;

   int            i, j, k, r, ipr, ipp, ipd;

   int            n_index, alpha_index;

   int            num_regions, *region_indices;
   int            ir, *fdir;

   /* Initialize relative permeabilities to 0.0 */
   PFVConstInit(0.0, phase_rel_perm);

   switch((public_xtra -> type))
   {

   case 0:    /* Constant relative permeability within regions */
   {
      double  *values;
      int      ir;

      dummy0 = (Type0 *)(public_xtra -> data);

      num_regions    = (dummy0 -> num_regions);
      region_indices = (dummy0 -> region_indices);
      values         = (dummy0 -> values);

      /* Compute rel perms for Dirichlet boundary conditions */
      for (ir = 0; ir < num_regions; ir++)
      {
	 gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);
	    pr_sub = VectorSubvector(phase_rel_perm,sg);

	    ix = SubgridIX(subgrid);
	    iy = SubgridIY(subgrid);
	    iz = SubgridIZ(subgrid);

	    nx = SubgridNX(subgrid);
	    ny = SubgridNY(subgrid);
	    nz = SubgridNZ(subgrid);

	    r = SubgridRX(subgrid);

	    prdat = SubvectorData(pr_sub);

	    if ( fcn == CALCFCN )
	    {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
		  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
		  prdat[ipr] = values[ir];
		  
	       });
	    }
	    else  /* fcn = CALCDER */
	    {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
		  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
		  prdat[ipr] = 0.0;
	       });
	    }   /* End else clause */
	 }      /* End subgrid loop */
      }         /* End loop over regions */

      /* Compute rel perms inside regions */
      for (ir = 0; ir < num_regions; ir++)
      {
	 gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);
	    pr_sub = VectorSubvector(phase_rel_perm,sg);

	    ix = SubgridIX(subgrid) - 1;
	    iy = SubgridIY(subgrid) - 1;
	    iz = SubgridIZ(subgrid) - 1;

	    nx = SubgridNX(subgrid) + 2;
	    ny = SubgridNY(subgrid) + 2;
	    nz = SubgridNZ(subgrid) + 2;

	    r = SubgridRX(subgrid);

	    prdat = SubvectorData(pr_sub);
	    if ( fcn == CALCFCN )
	    {
	       GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	       {
		  ipr = SubvectorEltIndex(pr_sub, i, j, k);
		  prdat[ipr] = values[ir];
	       });
	    }
	    else  /* fcn = CALCDER */
	    {
	       GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	       {
		  ipr = SubvectorEltIndex(pr_sub, i, j, k);
		  prdat[ipr] = 0.0;
	       });
	    }   /* End else clause */
	 }      /* End subgrid loop */
      }         /* End loop over regions */
      break;
   }         /* End case 0 */

   case 1:  /* Van Genuchten relative permeability */
   {
      int      data_from_file;
      double  *alphas, *ns, head;
      double   alpha, n, m, opahn, ahnm1, coeff;

      Vector  *n_values, *alpha_values;

      dummy1 = (Type1 *)(public_xtra -> data);

      num_regions    = (dummy1 -> num_regions);
      region_indices = (dummy1 -> region_indices);
      alphas         = (dummy1 -> alphas);
      ns             = (dummy1 -> ns);
      data_from_file = (dummy1->data_from_file);

      /* Compute rel perms for Dirichlet boundary conditions */
      if (data_from_file == 0)  /* alphas and ns given by region */
      {
      for (ir = 0; ir < num_regions; ir++)
      {
	 gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

	    pr_sub = VectorSubvector(phase_rel_perm,sg);
	    pp_sub = VectorSubvector(phase_pressure, sg);
	    pd_sub = VectorSubvector(phase_density,  sg);

	    ix = SubgridIX(subgrid);
	    iy = SubgridIY(subgrid);
	    iz = SubgridIZ(subgrid);

	    nx = SubgridNX(subgrid);
	    ny = SubgridNY(subgrid);
	    nz = SubgridNZ(subgrid);

	    r = SubgridRX(subgrid);

	    prdat = SubvectorData(pr_sub);
	    ppdat = SubvectorData(pp_sub);
	    pddat = SubvectorData(pd_sub);

	    if ( fcn == CALCFCN )
	    {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipd = SubvectorEltIndex(pd_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 1.0;
		  else
		  {
		     alpha      = alphas[ir];
		     n          = ns[ir];
		     m          = 1.0e0 - (1.0e0/n);

		     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head       = fabs(ppdat[ipp]);
		     opahn      = 1.0 + pow(alpha*head,n);
		     ahnm1      = pow(alpha*head,n-1);
		     prdat[ipr] = pow(1.0 - ahnm1/(pow(opahn,m)),2)
		                  /pow(opahn,(m/2));
		  }
	       });
	    }
	    else  /* fcn = CALCDER */
	    {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipd = SubvectorEltIndex(pd_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 0.0;
		  else
		  {
		     alpha    = alphas[ir];
		     n        = ns[ir];
		     m        = 1.0e0 - (1.0e0/n);

		     //head     = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head     = fabs(ppdat[ipp]);
		     opahn    = 1.0 + pow(alpha*head,n);
		     ahnm1    = pow(alpha*head,n-1);
		     coeff    = 1.0 - ahnm1*pow(opahn,-m);

		     prdat[ipr] = 2.0*(coeff/(pow(opahn,(m/2))))
		                   *((n-1)*pow(alpha*head,n-2)*alpha
				   *pow(opahn,-m)
				   - ahnm1*m*pow(opahn,-(m+1))*n*alpha*ahnm1)
			           + pow(coeff,2)*(m/2)*pow(opahn,(-(m+2)/2))
			            *n*alpha*ahnm1;
		  }
	       });
	    }   /* End else clause */
	 }      /* End subgrid loop */
      }         /* End loop over regions */
      }         /* End if data not from file */
   
      else if (data_from_file == 1)  /* ns and alphas from pfb file */
      {
	 gr_solid = ProblemDataGrDomain(problem_data);
	 n_values = dummy1->n_values;
	 alpha_values = dummy1->alpha_values;

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

	    pr_sub = VectorSubvector(phase_rel_perm,sg);
	    pp_sub = VectorSubvector(phase_pressure, sg);
	    pd_sub = VectorSubvector(phase_density,  sg);

	    n_values_sub = VectorSubvector(n_values, sg);
	    alpha_values_sub = VectorSubvector(alpha_values, sg);

	    ix = SubgridIX(subgrid);
	    iy = SubgridIY(subgrid);
	    iz = SubgridIZ(subgrid);

	    nx = SubgridNX(subgrid);
	    ny = SubgridNY(subgrid);
	    nz = SubgridNZ(subgrid);

	    r = SubgridRX(subgrid);

	    prdat = SubvectorData(pr_sub);
	    ppdat = SubvectorData(pp_sub);
	    pddat = SubvectorData(pd_sub);

	    n_values_dat = SubvectorData(n_values_sub);
	    alpha_values_dat = SubvectorData(alpha_values_sub);

	    if ( fcn == CALCFCN )
	    {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipd = SubvectorEltIndex(pd_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);

		  n_index = SubvectorEltIndex(n_values_sub, i, j, k);
		  alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 1.0;
		  else
		  {
		     alpha      = alpha_values_dat[alpha_index];
		     n          = n_values_dat[n_index];
		     m          = 1.0e0 - (1.0e0/n);

		     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head       = fabs(ppdat[ipp]);
		     opahn      = 1.0 + pow(alpha*head,n);
		     ahnm1      = pow(alpha*head,n-1);
		     prdat[ipr] = pow(1.0 - ahnm1/(pow(opahn,m)),2)
		                  /pow(opahn,(m/2));
		  }
	       });
	    }
	    else  /* fcn = CALCDER */
	    {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipd = SubvectorEltIndex(pd_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);

		  n_index = SubvectorEltIndex(n_values_sub, i, j, k);
		  alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 0.0;
		  else
		  {
		     alpha    = alpha_values_dat[alpha_index];
		     n        = n_values_dat[n_index];
		     m        = 1.0e0 - (1.0e0/n);

		     //head     = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head     = fabs(ppdat[ipp]);
		     opahn    = 1.0 + pow(alpha*head,n);
		     ahnm1    = pow(alpha*head,n-1);
		     coeff    = 1.0 - ahnm1*pow(opahn,-m);

		     prdat[ipr] = 2.0*(coeff/(pow(opahn,(m/2))))
		                   *((n-1)*pow(alpha*head,n-2)*alpha
				   *pow(opahn,-m)
				   - ahnm1*m*pow(opahn,-(m+1))*n*alpha*ahnm1)
			           + pow(coeff,2)*(m/2)*pow(opahn,(-(m+2)/2))
			            *n*alpha*ahnm1;
		  }
	       });
	    }   /* End else clause */
	 }      /* End subgrid loop */
      }         /* End if data_from_file */

      /* Compute rel. perms. on interior */
      if (data_from_file == 0)  /* alphas and ns given by region */
      {
      for (ir = 0; ir < num_regions; ir++)
      {
         gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

	    pr_sub = VectorSubvector(phase_rel_perm, sg);
	    pp_sub = VectorSubvector(phase_pressure, sg);
	    pd_sub = VectorSubvector(phase_density,  sg);

	    ix = SubgridIX(subgrid) - 1;
	    iy = SubgridIY(subgrid) - 1;
	    iz = SubgridIZ(subgrid) - 1;

	    nx = SubgridNX(subgrid) + 2;
	    ny = SubgridNY(subgrid) + 2;
	    nz = SubgridNZ(subgrid) + 2;

            r = SubgridRX(subgrid);

	    prdat = SubvectorData(pr_sub);
	    ppdat = SubvectorData(pp_sub);
	    pddat = SubvectorData(pd_sub);

	    if ( fcn == CALCFCN )
	    {
	       GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  ipd = SubvectorEltIndex(pd_sub, i, j, k);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 1.0;
		  else
		  {
		     alpha      = alphas[ir];
		     n          = ns[ir];
		     m          = 1.0e0 - (1.0e0/n);

		     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head       = fabs(ppdat[ipp]);
		     opahn      = 1.0 + pow(alpha*head,n);
		     ahnm1      = pow(alpha*head,n-1);
		     prdat[ipr] = pow(1.0 - ahnm1/(pow(opahn,m)),2)
		                  /pow(opahn,(m/2));
		  }
	       });
	    }    /* End if clause */
	    else /* fcn = CALCDER */
	    {
	       GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  ipd = SubvectorEltIndex(pd_sub, i, j, k);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 0.0;
		  else
		  {
		     alpha    = alphas[ir];
		     n        = ns[ir];
		     m        = 1.0e0 - (1.0e0/n);

		     //head     = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head     = fabs(ppdat[ipp]);
		     opahn    = 1.0 + pow(alpha*head,n);
		     ahnm1    = pow(alpha*head,n-1);
		     coeff    = 1.0 - ahnm1*pow(opahn,-m);

		     prdat[ipr] = 2.0*(coeff/(pow(opahn,(m/2))))
		                   *((n-1)*pow(alpha*head,n-2)*alpha
				   *pow(opahn,-m)
				   - ahnm1*m*pow(opahn,-(m+1))*n*alpha*ahnm1)
			           + pow(coeff,2)*(m/2)*pow(opahn,(-(m+2)/2))
			            *n*alpha*ahnm1;
		  }
	       });
	    }   /* End else clause */
	 }      /* End subgrid loop */
      }         /* End subregion loop */
      }         /* End if data not given by file */
      else if (data_from_file == 1) /* alphas and ns given in pfb files */
      {
         gr_solid = ProblemDataGrDomain(problem_data);
	 n_values = dummy1->n_values;
	 alpha_values = dummy1->alpha_values;

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

	    pr_sub = VectorSubvector(phase_rel_perm, sg);
	    pp_sub = VectorSubvector(phase_pressure, sg);
	    pd_sub = VectorSubvector(phase_density,  sg);

	    n_values_sub = VectorSubvector(n_values, sg);
	    alpha_values_sub = VectorSubvector(alpha_values, sg);

	    ix = SubgridIX(subgrid) - 1;
	    iy = SubgridIY(subgrid) - 1;
	    iz = SubgridIZ(subgrid) - 1;

	    nx = SubgridNX(subgrid) + 2;
	    ny = SubgridNY(subgrid) + 2;
	    nz = SubgridNZ(subgrid) + 2;

            r = SubgridRX(subgrid);

	    prdat = SubvectorData(pr_sub);
	    ppdat = SubvectorData(pp_sub);
	    pddat = SubvectorData(pd_sub);

	    n_values_dat = SubvectorData(n_values_sub);
	    alpha_values_dat = SubvectorData(alpha_values_sub);

	    if ( fcn == CALCFCN )
	    {
	       GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  ipd = SubvectorEltIndex(pd_sub, i, j, k);

		  n_index = SubvectorEltIndex(n_values_sub, i, j, k);
		  alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 1.0;
		  else
		  {
		     alpha      = alpha_values_dat[alpha_index];
		     n          = n_values_dat[n_index];
		     m          = 1.0e0 - (1.0e0/n);

		     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head       = fabs(ppdat[ipp]);
		     opahn      = 1.0 + pow(alpha*head,n);
		     ahnm1      = pow(alpha*head,n-1);
		     prdat[ipr] = pow(1.0 - ahnm1/(pow(opahn,m)),2)
		                  /pow(opahn,(m/2));
		  }
	       });
	    }    /* End if clause */
	    else /* fcn = CALCDER */
	    {
	       GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  ipd = SubvectorEltIndex(pd_sub, i, j, k);

		  n_index     = SubvectorEltIndex(n_values_sub, i, j, k);
		  alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 0.0;
		  else
		  {
		     alpha    = alpha_values_dat[alpha_index];
		     n        = n_values_dat[n_index];
		     m        = 1.0e0 - (1.0e0/n);

		     //head     = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
		     head     = fabs(ppdat[ipp]);
		     opahn    = 1.0 + pow(alpha*head,n);
		     ahnm1    = pow(alpha*head,n-1);
		     coeff    = 1.0 - ahnm1*pow(opahn,-m);

		     prdat[ipr] = 2.0*(coeff/(pow(opahn,(m/2))))
		                   *((n-1)*pow(alpha*head,n-2)*alpha
				   *pow(opahn,-m)
				   - ahnm1*m*pow(opahn,-(m+1))*n*alpha*ahnm1)
			           + pow(coeff,2)*(m/2)*pow(opahn,(-(m+2)/2))
			            *n*alpha*ahnm1;
		  }
	       });
	    }   /* End else clause */
	 }      /* End subgrid loop */
      }         /* End if data given by file */
      break;
   }         /* End case 1 */
  
   case 2:   /* Haverkamp et.al. relative permeability */
   {
      double  *As, *gammas, head, tmp;

      dummy2 = (Type2 *)(public_xtra -> data);

      num_regions    = (dummy2 -> num_regions);
      region_indices = (dummy2 -> region_indices);
      As             = (dummy2 -> As);
      gammas         = (dummy2 -> gammas);

      /* Compute rel. perms. for Dirichlet BC's */
      for (ir = 0; ir < num_regions; ir++)
      {
         gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

            pr_sub = VectorSubvector(phase_rel_perm, sg);
            pp_sub = VectorSubvector(phase_pressure, sg);
            pd_sub = VectorSubvector(phase_density,  sg);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            r = SubgridRX(subgrid);

            prdat = SubvectorData(pr_sub);
            ppdat = SubvectorData(pp_sub);
            pddat = SubvectorData(pd_sub);

            if ( fcn == CALCFCN )
            {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipd = SubvectorEltIndex(pd_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);

		  if (ppdat[ipp] >= 0.0)
		     prdat[ipr] = 1.0;
		  else
		  {
                     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
                     head       = fabs(ppdat[ipp]);
		     tmp        = As[ir] + pow(head, gammas[ir]);
                     prdat[ipr] = As[ir] / tmp;
                  }
               });
            }    /* End if clause */
            else /* fcn = CALCDER */
            {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipd = SubvectorEltIndex(pd_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);

		  if (ppdat[ipp] >= 0.0)
                     prdat[ipr] = 0.0;
                  else
                  {
                     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
                     head       = fabs(ppdat[ipp]);
		     tmp        = pow(head, gammas[ir]);
                     prdat[ipr] = As[ir] * gammas[ir] 
		                  * pow(head, gammas[ir]-1) / pow(tmp, 2);
                  }
               });
            }   /* End else clause */
         }      /* End subgrid loop */
      }         /* End subregion loop */

      /* Compute rel. perms. on interior */
      for (ir = 0; ir < num_regions; ir++)
      {
         gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

            pr_sub = VectorSubvector(phase_rel_perm, sg);
            pp_sub = VectorSubvector(phase_pressure, sg);
            pd_sub = VectorSubvector(phase_density,  sg);

            ix = SubgridIX(subgrid) - 1;
            iy = SubgridIY(subgrid) - 1;
            iz = SubgridIZ(subgrid) - 1;

            nx = SubgridNX(subgrid) + 2;
            ny = SubgridNY(subgrid) + 2;
            nz = SubgridNZ(subgrid) + 2;

            r = SubgridRX(subgrid);

            prdat = SubvectorData(pr_sub);
            ppdat = SubvectorData(pp_sub);
            pddat = SubvectorData(pd_sub);

            if ( fcn == CALCFCN )
            {
               GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
               {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  ipd = SubvectorEltIndex(pd_sub, i, j, k);

                  if (ppdat[ipp] >= 0.0)
                     prdat[ipr] = 1.0;
                  else
                  {
                     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
                     head       = fabs(ppdat[ipp]);
		     tmp        = As[ir] + pow(head, gammas[ir]);
                     prdat[ipr] = As[ir] / tmp;
                  }
               });
            }    /* End if clause */
            else /* fcn = CALCDER */
            {
               GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
               {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);
                  ipd = SubvectorEltIndex(pd_sub, i, j, k);

                  if (ppdat[ipp] >= 0.0)
                     prdat[ipr] = 0.0;
                  else
                  {
                     //head       = fabs(ppdat[ipp])/(pddat[ipd]*gravity);
                     head       = fabs(ppdat[ipp]);
		     tmp        = pow(head, gammas[ir]);
                     prdat[ipr] = As[ir] * gammas[ir] 
		                  * pow(head, gammas[ir]-1) / pow(tmp, 2);
                  }
               });
            }   /* End else clause */
         }      /* End subgrid loop */
      }         /* End subregion loop */

      break;
   }         /* End case 2 */
      
   case 3:   /* Data relative permeability */
   {
      dummy3 = (Type3 *)(public_xtra -> data);

      if(!amps_Rank(amps_CommWorld))
	 printf("Data curves for rel perms not supported currently.\n");
      break;
   }         /* End case 3 */
      
   case 4:   /* Polynomial function of pressure relative permeability */
   {
      int     *degrees, dg;
      double **coefficients, *region_coeffs;

      dummy4 = (Type4 *)(public_xtra -> data);

      num_regions    = (dummy4 -> num_regions);
      region_indices = (dummy4 -> region_indices);
      degrees        = (dummy4 -> degrees);
      coefficients   = (dummy4 -> coefficients);

      /* Compute rel. perms. for Dirichlet BC's */
      for (ir = 0; ir < num_regions; ir++)
      {
         gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
	 region_coeffs = coefficients[ir];

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

            pr_sub = VectorSubvector(phase_rel_perm, sg);
            pp_sub = VectorSubvector(phase_pressure, sg);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);
            iz = SubgridIZ(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);
            nz = SubgridNZ(subgrid);

            r = SubgridRX(subgrid);

            prdat = SubvectorData(pr_sub);
            ppdat = SubvectorData(pp_sub);

            if ( fcn == CALCFCN )
            {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
		  if (ppdat[ipp] == 0.0)
		     prdat[ipr] = region_coeffs[0];
		  else
                  {
		     prdat[ipr] = 0.0;
		     for (dg = 0; dg < degrees[ir]+1; dg++)
		     {
		        prdat[ipr] += region_coeffs[dg] * pow(ppdat[ipp],dg);
		     }
                  }
               });
            }    /* End if clause */
            else /* fcn = CALCDER */
            {
	       GrGeomSurfLoop(i, j, k, fdir, gr_solid, r, ix, iy, iz, 
			      nx, ny, nz,
	       {
                  ipr = SubvectorEltIndex(pr_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
                  ipp = SubvectorEltIndex(pp_sub, 
					  i+fdir[0], j+fdir[1], k+fdir[2]);
		  if (ppdat[ipp] == 0.0)
                     prdat[ipr] = 0.0;
                  else
                  {
		     prdat[ipr] = 0.0;
		     for (dg = 1; dg < degrees[ir]+1; dg++)
		     {
		        prdat[ipr] += region_coeffs[dg] * dg
			              * pow(ppdat[ipp],(dg-1));
		     }
                  }
               });
            }   /* End else clause */
         }      /* End subgrid loop */
      }         /* End subregion loop */

      /* Compute rel. perms. in interior */
      for (ir = 0; ir < num_regions; ir++)
      {
         gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
	 region_coeffs = coefficients[ir];

	 ForSubgridI(sg, subgrids)
	 {
	    subgrid = SubgridArraySubgrid(subgrids, sg);

            pr_sub = VectorSubvector(phase_rel_perm, sg);
            pp_sub = VectorSubvector(phase_pressure, sg);

            ix = SubgridIX(subgrid) - 1;
            iy = SubgridIY(subgrid) - 1;
            iz = SubgridIZ(subgrid) - 1;

            nx = SubgridNX(subgrid) + 2;
            ny = SubgridNY(subgrid) + 2;
            nz = SubgridNZ(subgrid) + 2;

            r = SubgridRX(subgrid);

            prdat = SubvectorData(pr_sub);
            ppdat = SubvectorData(pp_sub);

            if ( fcn == CALCFCN )
            {
               GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
               {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);

		  if (ppdat[ipp] == 0.0)
		     prdat[ipr] = region_coeffs[0];
		  else
		  {
		     prdat[ipr] = 0.0;
		     for (dg = 0; dg < degrees[ir]+1; dg++)
		     {
		        prdat[ipr] += region_coeffs[dg]*pow(ppdat[ipp],dg);
		     }
		  }
               });
            }    /* End if clause */
            else /* fcn = CALCDER */
            {
               GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
               {
                  ipr = SubvectorEltIndex(pr_sub, i, j, k);
                  ipp = SubvectorEltIndex(pp_sub, i, j, k);

                  if (ppdat[ipp] == 0.0)
                     prdat[ipr] = 0.0;
                  else
                  {
		     prdat[ipr] = 0.0;
		     for (dg = 1; dg < degrees[ir]+1; dg++)
		     {
		        prdat[ipr] += region_coeffs[dg] * dg
			              * pow(ppdat[ipp],(dg-1));
		     }
                  }
               });
            }   /* End else clause */
         }      /* End subgrid loop */
      }         /* End subregion loop */
		  
      break;
   }         /* End case 4 */
      
   }         /* End switch */

}

/*--------------------------------------------------------------------------
 * PhaseRelPermInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseRelPermInitInstanceXtra(grid, temp_data)

Grid   *grid;
double *temp_data;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   Type1         *dummy1;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      /* set new data */
      (instance_xtra ->grid) = grid;

      /* Use a spatially varying field */
      if (public_xtra ->type == 1) 
      {
	 dummy1 = (Type1 *)(public_xtra -> data);
         if ((dummy1->data_from_file) == 1)
	 {
	    (dummy1 -> n_values) = NewTempVector(grid, 1, 1);
	    (dummy1 -> alpha_values) = NewTempVector(grid, 1, 1);
	 }
      }

   }

   if ( temp_data != NULL )
   {
      (instance_xtra->temp_data) = temp_data;

      /* Uses a spatially varying field */
      if (public_xtra ->type == 1)  
      {
	 dummy1 = (Type1 *)(public_xtra -> data);
	 if ( (dummy1->data_from_file) == 1)
	 {
	    SetTempVectorData((dummy1 -> n_values), temp_data);
	    temp_data += SizeOfVector(dummy1 ->n_values);
	    SetTempVectorData((dummy1 -> alpha_values), temp_data);
	    temp_data += SizeOfVector(dummy1 ->alpha_values);

	    ReadPFBinary((dummy1 ->alpha_file), 
			 (dummy1 ->alpha_values));
	    ReadPFBinary((dummy1 ->n_file), 
			 (dummy1 ->n_values));
	 }
      }
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseRelPermFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  PhaseRelPermFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * PhaseRelPermNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PhaseRelPermNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   Type0            *dummy0;
   Type1            *dummy1;
   Type2            *dummy2;
   Type3            *dummy3;
   Type4            *dummy4;

   int               num_regions, ir, ic;
   
   char *switch_name;
   char *region;
   
   char key[IDB_MAX_KEY_LEN];

   NameArray       type_na;

   /*----------------------------------------------------------
    * The name array to map names to switch values 
    *----------------------------------------------------------*/
   type_na = NA_NewNameArray("Constant VanGenuchten Haverkamp Data Polynomial");

   public_xtra = ctalloc(PublicXtra, 1);

   switch_name = GetString("Phase.RelPerm.Type");
   public_xtra -> type = NA_NameToIndex(type_na, switch_name);

   
   switch_name = GetString("Phase.RelPerm.GeomNames");
   public_xtra -> regions = NA_NewNameArray(switch_name);
   
   num_regions = NA_Sizeof(public_xtra -> regions);

   switch((public_xtra -> type))
   {
      case 0:
      {
	 dummy0 = ctalloc(Type0, 1);

	 dummy0 -> num_regions = num_regions;
	 
	 (dummy0 -> region_indices) = ctalloc(int,    num_regions);
	 (dummy0 -> values        ) = ctalloc(double, num_regions);
	 
	 for (ir = 0; ir < num_regions; ir++)
	 {
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy0 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);
	    
	    sprintf(key, "Geom.%s.RelPerm.Value", region);
	    dummy0 -> values[ir] = GetDouble(key);
	 }
	 
	 (public_xtra -> data) = (void *) dummy0;
	 
	 break;
      }
      
      case 1:
      {
	 dummy1 = ctalloc(Type1, 1);

	 sprintf(key, "Phase.RelPerm.VanGenuchten.File");
	 dummy1->data_from_file = GetIntDefault(key,0);

	 if ( (dummy1->data_from_file) == 0)
	 {
	    dummy1 -> num_regions = num_regions;

	    (dummy1 -> region_indices) = ctalloc(int,    num_regions);
	    (dummy1 -> alphas        ) = ctalloc(double, num_regions);
	    (dummy1 -> ns            ) = ctalloc(double, num_regions);
	 
	    for (ir = 0; ir < num_regions; ir++)
	    {
	       region = NA_IndexToName(public_xtra -> regions, ir);
	    
	       dummy1 -> region_indices[ir] = 
		 NA_NameToIndex(GlobalsGeomNames, region);

	       sprintf(key, "Geom.%s.RelPerm.Alpha", region);
	       dummy1 -> alphas[ir] = GetDouble(key);

	       sprintf(key, "Geom.%s.RelPerm.N", region);
	       dummy1 -> ns[ir] = GetDouble(key);
	    }
	    
	    dummy1->alpha_file = NULL;
	    dummy1->n_file = NULL;
	    dummy1->alpha_values = NULL;
	    dummy1->n_values = NULL;
	 }
	 else
	 {
	    sprintf(key, "Geom.%s.RelPerm.Alpha.Filename", "domain");
	    dummy1->alpha_file = GetString(key);
	    sprintf(key, "Geom.%s.RelPerm.N.Filename", "domain");
	    dummy1->n_file = GetString(key);
	      
	    dummy1->num_regions = 0;
	    dummy1->region_indices = NULL;
	    dummy1->alphas = NULL;
	    dummy1->ns = NULL;
	 }

	 (public_xtra ->data) = (void *) dummy1;
	 
	 break;
      }
      
      case 2:
      {
	 dummy2 = ctalloc(Type2, 1);

	 dummy2 -> num_regions = num_regions;
	 
	 (dummy2 -> region_indices) = ctalloc(int,    num_regions);
	 (dummy2 -> As            ) = ctalloc(double, num_regions);
	 (dummy2 -> gammas        ) = ctalloc(double, num_regions);
	 
	 for (ir = 0; ir < num_regions; ir++)
	 {
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy2 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);

	    sprintf(key, "Geom.%s.RelPerm.A", region);
	    dummy2 -> As[ir] = GetDouble(key);

	    sprintf(key, "Geom.%s.RelPerm.gamma", region);
	    dummy2 -> gammas[ir] = GetDouble(key);
	 }
	 
	 (public_xtra -> data) = (void *) dummy2;
	 
	 break;
      }
      
      case 3:
      {
	 dummy3 = ctalloc(Type3, 1);

	 dummy3 -> num_regions = num_regions;

	 (dummy3 -> region_indices) = ctalloc(int,    num_regions);
	 
	 for (ir = 0; ir < num_regions; ir++)
	 {
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy3 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);
	 }
	 
	 (public_xtra -> data) = (void *) dummy3;
	 
	 break;
      }
      
      case 4:
      {
	 int degree;
	 
	 dummy4 = ctalloc(Type4, 1);

	 dummy4 -> num_regions = num_regions;

	 (dummy4 -> region_indices) = ctalloc(int,     num_regions);
	 (dummy4 -> degrees)        = ctalloc(int,     num_regions);
	 (dummy4 -> coefficients)   = ctalloc(double*, num_regions);
	 
	 for (ir = 0; ir < num_regions; ir++)
	 {
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy4 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);

	    sprintf(key, "Geom.%s.RelPerm.Degree", region);
	    dummy4 -> degrees[ir] = GetInt(key);
	    
	    degree = (dummy4 -> degrees[ir]);
	    dummy4 -> coefficients[ir] = ctalloc(double, degree+1);
	    
	    for (ic = 0; ic < degree+1; ic++)
	    {
	       sprintf(key, "Geom.%s.RelPerm.Coeff.%d", region, ic);
	       dummy4 -> coefficients[ir][ic] = GetDouble(key);
	    }
	 }
	 
	 (public_xtra -> data) = (void *) dummy4;

	 break;
      }

      default:
      {
	 InputError("Error: invalid type <%s> for key <%s>\n",
		    switch_name, key);
      }

   }     /* End switch */

   NA_FreeNameArray(type_na);
   
   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*--------------------------------------------------------------------------
 * PhaseRelPermFreePublicXtra
 *--------------------------------------------------------------------------*/

void  PhaseRelPermFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   Type0       *dummy0;
   Type1       *dummy1;
   Type2       *dummy2;
   Type3       *dummy3;
   Type4       *dummy4;

   int          num_regions, ir;

   if (public_xtra )
   {

      NA_FreeNameArray(public_xtra -> regions);

      switch((public_xtra -> type))
     {
     case 0:
     {
        dummy0 = (Type0 *)(public_xtra -> data);

	tfree(dummy0 -> region_indices);
	tfree(dummy0 -> values);
	tfree(dummy0);

	break;
     }
     case 1:
     {
        dummy1 = (Type1 *)(public_xtra -> data);

	if (dummy1->data_from_file == 1)
	{
	   FreeTempVector(dummy1->alpha_values);
	   FreeTempVector(dummy1->n_values);
	}

	tfree(dummy1 -> region_indices);
	tfree(dummy1 -> alphas);
	tfree(dummy1 -> ns);
	tfree(dummy1);

	break;
     }

     case 2:
     {
        dummy2 = (Type2 *)(public_xtra -> data);

	tfree(dummy2 -> region_indices);
	tfree(dummy2 -> As);
	tfree(dummy2 -> gammas);
	tfree(dummy2);

	break;
     }

     case 3:
     {
        dummy3 = (Type3 *)(public_xtra -> data);

	tfree(dummy3 -> region_indices);
	tfree(dummy3);

	break;
     }

     case 4:
     {
        dummy4 = (Type4 *)(public_xtra -> data);

	num_regions = (dummy4 -> num_regions);
	for (ir = 0; ir < num_regions; ir++)
	{
	   tfree(dummy4 -> coefficients[ir]);
	}

	tfree(dummy4 -> region_indices);
	tfree(dummy4 -> degrees);
	tfree(dummy4 -> coefficients);
	tfree(dummy4);

	break;
     }

     }
     tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * PhaseRelPermSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PhaseRelPermSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Type1         *dummy1;

   int  sz = 0;

   if (public_xtra->type ==1)
   {
      dummy1 = (Type1 *)(public_xtra -> data);
      if ((dummy1->data_from_file) == 1)
      {
         /* add local TempData size to `sz' */
	 sz += SizeOfVector(dummy1 -> n_values);
	 sz += SizeOfVector(dummy1 -> alpha_values);
      }
   }

   return sz;
}
