/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/
/******************************************************************************
 *
 *  This module computes the contributions for the spatial discretization of the
 *  kinematic equation for the overland flow boundary condition:KE,KW,KN,KS.
 *
 *  It also computes the derivatives of these terms for inclusion in the Jacobian.
 *
 * Could add a switch statement to handle the diffusion wave also. 
 * -DOK
 *****************************************************************************/

#include "parflow.h"
#include "llnlmath.h"
//#include "llnltyps.h"
/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef void InstanceXtra;

/*-------------------------------------------------------------------------
 * OverlandFlowEval
 *-------------------------------------------------------------------------*/

void    OverlandFlowEval(
   Grid *grid,  /* data struct for computational grid */
   int sg,  /* current subgrid */
   BCStruct *bc_struct,  /* data struct of boundary patch values */
   int ipatch,  /* current boundary patch */
   ProblemData *problem_data, /* Geometry data for problem */
   Vector *pressure,  /* Vector of phase pressures at each block */
   double *ke_v,       /* return array corresponding to the east face KE  */
   double *kw_v,       /* return array corresponding to the west face KW */
   double *kn_v,       /* return array corresponding to the north face KN */
   double *ks_v,       /* return array corresponding to the south face KS */
   int     fcn)             /* Flag determining what to calculate 
			     * fcn = CALCFCN => calculate the function value
			     * fcn = CALCDER => calculate the function 
			     *                  derivative */
{
   PFModule      *this_module   = ThisPFModule;
/*   PublicXtra    *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module); */

//   Grid          *grid;

   Vector      *slope_x           = ProblemDataTSlopeX(problem_data); 
   Vector      *slope_y           = ProblemDataTSlopeY(problem_data);
   Vector      *mannings          = ProblemDataMannings(problem_data);
   Vector      *top               = ProblemDataIndexOfDomainTop(problem_data);

   Subvector     *sx_sub, *sy_sub, *mann_sub, *top_sub, *p_sub;
//   Subvector     *ke_sub, *kw_sub, *kn_sub, *ks_sub;

   double        *sx_dat, *sy_dat, *mann_dat, *top_dat, *pp; 
//   double        *ke_, *kw_, *kn_, *ks_; 
   double        xdir, ydir, dx, dy; 
   double 	 q_lo, q_mid_hi, q_mid_lo, q_hi, sf;

   Subgrid       *subgrid;

   int            ival;
   int 		  *fdir;
/*
   int            ix,   iy,   iz;
   int            nx,   ny,   nz;
   int            nx_p, ny_p, nz_p;
   int            nx_d, ny_d, nz_d;
*/
   int            i, j, k, ip, ic, io, itop;
   int		  i1, j1, k1;


//   ForSubgridI(sg, GridSubgrids(grid))
//   {
      subgrid = GridSubgrid(grid, sg);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);

      p_sub = VectorSubvector(pressure, sg);
/*
      kw_sub = VectorSubvector(Wface_v, sg);
      ke_sub = VectorSubvector(Eface_v, sg);
      kn_sub = VectorSubvector(Nface_v, sg);
      ks_sub = VectorSubvector(Sface_v, sg);
*/
      sx_sub = VectorSubvector(slope_x, sg);
      sy_sub = VectorSubvector(slope_y, sg);
      mann_sub = VectorSubvector(mannings, sg);
      top_sub = VectorSubvector(top, sg);

/*
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid); 
*/
      pp = SubvectorData(p_sub);      
/*
      kw_ = SubvectorData(kw_sub);
      ke_ = SubvectorData(ke_sub);
      kn_ = SubvectorData(kn_sub);
      ks_ = SubvectorData(ks_sub);
*/
      sx_dat = SubvectorData(sx_sub);
      sy_dat = SubvectorData(sy_sub);
      mann_dat = SubvectorData(mann_sub);
      top_dat = SubvectorData(top_sub);

      if(fcn == CALCFCN)
      {

         BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, sg,
	 {
	    if (fdir[2])
	    {
		switch(fdir[2])
		{
			case 1:
                	/* compute east and west faces */
			q_lo = 0.0; /*initialize q_lo for inactive region */
			q_hi = 0.0; /*initialize q_hi for inactive region */
                	/* left cell */
                	i1 = i-1;
                	itop = SubvectorEltIndex(top_sub, i1, j, 0);
                	k1 = (int)top_dat[itop];
			
			if(!(k1<0)) /* active region */
                        {
				io   = SubvectorEltIndex(sx_sub, i1, j, 0);
				ip   = SubvectorEltIndex(p_sub, i1, j, k1);
				ic   = SubvectorEltIndex(p_sub, i, j, k);

                		xdir = 0.0;
				sf = sx_dat[io] - (pp[ic] - pp[ip])/dx;
				if(sx_dat[io] > 0.0)
		   		   xdir = -1.0;
				else if(sx_dat[io] < 0.0)
                   		   xdir = 1.0; 

				q_lo = xdir * (RPowerR(fabs(sx_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));
			}
			
	                /* right cell */
	                i1 = i+1;
	                itop = SubvectorEltIndex(top_sub, i1, j, 0);
	                k1 = (int)top_dat[itop];
			
			if(!(k1<0)) /*active region */
			{
				io   = SubvectorEltIndex(sx_sub, i1, j, 0);
				ip   = SubvectorEltIndex(p_sub, i1, j, k1);

		                xdir = 0.0;
				if(sx_dat[io] > 0.0) 
		                   xdir = -1.0;
				else if(sx_dat[io] < 0.0) 
		                   xdir = 1.0; 

				q_hi = xdir * (RPowerR(fabs(sx_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));
        		}
     
	                /* current cell */
			io   = SubvectorEltIndex(sx_sub, i, j, 0);
			ip   = SubvectorEltIndex(p_sub, i, j, k);

	                xdir = 0.0;
			if(sx_dat[io] > 0.0) 
	                   xdir = -1.0;
			else if(sx_dat[io] < 0.0) 
	                   xdir = 1.0; 

			q_mid = xdir * (RPowerR(fabs(sx_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));

	                /* compute kw and ke - NOTE: io is for current cell */
			kw_v[io] = max(q_lo,0.0) - max(-q_mid,0.0);  
			ke_v[io] = max(q_mid,0.0) - max(-q_hi,0.0);                              


	                /* compute north and south faces */
			q_lo = 0.0; /*initialize q_lo for inactive region */
			q_hi = 0.0; /*initialize q_hi for inactive region */
        	        /* south cell */
	                j1 = j-1;
        	        itop = SubvectorEltIndex(top_sub, i, j1, 0);
        	        k1 = (int)top_dat[itop];
			if(!(k1<1)) /* active region */
			{
				io   = SubvectorEltIndex(sy_sub, i, j1, 0);
				ip   = SubvectorEltIndex(p_sub, i, j1, k1);

        		        ydir = 0.0;
				if(sy_dat[io] > 0.0) 
        		           ydir = -1.0;
				else if(sy_dat[io] < 0.0) 
        		           ydir = 1.0; 

				q_lo = ydir * (RPowerR(fabs(sy_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));
			}

        	        /* north cell */
        	        j1 = j+1;
        	        itop = SubvectorEltIndex(top_sub, i, j1, 0);
        	        k1 = (int)top_dat[itop];
			if(!(k1<0)) /*active region */
			{
				io   = SubvectorEltIndex(sy_sub, i, j1, 0);
				ip   = SubvectorEltIndex(p_sub, i, j1, k1);

        		        ydir = 0.0;
				if(sy_dat[io] > 0.0) 
        		           ydir = -1.0;
				else if(sy_dat[io] < 0.0) 
        		           ydir = 1.0; 

				q_hi = ydir * (RPowerR(fabs(sy_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));
        		}     
        	        
			/* current cell */
			io   = SubvectorEltIndex(sy_sub, i, j, 0);
			ip   = SubvectorEltIndex(p_sub, i, j, k);

        	        ydir = 0.0;
			if(sy_dat[io] > 0.0) 
        	           ydir = -1.0;
			else if(sy_dat[io] < 0.0) 
        	           ydir = 1.0; 

			q_mid = ydir * (RPowerR(fabs(sy_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(5.0/3.0));

        	        /* compute ks and kn - NOTE: io is for current cell */
			ks_v[io] = max(q_lo,0.0) - max(-q_mid,0.0);  
			kn_v[io] = max(q_mid,0.0) - max(-q_hi,0.0);                              

 			break;
        	 }
	    }

	  });
      }
      else /* fcn == CALCDER: derivs of KE,KW,KN,KS w.r.t. current cell (i,j,k) */
      {
         BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, sg,
	 {
	    if (fdir[2])
	    {
		switch(fdir[2])
		{
			case 1:
	                /* compute derivs for east and west faces */
    	     
        	        /* current cell */
			io   = SubvectorEltIndex(sx_sub, i, j, 0);
			ip   = SubvectorEltIndex(p_sub, i, j, k);
	
        	        xdir = 0.0;
			if(sx_dat[io] > 0.0) 
        	           xdir = -1.0;
			else if(sx_dat[io] < 0.0) 
        	           xdir = 1.0; 

			q_mid = xdir * (5.0/3.0)*(RPowerR(fabs(sx_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(2.0/3.0));

        	        /* compute derivs of kw and ke - NOTE: io is for current cell */
			kw_v[io] = - max(-q_mid,0.0);  
			ke_v[io] = max(q_mid,0.0);                              
	

        	        /* compute north and south faces */
        	        ydir = 0.0;
			if(sy_dat[io] > 0.0) 
        	           ydir = -1.0;
			else if(sy_dat[io] < 0.0) 
        	           ydir = 1.0; 

			q_mid = ydir * (5.0/3.0)*(RPowerR(fabs(sy_dat[io]),0.5) / mann_dat[io]) * RPowerR(max((pp[ip]),0.0),(2.0/3.0));

        	        /* compute derivs of ks and kn - NOTE: io is for current cell */
			ks_v[io] = - max(-q_mid,0.0);  
			kn_v[io] = max(q_mid,0.0);  

 			break;
        	 }
	    }

	  });
     }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalInitInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

#if 0
   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
#endif
   instance_xtra = NULL;

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * OverlandFlowEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  OverlandFlowEvalFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   public_xtra = NULL;

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*-------------------------------------------------------------------------
 * OverlandFlowEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void  OverlandFlowEvalFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  OverlandFlowEvalSizeOfTempData()
{
   return 0;
}
