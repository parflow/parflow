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
*
*  This module computes the contributions for the spatial discretization of the
*  diffusive wave approximation for the overland flow boundary condition:KE,KW,KN,KS.
*
*  It also computes the derivatives of these terms for inclusion in the Jacobian.
*
* Could add a switch statement to handle the Kinemative wave approx. also.
* @LEC, @RMM
*****************************************************************************/
#include "parflow.h"
#include "llnlmath.h"
//#include "llnltyps.h"
/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef void InstanceXtra;

/*---------------------------------------------------------------------
 * Define macros for function evaluation
 *---------------------------------------------------------------------*/
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)

/*-------------------------------------------------------------------------
 * OverlandFlowEval
 *-------------------------------------------------------------------------*/

void    OverlandFlowEvalDiff(
                             Grid *       grid, /* data struct for computational grid */
                             int          sg, /* current subgrid */
                             BCStruct *   bc_struct, /* data struct of boundary patch values */
                             int          ipatch, /* current boundary patch */
                             ProblemData *problem_data, /* Geometry data for problem */
                             Vector *     pressure, /* Vector of phase pressures at each block */
                             Vector *     old_pressure, /* Vector of phase pressures at previous time */
                             double *     ke_v, /* return array corresponding to the east face KE  */
                             double *     kw_v, /* return array corresponding to the west face KW */
                             double *     kn_v, /* return array corresponding to the north face KN */
                             double *     ks_v, /* return array corresponding to the south face KS */
                             double *     ke_vns, /* return array corresponding to the nonsymetric east face KE derivative  */
                             double *     kw_vns, /* return array corresponding to the nonsymetricwest face KW derivative */
                             double *     kn_vns, /* return array corresponding to the nonsymetricnorth face KN derivative */
                             double *     ks_vns, /* return array corresponding to the nonsymetricsouth face KS derivative*/
                             double *     qx_v, /* return array corresponding to the flux in x-dir */
                             double *     qy_v, /* return array corresponding to the flux in y-dir */
                             int          fcn) /* Flag determining what to calculate
                                                * fcn = CALCFCN => calculate the function value
                                                * fcn = CALCDER => calculate the function
                                                *                  derivative */
{
  PFModule      *this_module = ThisPFModule;

  Vector      *slope_x = ProblemDataTSlopeX(problem_data);
  Vector      *slope_y = ProblemDataTSlopeY(problem_data);
  Vector      *mannings = ProblemDataMannings(problem_data);
  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);

  // printf("overland_eval_diffusive called\n");
  Subvector     *sx_sub, *sy_sub, *mann_sub, *top_sub, *p_sub, *op_sub;

  Subgrid      *subgrid;

  double        *sx_dat, *sy_dat, *mann_dat, *top_dat, *pp, *opp;

  double xdir, ydir;
  double q_lo, q_mid, q_hi;
  double q_v[4], slope_fx_lo, slope_fx_hi, slope_fx_mid;
  double slope_fy_lo, slope_fy_hi, slope_fy_mid, dx, dy;
  double coeff, Pmean, P2, P3, Pdel, Pcen;
  double slope_mean, manning, s1, s2, Sf_mag;
  double Press_x, Press_y, Sf_x, Sf_y, Sf_xo, Sf_yo;
  double Pupx, Pupy, Pupox, Pupoy, Pdown, Pdowno;

  int ival, sy_v, step;
  int            *fdir;

  int i, ii, j, k, ip, ip2, ip3, ip4, ip0, io, itop;
  int i1, j1, k1, k0x, k0y, iojm1, iojp1, ioip1, ioim1;
  /* @RMM get grid from global (assuming this is comp grid) to pass to CLM */
  int gnx = BackgroundNX(GlobalsBackground);
  int gny = BackgroundNY(GlobalsBackground);

  p_sub = VectorSubvector(pressure, sg);
  op_sub = VectorSubvector(old_pressure, sg);
  sx_sub = VectorSubvector(slope_x, sg);
  sy_sub = VectorSubvector(slope_y, sg);
  mann_sub = VectorSubvector(mannings, sg);
  top_sub = VectorSubvector(top, sg);

  pp = SubvectorData(p_sub);
  opp = SubvectorData(op_sub);

  sx_dat = SubvectorData(sx_sub);
  sy_dat = SubvectorData(sy_sub);
  mann_dat = SubvectorData(mann_sub);
  top_dat = SubvectorData(top_sub);

  subgrid = GridSubgrid(grid, sg);
  dx = SubgridDX(subgrid);
  dy = SubgridDY(subgrid);

  sy_v = SubvectorNX(top_sub);



  if (fcn == CALCFCN)
  {

    BCStructPatchLoopOvrlnd(i, j, k, fdir, ival, bc_struct, ipatch, sg,
    {
      if (fdir[2] == 1)
      {
        io = SubvectorEltIndex(sx_sub, i, j, 0);
        itop = SubvectorEltIndex(top_sub, i, j, 0);

        k1 = (int)top_dat[itop];
        k0x = (int)top_dat[itop - 1];
        k0y = (int)top_dat[itop - sy_v];
        double ov_epsilon= 1.0e-5;
        //printf("i=%d j=%d k=%d k1=%d k0x=%d k0y=%d\n",i,j,k,k1, k0x, k0y);


        if (k1 >= 0)
        {
          ip = SubvectorEltIndex(p_sub, i, j, k1);
          Pupx = pfmax(pp[ip+1],0.0);
          Pupy = pfmax(pp[ip+sy_v],0.0);
          Pupox = pfmax(opp[ip+1],0.0);
          Pupoy = pfmax(opp[ip+sy_v],0.0);
          Pdown = pfmax(pp[ip],0.0);
          Pdowno = pfmax(opp[ip],0.0);

          Sf_x = sx_dat[io]+(Pupx - Pdown)/dx;
          Sf_y = sy_dat[io]+(Pupy - Pdown)/dy;

          Sf_xo = sx_dat[io] +(Pupox - Pdowno)/dx;
          Sf_yo = sy_dat[io] +(Pupoy - Pdowno)/dy;

          //printf("i=%d j=%d k=%d k1=%d pdown=%f pdowno=%f \n",i,j,k,k1, k0x, k0y, Pdown, Pdowno);
          //printf("i=%d j=%d k=%d k1=%d P=%f oldP=%f \n",i,j,k,k1,Pdown, Pdowno);

          Sf_mag = RPowerR(Sf_xo*Sf_xo+Sf_yo*Sf_yo,0.5); //+ov_epsilon;
          if (Sf_mag < ov_epsilon)
          Sf_mag = ov_epsilon;

          Press_x = RPMean(-Sf_x, 0.0, pfmax((pp[ip]), 0.0), pfmax((pp[ip+1]), 0.0));
          Press_y = RPMean(-Sf_y, 0.0, pfmax((pp[ip]), 0.0),pfmax((pp[ip+sy_v]), 0.0));

          qx_v[io] = -(Sf_x / (RPowerR(fabs(Sf_mag),0.5)*mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
          qy_v[io] = -(Sf_y / (RPowerR(fabs(Sf_mag),0.5)*mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));

        }

        //fix for lower x boundary
        if (k0x < 0.0) {
              Press_x = pfmax((pp[ip]), 0.0);
              Sf_x = sx_dat[io] +(Press_x - 0.0)/dx;

              Pupox = pfmax(opp[ip],0.0);
              Sf_xo = sx_dat[io] +(Pupox - 0.0)/dx;

              double Sf_mag = RPowerR(Sf_xo*Sf_xo+Sf_yo*Sf_yo,0.5); //+ov_epsilon;
              if (Sf_mag < ov_epsilon)
              Sf_mag = ov_epsilon;
              //printf("Left: i=%d j=%d k=%d k1=%d k0x=%d k0y=%d Sf_x=%f Sf_y=%f Sf_mag=%f \n",i,j,k,k1, k0x, k0y, Sf_x, Sf_y, Sf_mag);
              if (Sf_x > 0.0) {
                qx_v[io-1] = -(Sf_x / (RPowerR(fabs(Sf_mag),0.5)*mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
                //printf("New Left q: i=%d j=%d k=%d k1=%d k0x=%d k0y=%d Sf_x=%f Sf_y=%f Sf_mag=%f press_x=%f press_y=%f pressx_old=%f pressy_old=%f qx_v=%f\n",i,j,k,k1, k0x, k0y, Sf_x, Sf_y, Sf_mag, Press_x, Press_y, Pupox, Pupoy, qx_v[io-1]);
            }
          }

          //fix for lower y boundary
          if (k0y < 0.0) {
                Press_y = pfmax((pp[ip]), 0.0);
                Sf_y = sy_dat[io] +(Press_y - 0.0)/dx;

                Pupoy = pfmax(opp[ip],0.0);
                Sf_yo = sy_dat[io] +(Pupoy - 0.0)/dx;

                double Sf_mag = RPowerR(Sf_xo*Sf_xo+Sf_yo*Sf_yo,0.5); //Note that the sf_xo was already corrected above
                if (Sf_mag < ov_epsilon)
                Sf_mag = ov_epsilon;
                //printf("Bottom: i=%d j=%d k=%d k1=%d k0x=%d k0y=%d Sf_x=%f Sf_y=%f Sf_mag=%f \n",i,j,k,k1, k0x, k0y, Sf_x, Sf_y, Sf_mag);

                if (Sf_y > 0.0) {
                  qy_v[io-sy_v] = -(Sf_y / (RPowerR(fabs(Sf_mag),0.5)*mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));
                  //printf("New Bottom q: i=%d j=%d k=%d k1=%d k0x=%d k0y=%d Sf_x=%f Sf_y=%f Sf_mag=%f press_x=%f press_y=%f pressx_old=%f pressy_old=%f qy_v=%f\n",i,j,k,k1, k0x, k0y, Sf_x, Sf_y, Sf_mag, Press_x, Press_y, Pupox, Pupoy, qy_v[io-sy_v]);
                }

                // Recalculating the x flow in the case whith both the lower and left boundaries
                // This is exactly the same as the q_x in the left boundary conditional above but
                // recalculating qx_v here again becuase the sf_mag will be adjusted with the new sf_yo above
                if(k0x < 0.0){
                  if (Sf_x > 0.0) {
                    qx_v[io-1] = -(Sf_x / (RPowerR(fabs(Sf_mag),0.5)*mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
                    printf("New LL q: i=%d j=%d Sf_x=%f Sf_y=%f Sf_mag=%f press_x=%f press_y=%f  pressx_old=%f pressy_old=%f qx_v=%f\n",i,j,Sf_x, Sf_y, Sf_mag, Press_x, Press_y, Pupox, Pupoy, qx_v[io-1]);
                  }
                }
            }
       }
    });

    BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, sg,
    {
      if (fdir[2] == 1)
      {
        io = SubvectorEltIndex(sx_sub, i, j, 0);
        ke_v[io] = qx_v[io];
        kw_v[io] = qx_v[io-1];
        kn_v[io] = qy_v[io];
        ks_v[io] = qy_v[io-sy_v];
        //printf("i=%d j=%d k=%d ke_v=%d kw_v=%d kn_v=%d ks_v=%f\n",i,j,k,ke_v[io],kw_v[io],kn_v[io],ks_v[io]);
      }
    });

  }
  else          //fcn = CALCDER calculates the derivs of KE KW KN KS wrt to current cell (i,j,k)
  {
    BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, sg,
    {
      if (fdir[2] == 1)
      {
        //printf("Made it inside CALCDER");

        io = SubvectorEltIndex(sx_sub, i, j, 0);
        itop = SubvectorEltIndex(top_sub, i, j, 0);
        /* Current node */
        k1 = (int)top_dat[itop];
        ip0 = SubvectorEltIndex(p_sub, i, j, k1);
        manning = mann_dat[io];

        /*KW  - look at nodes i-1 and i*/
        k1 = (int)top_dat[itop - 1];
        ip = SubvectorEltIndex(p_sub, (i - 1), j, k1);


        /*Calcualte Friction Slope */
        if (i > 0)
        {
          slope_mean = sx_dat[io - 1];
          slope_fx_lo = slope_mean + (((pfmax((pp[ip0]), 0.0)) - (pfmax((pp[ip]), 0.0))) / dx);
        }
        else
        {
          slope_mean = sx_dat[io];
          slope_fx_lo = slope_mean;
        }

        manning = (1.0 + 0.0000001 / fabs(slope_fx_lo)) * mann_dat[io];
        Pcen = pfmax(pp[ip0], 0.0);                 //pressure of current cell
        Pdel = pfmax(pp[ip], 0.0);                  // pressure cell to the west

        /* Caluculate Derivative */
        if (fabs(slope_fx_lo) < 0.0000001)
        {
          kw_vns[io] = 0;
          kw_v[io] = 0;
        }

        else if (slope_fx_lo > 0.0)
        {
          xdir = -1.0;
          kw_vns[io] = xdir * (-1 / (2 * dx * manning)) * RPowerR(fabs(slope_fx_lo), -0.5) * RPowerR(Pcen, (5.0 / 3.0));

          kw_v[io] = xdir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fx_lo), 0.5) * RPowerR(Pcen, (2.0 / 3.0)) +
                             1 / (2 * dx * manning) * RPowerR(fabs(slope_fx_lo), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));
        }
        else if (slope_fx_lo < 0.0)
        {
          xdir = 1.0;
          /*This is dfi-1/di-1 East*/
          kw_vns[io] = xdir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fx_lo), 0.5) * RPowerR(Pdel, (2.0 / 3.0)) -
                               1 / (2 * dx * manning) * RPowerR(fabs(slope_fx_lo), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));
          /* This is dfi-1/di East */
          kw_v[io] = xdir * ((1 / (2 * dx * manning)) * RPowerR(fabs(slope_fx_lo), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));
          //kw_v[io]=0;
        }
        //else{
        //  kw_vns[io]=0;
        // kw_v[io]=0;
        //}

        //printf("WEST: i %d j %d %4.5e %4.5e %4.5e %4.5e %4.5e %4.5e \n", i, j, Pcen, Pdel, slope_fx_lo, slope_mean, kw_v[io], kw_vns[io]);


        /* KE - look at nodes i+1 and i */
        k1 = (int)top_dat[itop + 1];
        ip2 = SubvectorEltIndex(p_sub, (i + 1), j, k1);

        /*Calcualte Friction Slope */
        if (i < gnx - 1)
        {
          slope_mean = sx_dat[io];
          slope_fx_hi = slope_mean + (((pfmax((pp[ip2]), 0.0)) - (pfmax((pp[ip0]), 0.0))) / dx);
        }
        else
        {
          slope_mean = sx_dat[io];
          slope_fx_hi = slope_mean;
        }

        manning = (1.0 + 0.0000001 / fabs(slope_fx_hi)) * mann_dat[io];
        Pcen = pfmax(pp[ip0], 0.0);                 //pressure of current cel
        Pdel = pfmax(pp[ip2], 0.0);                  // pressure cell to the east

        /* Caluculate Derivative */
        if (fabs(slope_fx_hi) < 0.0000001)
        {
          ke_vns[io] = 0;
          ke_v[io] = 0;
        }

        else if (slope_fx_hi > 0.0)
        {
          xdir = -1.0;
          /*This is dfi+1/di+1 for kw */
          ke_vns[io] = xdir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fx_hi), 0.5) * RPowerR(Pdel, (2.0 / 3.0)) +
                               1 / (2 * dx * manning) * RPowerR(fabs(slope_fx_hi), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));

          /* This is dfi+1/di for kw */
          ke_v[io] = xdir * ((-1 / (2 * dx * manning)) * RPowerR(fabs(slope_fx_hi), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));
          //  ke_v[io]=0;
        }
        else if (slope_fx_hi < 0.0)
        {
          xdir = 1.0;
          ke_vns[io] = xdir * ((1 / (2 * dx * manning)) * RPowerR(fabs(slope_fx_hi), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));

          ke_v[io] = xdir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fx_hi), 0.5) * RPowerR(Pcen, (2.0 / 3.0)) -
                             1 / (2 * dx * manning) * RPowerR(fabs(slope_fx_hi), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));
        }
        // else{
        //   ke_vns[io]=0;
        //  ke_v[io]=0;
        //}

        // printf("i %d j %d %4.5f %4.5f %4.5f %4.5f %4.5e %4.5e %4.5e %4.5e \n", i, j, slope_mean, slope_fx_lo, Pcen, Pdel, kw_v[io], kw_vns[io], ke_v[io], ke_vns[io]);
        //printf("EAST: i %d j %d %4.5e %4.5e %4.5e %4.5e %4.5e %4.5e \n", i, j,  Pcen, Pdel, slope_fx_hi, slope_mean, ke_v[io], ke_vns[io]);


        /*KS  - look at nodes j-1 and j*/
        k1 = (int)top_dat[itop - sy_v];
        ip3 = SubvectorEltIndex(p_sub, i, (j - 1), k1);

        /*Calcualte Friction Slope */
        if (j > 0)
        {
          slope_mean = sy_dat[io - sy_v];
          slope_fy_lo = slope_mean + (((pfmax((pp[ip0]), 0.0)) - (pfmax((pp[ip3]), 0.0))) / dy);
        }
        else
        {
          slope_mean = sy_dat[io];
          slope_fy_lo = slope_mean;
        }
        manning = (1.0 + 0.0000001 / fabs(slope_fy_lo)) * mann_dat[io];
        Pcen = pfmax(pp[ip0], 0.0);                 //pressure of current cel
        Pdel = pfmax(pp[ip3], 0.0);                  // pressure cell to the south

        /* Caluculate Derivative */
        if (fabs(slope_fy_lo) < 0.0000001)
        {
          ks_vns[io] = 0;
          ks_v[io] = 0;
        }
        else if (slope_fy_lo > 0.0)
        {
          ydir = -1.0;
          ks_vns[io] = ydir * ((-1 / (2 * dy * manning)) * RPowerR(fabs(slope_fy_lo), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));

          ks_v[io] = ydir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fy_lo), 0.5) * RPowerR(Pcen, (2.0 / 3.0)) +
                             1 / (2 * dy * manning) * RPowerR(fabs(slope_fy_lo), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));
        }
        else if (slope_fy_lo < 0.0)
        {
          ydir = 1.0;
          ks_vns[io] = ydir * (5 / (3 * manning) * RPowerR(fabs(slope_fy_lo), 0.5) * RPowerR(Pdel, (2.0 / 3.0)) -
                               1 / (2 * dy * manning) * RPowerR(fabs(slope_fy_lo), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));

          //ks_v[io]=0.0;
          ks_v[io] = ydir * ((1 / (2 * dy * manning)) * RPowerR(fabs(slope_fy_lo), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));
        }
        //else{
        //  ks_vns[io]=0;
        // ks_v[io]=0;
        // }

        //printf("SOUTH: i %d j %d %4.5e %4.5e %4.5e %4.5e %4.5e %4.5e \n", i, j,  Pcen, Pdel, slope_fy_lo, slope_mean,  ks_v[io], ks_vns[io]);


        /* KN - look at nodes j+1 and j */
        k1 = (int)top_dat[itop + sy_v];
        ip4 = SubvectorEltIndex(p_sub, i, (j + 1), k1);

        /*Calcualte Friction Slope */
        if (j < gny - 1)
        {
          slope_mean = sy_dat[io];
          slope_fy_hi = slope_mean + (((pfmax((pp[ip4]), 0.0)) - (pfmax((pp[ip0]), 0.0))) / dy);
        }
        else
        {
          slope_mean = sy_dat[io];
          slope_fy_hi = slope_mean;
        }

        manning = (1.0 + 0.0000001 / fabs(slope_fy_hi)) * mann_dat[io];
        Pcen = pfmax(pp[ip0], 0.0);                 //pressure of current cel
        Pdel = pfmax(pp[ip4], 0.0);                  // pressure cell to the east

        /* Caluculate Derivative */
        if (fabs(slope_fy_hi) < 0.0000001)
        {
          kn_vns[io] = 0;
          kn_v[io] = 0;
        }

        else if (slope_fy_hi > 0.0)
        {
          ydir = -1.0;
          kn_vns[io] = ydir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fy_hi), 0.5) * RPowerR(Pdel, (2.0 / 3.0)) +
                               1 / (2 * dy * manning) * RPowerR(fabs(slope_fy_hi), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));

          //kn_v[io]=0.0;

          kn_v[io] = ydir * ((-1 / (2 * dy * manning)) * RPowerR(fabs(slope_fy_hi), -0.5) * RPowerR(Pdel, (5.0 / 3.0)));
        }
        else if (slope_fy_hi < 0.0)
        {
          ydir = 1.0;
          kn_vns[io] = ydir * ((1 / (2 * dy * manning)) * RPowerR(fabs(slope_fy_hi), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));

          kn_v[io] = ydir * ((5 / (3 * manning)) * RPowerR(fabs(slope_fy_hi), 0.5) * RPowerR(Pcen, (2.0 / 3.0)) -
                             1 / (2 * dy * manning) * RPowerR(fabs(slope_fy_hi), -0.5) * RPowerR(Pcen, (5.0 / 3.0)));
        }
        //else{
        //   kn_vns[io]=0;
        //   kn_v[io]=0;
        //}
        //printf("NORTH: i %d j %d %4.5e %4.5e %4.5e %4.5e %4.5e %4.5e \n", i, j,  Pcen, Pdel, slope_fy_hi, slope_mean,  kn_v[io], kn_vns[io]);
      }
    });
    //}
  }

}

//*/
/*--------------------------------------------------------------------------
 * OverlandFlowEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalDiffInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * OverlandFlowEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  OverlandFlowEvalDiffFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalDiffNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * OverlandFlowEvalFreePublicXtra
 *-------------------------------------------------------------------------*/

void  OverlandFlowEvalDiffFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  OverlandFlowEvalDiffSizeOfTempData()
{
  return 0;
}
