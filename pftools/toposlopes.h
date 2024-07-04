/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
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
#ifndef TOPOSLOPES
#define TOPOSLOPES

#include "databox.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

void ComputeSlopeXUpwind(
                         Databox *dem,
                         double   dx,
                         Databox *sx);

void ComputeSlopeYUpwind(
                         Databox *dem,
                         double   dy,
                         Databox *sy);

int ComputeTestParent(
                      int      i,
                      int      j,
                      int      ii,
                      int      jj,
                      Databox *dem,
                      Databox *sx,
                      Databox *sy);

void ComputeParentMap(
                      int      i,
                      int      j,
                      Databox *dem,
                      Databox *sx,
                      Databox *sy,
                      Databox *parentmap);

void ComputeUpstreamArea(
                         Databox *dem,
                         Databox *sx,
                         Databox *sy,
                         Databox *area);

void ComputeFillFlats(
                      Databox *dem,
                      Databox *newdem);

int ComputePitFill(
                   Databox *dem,
                   double   dpit);

int ComputeMovingAvg(
                     Databox *dem,
                     double   wsize);

void ComputeSlopeXD4(
                     Databox *dem,
                     Databox *sx);

void ComputeSlopeYD4(
                     Databox *dem,
                     Databox *sy);

void ComputeSlopeD8(
                    Databox *dem,
                    Databox *slope);

void ComputeSegmentD8(
                      Databox *dem,
                      Databox *ds);

void ComputeChildD8(
                    Databox *dem,
                    Databox *child);

void ComputeFlintsLawRec(
                         int      i,
                         int      j,
                         Databox *dem,
                         Databox *demflint,
                         Databox *child,
                         Databox *area,
                         Databox *ds,
                         double   c,
                         double   p);

void ComputeFlintsLaw(
                      Databox *dem,
                      double   c,
                      double   p,
                      Databox *flintdem);

void ComputeFlintsLawFit(
                         Databox *dem,
                         double   c,
                         double   p,
                         int      maxiter,
                         Databox *demflint);

void ComputeFlintsLawByBasin(
                             Databox *dem,
                             double   c,
                             double   p,
                             int      maxiter,
                             Databox *demflint);

void ComputeFlintLM(
                    Databox *dem,
                    Databox *demflint,
                    Databox *area,
                    Databox *child,
                    Databox *ds,
                    double   c,
                    double   p,
                    Databox *dzdc,
                    Databox *dzdp);

double ComputeLMCoeff(
                      Databox *demflint,
                      Databox *dem,
                      Databox *area,
                      Databox *child,
                      Databox *ds,
                      double   c,
                      double   p,
                      double   alpha[][2],
                      double   beta[],
                      double   chisq);

void ComputeGaussJordan(
                        double a[][2],
                        int    n,
                        double b[][2],
                        int    m);

void ComputeSatTransmissivity(
                              int      nlayers,
                              Databox *mask,
                              Databox *perm,
                              Databox *trans);

void ComputeSatStorage(
                       Databox *mask,
                       Databox *porosity,
                       Databox *ssat,
                       Databox *sres,
                       Databox *stor);

void ComputeTopoRecharge(
                         int      river[][2],
                         int      nriver,
                         Databox *trans,
                         Databox *dem,
                         Databox *sx,
                         Databox *sy,
                         Databox *recharge);

void ComputeEffectiveRecharge(
                              Databox *precip,
                              Databox *et,
                              Databox *runoff,
                              Databox *sx,
                              Databox *sy,
                              Databox *dem,
                              Databox *recharge);

void ComputeTopoIndex(
                      Databox *dem,
                      Databox *sx,
                      Databox *sy,
                      Databox *topoindex);

void ComputeTopoDeficit(
                        int      profile,
                        double   m,
                        Databox *trans,
                        Databox *dem,
                        Databox *sx,
                        Databox *sy,
                        Databox *recharge,
                        Databox *ssat,
                        Databox *sres,
                        Databox *porosity,
                        Databox *mask,
                        Databox *deficit);

void ComputeTopoDeficitToWT(
                            Databox *deficit,
                            Databox *porosity,
                            Databox *ssat,
                            Databox *sres,
                            Databox *mask,
                            Databox *top,
                            Databox *wtdepth);

void ComputeHydroStatFromWT(
                            Databox *wtdepth,
                            Databox *top,
                            Databox *mask,
                            Databox *press0);

#ifdef __cplusplus
}
#endif

#endif

