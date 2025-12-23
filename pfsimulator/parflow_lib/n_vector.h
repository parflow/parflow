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

#ifndef _N_VECTOR_HEADER
#define _N_VECTOR_HEADER

#include "parflow.h"

#define N_VLinearSum(a, x, b, y, z)   PFVLinearSum(a, x, b, y, z)
#define N_VConst(c, z)                PFVConstInit(c, z)
#define N_VProd(x, y, z)              PFVProd(x, y, z)
#define N_VDiv(x, y, z)               PFVDiv(x, y, z)
#define N_VScale(c, x, z)             PFVScale(c, x, z)
#define N_VAbs(x, z)                  PFVAbs(x, z)
#define N_VInv(x, z)                  PFVInv(x, z)
#define N_VAddConst(x, b, z)          PFVAddConst(x, b, z)

#define N_VDotProd(x, y)              PFVDotProd(x, y)
#define N_VMaxNorm(x)                 PFVMaxNorm(x)
#define N_VWrmsNorm(x, w)             PFVWrmsNorm(x, w)
#define N_VWL2Norm(x, w)              PFVWL2Norm(x, w)
#define N_VL1Norm(x)                  PFVL1Norm(x)
#define N_VMin(x)                     PFVMin(x)
#define N_VConstrProdPos(c, x)        PFVConstrProdPos(c, x)

#define N_VCompare(c, x, z)           PFVCompare(c, x, z)
#define N_VInvTest(x, z)              PFVInvTest(x, z)

#endif

