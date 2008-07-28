/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _N_VECTOR_HEADER
#define _N_VECTOR_HEADER

#include "parflow.h"



#define N_VFree(x)                    FreeVector(x)

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

