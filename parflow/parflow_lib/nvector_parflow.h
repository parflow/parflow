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

#endif

struct _N_VectorContent_Parflow {
  Vector **specie;
  int num_species;
  int nvector_allocated_pfvectors;
};

typedef struct _N_VectorContent_Parflow *N_VectorContent_Parflow;

#define NV_CONTENT_PF(v) ( (N_VectorContent_Parflow)(v->content) )

#define NUM_SPECIES      2
