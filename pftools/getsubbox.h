
#ifndef GETSUBBOX_HEADER
#define GETSUBBOX_HEADER

#ifdef __cplusplus
extern "C" {
#endif

#include "databox.h"

#include <stdio.h>
#include <math.h>


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

Databox * CompSubBox(Databox *fun,
                     int il, int jl, int kl,
                     int iu, int ju, int ku);

#ifdef __cplusplus
}
#endif

#endif

