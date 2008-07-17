/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"

N_Vector N_VNew_Parflow(grid)
Grid   *grid;
{
  N_Vector v;
  N_VectorContent_Parflow content;
  int i;
  int nspecies = NUM_SPECIES;


  v = N_VNewEmpty_Parflow(nspecies);

  content = NV_CONTENT_PF(v);
  

  for (i = 0; i < nspecies; i++)
  {
    content->specie[i] = NewVector(grid,1,1);
  }
  content -> nvector_allocated_pfvectors = TRUE;

  return(v);  

}

N_Vector N_VNewEmpty_Parflow(nspecies)
int nspecies;
{
  N_Vector v;
  N_Vector_Ops ops;
  N_VectorContent_Parflow content;
  int i;

  v = (N_Vector) malloc(sizeof *v);
  ops = (N_Vector_Ops) malloc(sizeof(struct _generic_N_Vector_Ops));
  
  ops->nvclone           = N_VClone_Parflow;
  ops->nvcloneempty      = N_VCloneEmpty_Parflow; 
  ops->nvdestroy         = N_VDestroy_Parflow;
  ops->nvspace           = N_VSpace_Parflow; 
  ops->nvgetarraypointer = NULL;
  ops->nvsetarraypointer = NULL;
  ops->nvlinearsum       = N_VLinearSum_Parflow;
  ops->nvconst           = N_VConst_Parflow;
  ops->nvprod            = N_VProd_Parflow;
  ops->nvdiv             = N_VDiv_Parflow; 
  ops->nvscale           = N_VScale_Parflow;
  ops->nvabs             = N_VAbs_Parflow; 
  ops->nvinv             = N_VInv_Parflow;
  ops->nvaddconst        = NULL;
  ops->nvdotprod         = N_VDotProd_Parflow; 
  ops->nvmaxnorm         = N_VMaxNorm_Parflow;
  ops->nvwrmsnormmask    = NULL;
  ops->nvwrmsnorm        = NULL;
  ops->nvmin             = N_VMin_Parflow;
  ops->nvwl2norm         = N_VWL2Norm_Parflow; 
  ops->nvl1norm          = N_VL1Norm_Parflow; 
  ops->nvcompare         = NULL;
  ops->nvinvtest         = NULL;
  ops->nvconstrmask      = N_VConstrMask_Parflow;
  ops->nvminquotient     = N_VMinQuotient_Parflow;

  content = (N_VectorContent_Parflow) malloc(sizeof(struct _N_VectorContent_Parflow));
  content->specie = (Vector **) malloc(nspecies * sizeof(Vector *)); 
  content->nvector_allocated_pfvectors = FALSE;
  
  content->num_species = nspecies;

  for (i = 0; i < nspecies; i++)
  {
    content->specie[i] = NULL;
  }

  v->content = content;
  v->ops = ops;
 
  return(v);
}

void N_VPrint(x)
N_Vector x;
{
  Grid       *grid; 
  N_VectorContent_Parflow content;
  Vector     *v;
  Subgrid    *subgrid;
 
  Subvector  *v_sub;

  double     *vp;

  int         ix,   iy,   iz;
  int         nx,   ny,   nz;
  int         nx_x, ny_x, nz_x;
  
  int         sg, i, j, k, i_x;

  int         nspecies;

  content = NV_CONTENT_PF(x);

  grid = VectorGrid(content->specie[0]);

  nspecies = content->num_species;

  v = content->specie[0];

  ForSubgridI(sg, GridSubgrids(grid))
  {
     subgrid = GridSubgrid(grid, sg);

     v_sub = VectorSubvector(v, sg);

     ix = SubgridIX(subgrid);
     iy = SubgridIY(subgrid);
     iz = SubgridIZ(subgrid);

     nx = SubgridNX(subgrid);
     ny = SubgridNY(subgrid);
     nz = SubgridNZ(subgrid);

     nx_x = SubvectorNX(v_sub);
     ny_x = SubvectorNY(v_sub);
     nz_x = SubvectorNZ(v_sub);

     vp = SubvectorElt(v_sub, ix, iy, iz);

     i_x = 0;
     BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
	       i_x, nx_x, ny_x, nz_x, 1, 1, 1,
	       {
		  //printf("%g\n", vp[i_x]);
                  fflush(NULL);
	       });
  }
  fflush(NULL);
}


/* N_VCloneEmpty_Parflow creates a new N_Vector of the same type as an 
   existing w vector; it does not allocate storage for the data array*/

N_Vector N_VCloneEmpty_Parflow(w)
N_Vector w;
{
  N_Vector clone;
  N_Vector_Ops ops;
  N_VectorContent_Parflow clone_content,wcontent;
  int mode = NumUpdateModes;
  int sg;
  int i,nspecies;
  
  wcontent = NV_CONTENT_PF(w);

  /* Extract the content of the incoming N_Vector */
  nspecies = wcontent->num_species;

  clone = (N_Vector) malloc(sizeof *w);
  
  ops = (N_Vector_Ops) malloc (sizeof(struct _generic_N_Vector_Ops));

  ops->nvclone           = w->ops->nvclone;
  ops->nvcloneempty      = w->ops->nvcloneempty;      
  ops->nvdestroy         = w->ops->nvdestroy;         
  ops->nvspace           = w->ops->nvspace;           
  ops->nvgetarraypointer = w->ops->nvgetarraypointer;
  ops->nvsetarraypointer = w->ops->nvsetarraypointer;
  ops->nvlinearsum       = w->ops->nvlinearsum;
  ops->nvconst           = w->ops->nvconst;
  ops->nvprod            = w->ops->nvprod;
  ops->nvdiv             = w->ops->nvdiv;
  ops->nvscale           = w->ops->nvscale;
  ops->nvabs             = w->ops->nvabs; 
  ops->nvinv             = w->ops->nvinv;
  ops->nvaddconst        = w->ops->nvaddconst;
  ops->nvdotprod         = w->ops->nvdotprod;
  ops->nvmaxnorm         = w->ops->nvmaxnorm;
  ops->nvwrmsnormmask    = w->ops->nvwrmsnormmask;
  ops->nvwrmsnorm        = w->ops->nvwrmsnorm;
  ops->nvmin             = w->ops->nvmin;
  ops->nvwl2norm         = w->ops->nvwl2norm;
  ops->nvl1norm          = w->ops->nvl1norm;
  ops->nvcompare         = w->ops->nvcompare;
  ops->nvinvtest         = w->ops->nvinvtest;
  ops->nvconstrmask      = w->ops->nvconstrmask;
  ops->nvminquotient     = w->ops->nvminquotient;
 
  clone_content = (N_VectorContent_Parflow) malloc(sizeof(struct _N_VectorContent_Parflow)); 
  clone_content->specie = NULL;
  clone_content -> nvector_allocated_pfvectors = FALSE;

  clone_content->num_species = nspecies;

  clone->content = clone_content;
  clone->ops = ops;

  return(clone);

}

/* N_VClone_Parflow returns a new Vector of the same form as the
   input N_Vector; it does not copy the vector but allocates the storage*/

N_Vector N_VClone_Parflow(w)
N_Vector w;
{
  N_Vector clone;
  Grid *grid;
  N_VectorContent_Parflow clone_content,w_content;
  int i,nspecies;

  w_content = NV_CONTENT_PF(w);
  grid = VectorGrid(w_content->specie[0]);

  clone = N_VCloneEmpty_Parflow(w);
  clone_content = NV_CONTENT_PF(clone);
  nspecies = clone_content->num_species;

  clone_content->specie = (Vector **) malloc(nspecies * sizeof(Vector *)); 

  for (i = 0; i < nspecies; i++)
  {
    clone_content->specie[i] = NewVector(grid,1,1);
  }
  clone_content -> nvector_allocated_pfvectors = TRUE;
  
  //clone->content = clone_content;

  return(clone);

}

/* Destroy storage of N_Vector */
void N_VDestroy_Parflow(v)
N_Vector v;
{
  N_VectorContent_Parflow content;
  Vector *specie;
  int i, nspecies;
  
  content = NV_CONTENT_PF(v);
  
  if (content -> nvector_allocated_pfvectors) 
  {
     nspecies = content->num_species;
     for (i = 0; i < nspecies; i++) 
     {
	specie = content->specie[i];
	FreeVector(specie);
     }
  }
  free(content->specie);
 
  free(content);

  free(v -> ops);
  free(v);
}
 
/* Returns space requirements for one N_Vector */
void N_VSpace_Parflow(v, lrw, liw)
N_Vector v;
long int *lrw, *liw;
{
  N_VectorContent_Parflow content;
  Vector *specie;
  int nprocs;
  int i, nspecies;
  
  content = NV_CONTENT_PF(v);
  *lrw = 0;

  nspecies = content->num_species;
  for (i = 0; i < nspecies; i++) 
  {
    specie = content->specie[i];
    *lrw +=SizeOfVector(specie);
  }  

  //MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  *liw = 0;
}

/* Perform operation z = ax + by */
void N_VLinearSum_Parflow(a, x, b, y, z)
double a, b;
N_Vector x, y, z;
{
  Vector *xx, *yy, *zz;
  N_VectorContent_Parflow xcont, ycont, zcont;
  int i, nspecies;

  xcont = NV_CONTENT_PF(x);  
  ycont = NV_CONTENT_PF(y);
  zcont = NV_CONTENT_PF(z);

  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++) 
  {
    xx = xcont->specie[i];
    yy = ycont->specie[i];
    zz = zcont->specie[i];
    PFVLinearSum(a, xx, b, yy, zz);
  }

}

/* Sets all components of z to c */
void N_VConst_Parflow(c, z)
double c;
N_Vector z;
{
  N_VectorContent_Parflow content;
  Vector *zz;
  int i, nspecies;

  content= NV_CONTENT_PF(z);
  
  nspecies = content->num_species;
  for (i = 0; i < nspecies; i++)
  {
    zz = content->specie[i];
    PFVConstInit(c, zz);
  }

}

/* Sets z to the component-wise product of inputs x and y */
void N_VProd_Parflow(x, y, z)
N_Vector x, y, z;
{
  N_VectorContent_Parflow xcont, ycont, zcont;
  Vector *xx, *yy, *zz;
  int i,nspecies;
  
  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
  zcont = NV_CONTENT_PF(z); 

  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
     xx = xcont->specie[i];
     yy = ycont->specie[i];
     zz = zcont->specie[i];
     PFVProd(xx, yy, zz);
  }  
    
}

/* Sets z to be the component-wise ratio of the inputs x and y */
void N_VDiv_Parflow(x, y, z)
N_Vector x, y, z;
{
  N_VectorContent_Parflow xcont, ycont, zcont;
  Vector *xx, *yy, *zz;
  int i, nspecies;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
  zcont = NV_CONTENT_PF(z);
 
  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    yy = ycont->specie[i];
    zz = zcont->specie[i]; 
    PFVDiv(xx, yy, zz);
  }

}

/* Scales x by the scalar c and returns the result in z */
void N_VScale_Parflow(c, x, z)
double c;
N_Vector x, z;
{
  N_VectorContent_Parflow xcont, zcont;
  Vector *xx, *zz;
  int i, nspecies;

  xcont = NV_CONTENT_PF(x);
  zcont = NV_CONTENT_PF(z);

  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    zz = zcont->specie[i];
    PFVScale(c, xx, zz);
  } 
    
}

/* Sets the components of y to be the absolute values of the components of x */
void N_VAbs_Parflow(x, y)
N_Vector x, y;
{
  N_VectorContent_Parflow xcont, ycont;
  Vector *xx, *yy;
  int i, nspecies;
 
  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
 
  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    yy = ycont->specie[i];
    PFVAbs(xx, yy);
  }
  
}

/* Sets the components of y to be the inverse of the components of x;
   does not check for div by zero */
void N_VInv_Parflow(x, y)
N_Vector x, y;
{
  N_VectorContent_Parflow xcont, ycont; 
  Vector *xx, *yy; 
  int i, nspecies;
 
  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
 
  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    yy = ycont->specie[i];
    PFVInvTest(xx, yy); 
  }
    
}

/* Returns the value of the ordinary dot product of x and y */
double N_VDotProd_Parflow(x, y)
N_Vector x, y;
{
  N_VectorContent_Parflow xcont, ycont;
  Vector *xx, *yy;
  double dprod;
  int i, nspecies;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);

  dprod = 0.0;

  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    yy = ycont->specie[i];
    dprod += PFVDotProd(xx, yy);
  }

  return(dprod);
}

/* Returns the maximum norm of x */
double N_VMaxNorm_Parflow(x)
N_Vector x;
{
  N_VectorContent_Parflow xcont;
  Vector *xx;
  double maxnormpress, maxnormtemp, maxnormspecie;
  double maxnorm = 0.0;
  int i,nspecies;

  xcont = NV_CONTENT_PF(x);

  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    maxnormspecie = PFVMaxNorm(xx);
    if (maxnormspecie > maxnorm) maxnorm = maxnormspecie;
  }

  return(maxnorm);
}

/* Returns the smallest element of x */
double N_VMin_Parflow(x)
N_Vector x;
{
  N_VectorContent_Parflow xcont; 
  Vector *xx;
  double minspecie;
  double min = BIG_REAL;
  int i,nspecies;
     
  xcont = NV_CONTENT_PF(x); 
 
  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    minspecie= PFVMin(xx);
    if (minspecie < min) min = minspecie;
  }

  return(min);
}

/* Returns the weighted Euclidean nrom of x with weights w */
double N_VWL2Norm_Parflow(x, w)
N_Vector x, w;
{
  N_VectorContent_Parflow xcont, wcont;
  Vector *xx, *ww;
  double sum = 0.0;
  double l2normm;
  int i,nspecies;

  xcont = NV_CONTENT_PF(x);
  wcont = NV_CONTENT_PF(w);

  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    ww = wcont->specie[i];
    /* The modified PFVWL2Norm function returns only the sum, NOT sqrt(sum) */
    sum += PFVWL2Norm(xx, ww);
  }
    
  l2normm = sqrt(sum);

  return(l2normm);
}  

/* Returns the L1 norm of x */
double N_VL1Norm_Parflow(x)
N_Vector x;
{
  N_VectorContent_Parflow xcont;
  Vector *xx;
  double sum = 0.0;
  int i, nspecies; 

  xcont = NV_CONTENT_PF(x);
 
  nspecies = xcont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    xx = xcont->specie[i];
    sum += PFVL1Norm(xx);
  }
 
  return(sum);
}

/* Performs a constraint test (see manual) */
booleantype N_VConstrMask_Parflow(c, x, m)
N_Vector c, x, m;
{
  N_VectorContent_Parflow ccont, xcont, mcont;
  Vector *cc, *xx, *mm;
  int testspecie;
  int  test = 0;
  int i, nspecies;

  ccont = NV_CONTENT_PF(c);
  xcont = NV_CONTENT_PF(x);
  mcont = NV_CONTENT_PF(m);

  nspecies = ccont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    cc = ccont->specie[i];
    xx = xcont->specie[i];
    mm = mcont->specie[i];
    testspecie = PFVConstrMask(cc, xx, mm);
    if (testspecie < test) test  = testspecie;
  }

  if (test == 0)
    return(FALSE);
  else
    return(TRUE);
}

/* This routine returns the minimum of the quotients obtained 
   by termswise dividing num by denom; zero elements in denom
   will be skipped; if no quotients are found, thenthe large 
   value BIG_REAL in sundialstypes.h is returned */
double N_VMinQuotient_Parflow(num, denom)
N_Vector num, denom;
{
  N_VectorContent_Parflow ncont, dcont;
  Vector *n, *d;
  double minspecie;
  double min = BIG_REAL;
  int i, nspecies;

  ncont = NV_CONTENT_PF(num);
  dcont = NV_CONTENT_PF(denom);
 
  nspecies = ncont->num_species;
  for (i = 0; i < nspecies; i++)
  {
    n = ncont->specie[i];
    d = dcont->specie[i];
    minspecie = PFVMinQuotient(n, d);
    if (minspecie < min) min = minspecie;
  }
    
  return(min);
}

/*void N_VVector_Parflow(template, pressure, temperature)
N_Vector template;
Vector *pressure;
Vector *temperature;
{
  N_VectorContent_Parflow content = NV_CONTENT_PF(template);
  
  content->specie[0] = pressure;
//  content->specie[1] = temperature;

// Below is rubbish
  PFVVector(content->pressure,pressure);
  PFVVector(content->temperature,temperature);
  
  PFVVector(content->specie[1],pressure);
  PFVVector(content->specie[2],temperature);

}*/

/* ---------------------------------------------------------------- 
 * Function to create a parallel N_Vector with user data component 
 */ 
                               
N_Vector N_VMake_Parflow(pressure,temperature)
Vector *pressure;
Vector *temperature;
{  
  N_Vector v; 
  Grid *grid = VectorGrid(pressure);
  N_VectorContent_Parflow content;
  int nspecies = NUM_SPECIES;
 
  v = NULL; 
  v = N_VNewEmpty_Parflow(nspecies);

  content = NV_CONTENT_PF(v);

  if (v == NULL) {
     printf("COULD NOT MAKE MULTISPECIES N_VECTOR\n");
     return(NULL); 
  }
   
  if (v != NULL) { 
    /* Attach content*/
    content->specie[0] = pressure;
    content->specie[1] = temperature;
  } 
 
  return(v);
} 

