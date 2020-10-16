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

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
    int type;     /* input type */
    void  *data;  /* pointer to Type structure */

    NameArray regions;
} PublicXtra;

typedef struct {
    Grid    *grid;

    double  *temp_data;
} InstanceXtra;

typedef struct {
  NameArray regions;
  int num_regions;
  int     *region_indices;
  double  *values;
} Type0;                       /* constant regions */

typedef struct {
    int num_regions;
    int    *region_indices;
    int data_from_file;
    char   *alpha_file;
    char   *n_file;
    char   *s_sat_file;
    char   *s_res_file;
    double *alphas;
    double *ns;
    double *s_ress;
    double *s_difs;
    Vector *alpha_values;
    Vector *n_values;
    Vector *s_res_values;
    Vector *s_sat_values;
} Type1;                      /* Van Genuchten Saturation Curve */

typedef struct {
    int num_regions;
    int    *region_indices;
    double *alphas;
    double *betas;
    double *s_ress;
    double *s_difs;
} Type2;                      /* Haverkamp et.al. Saturation Curve */

typedef struct {
    int num_regions;
    int    *region_indices;
} Type3;                      /* Data points for Saturation Curve */

typedef struct {
    int num_regions;
    int     *region_indices;
    int     *degrees;
    double **coefficients;
} Type4;                      /* Polynomial function for Saturation Curve */

typedef struct {
    char    *filename;

    Vector  *satRF;
} Type5;                      /* Spatially varying field over entire domain
                               * read from a file */

/*--------------------------------------------------------------------------
 * Saturation van genuchten write values into problemdata
 *--------------------------------------------------------------------------*/
void     vanGenuchten(
    Vector *     pd_alpha,
    Vector *     pd_n,
    Vector *     pd_sres,
    Vector *     pd_ssat,

    ProblemData *problem_data /* Contains geometry info. for the problem */
)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra->type != 1)  //BB only calculate van genuchten parameters if using van genuchten.
  {
    return;
  }

//  Type0         *dummy0;
  Type1         *dummy1;
//  Type2         *dummy2;
//  Type3         *dummy3;
//  Type4         *dummy4;
//  Type5         *dummy5;

//  Grid          *grid = VectorGrid(phase_saturation);
  Grid          *grid = VectorGrid(pd_alpha);  //BB

  GrGeomSolid   *gr_solid;

  Subvector     *n_values_sub;
  Subvector     *alpha_values_sub;
  Subvector     *s_res_values_sub;
  Subvector     *s_sat_values_sub;

  double        *n_values_dat, *alpha_values_dat;
  double        *s_res_values_dat, *s_sat_values_dat;

  SubgridArray  *subgrids = GridSubgrids(grid);

  Subgrid       *subgrid;

  int sg;

  int ix, iy, iz, r;
  int nx, ny, nz;

  int i, j, k, ind;

  int n_index, alpha_index, s_res_index, s_sat_index;

  int            *region_indices, num_regions, ir;

  Subvector      *pd_alpha_sub;         //BB
  Subvector      *pd_n_sub;         //BB
  Subvector      *pd_sres_sub;         //BB
  Subvector      *pd_ssat_sub;         //BB
  double *pd_alpha_dat, *pd_n_dat, *pd_sres_dat, *pd_ssat_dat;    //BB


  /* Initialize saturations */
  InitVector(pd_alpha, 0.0);  //BB
  InitVector(pd_n, 0.0);      //BB
  InitVector(pd_sres, 0.0);   //BB
  InitVector(pd_ssat, 0.0);   //BB

// SGS FIXME why is this needed?
#undef max
//  InitVectorAll(phase_saturation, -FLT_MAX);

  int data_from_file;
  double *alphas, *ns, *s_ress, *s_difs;
  double alpha, n, s_res, s_sat;

  Vector *n_values, *alpha_values, *s_res_values, *s_sat_values;

  dummy1 = (Type1*)(public_xtra->data);

  num_regions = (dummy1->num_regions);
  region_indices = (dummy1->region_indices);
  alphas = (dummy1->alphas);
  ns = (dummy1->ns);
  s_ress = (dummy1->s_ress);
  s_difs = (dummy1->s_difs);
//  s_sats = (dummy1->s_sats);  //BB does not exist
  data_from_file = (dummy1->data_from_file);

  if (data_from_file == 0) /* Soil parameters given by region */
  {
    for (ir = 0; ir < num_regions; ir++)
    {
      gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

      ForSubgridI(sg, subgrids)
      {
        subgrid = SubgridArraySubgrid(subgrids, sg);
//        ps_sub = VectorSubvector(phase_saturation, sg);
//        pp_sub = VectorSubvector(phase_pressure, sg);
//        pd_sub = VectorSubvector(phase_density, sg);

        pd_alpha_sub = VectorSubvector(pd_alpha, sg);   //BB
        pd_n_sub = VectorSubvector(pd_n, sg);           //BB
        pd_sres_sub = VectorSubvector(pd_sres, sg);     //BB
        pd_ssat_sub = VectorSubvector(pd_ssat, sg);     //BB

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        r = SubgridRX(subgrid);

//        psdat = SubvectorData(ps_sub);
//        ppdat = SubvectorData(pp_sub);
//        pddat = SubvectorData(pd_sub);

        pd_alpha_dat = SubvectorData(pd_alpha_sub);   //BB
        pd_n_dat = SubvectorData(pd_n_sub);           //BB
        pd_sres_dat = SubvectorData(pd_sres_sub);     //BB
        pd_ssat_dat = SubvectorData(pd_ssat_sub);     //BB

        GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,

//                       ips = SubvectorEltIndex(ps_sub, i, j, k);
//                       ipp = SubvectorEltIndex(pp_sub, i, j, k);
//                       ipd = SubvectorEltIndex(pd_sub, i, j, k);
                     {ind = SubvectorEltIndex(pd_alpha_sub, i, j, k);


                       alpha = alphas[ir];
                       n = ns[ir];
                       s_res = s_ress[ir];
                       s_sat = s_difs[ir] + s_ress[ir];  //new
                       //s_dif = s_difs[ir]; // BB not needed

                       pd_alpha_dat[ind] = alpha;  //BB
                       pd_n_dat[ind] = n;  //BB
                       pd_sres_dat[ind] = s_res;  //BB  // no ssat???
                       pd_ssat_dat[ind] = s_sat;
                     });
      }     /* End subgrid loop */
    }       /* End loop over regions */
  }         /* End if data not from file */
  else
  {
    gr_solid = ProblemDataGrDomain(problem_data);
    n_values = dummy1->n_values;
    alpha_values = dummy1->alpha_values;
    s_res_values = dummy1->s_res_values;
    s_sat_values = dummy1->s_sat_values;

    ForSubgridI(sg, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, sg);
//      ps_sub = VectorSubvector(phase_saturation, sg);
//      pp_sub = VectorSubvector(phase_pressure, sg);
//      pd_sub = VectorSubvector(phase_density, sg);

      n_values_sub = VectorSubvector(n_values, sg);
      alpha_values_sub = VectorSubvector(alpha_values, sg);
      s_res_values_sub = VectorSubvector(s_res_values, sg);
      s_sat_values_sub = VectorSubvector(s_sat_values, sg);

      pd_alpha_sub = VectorSubvector(pd_alpha, sg);   //BB
      pd_n_sub = VectorSubvector(pd_n, sg);           //BB
      pd_sres_sub = VectorSubvector(pd_sres, sg);     //BB
      pd_ssat_sub = VectorSubvector(pd_ssat, sg);     //BB

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      r = SubgridRX(subgrid);

//      psdat = SubvectorData(ps_sub);
//      ppdat = SubvectorData(pp_sub);
//      pddat = SubvectorData(pd_sub);

      n_values_dat = SubvectorData(n_values_sub);
      alpha_values_dat = SubvectorData(alpha_values_sub);
      s_res_values_dat = SubvectorData(s_res_values_sub);
      s_sat_values_dat = SubvectorData(s_sat_values_sub);

      pd_alpha_dat = SubvectorData(pd_alpha_sub);   //BB
      pd_n_dat = SubvectorData(pd_n_sub);           //BB
      pd_sres_dat = SubvectorData(pd_sres_sub);     //BB
      pd_ssat_dat = SubvectorData(pd_ssat_sub);     //BB

      GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
                   {
//                     ips = SubvectorEltIndex(ps_sub, i, j, k);
//                   ipp = SubvectorEltIndex(pp_sub, i, j, k);
//                   ipd = SubvectorEltIndex(pd_sub, i, j, k);
                     ind = SubvectorEltIndex(pd_alpha_sub, i, j, k);

                     n_index = SubvectorEltIndex(n_values_sub, i, j, k);
                     alpha_index = SubvectorEltIndex(alpha_values_sub, i, j, k);
                     s_res_index = SubvectorEltIndex(s_res_values_sub, i, j, k);
                     s_sat_index = SubvectorEltIndex(s_sat_values_sub, i, j, k);

                     alpha = alpha_values_dat[alpha_index];
                     n = n_values_dat[n_index];
                     s_res = s_res_values_dat[s_res_index];
                     s_sat = s_sat_values_dat[s_sat_index];

                     pd_alpha_dat[ind] = alpha;  //BB
                     pd_n_dat[ind] = n;  //BB
                     pd_sres_dat[ind] = s_res;  //BB
                     pd_ssat_dat[ind] = s_sat;  //BB
                   });
    }       /* End subgrid loop */
  }         /* End if data_from_file */
}


/*--------------------------------------------------------------------------
 * VanGenuchtenInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *vanGenuchtenInitInstanceXtra(
    Grid *  grid,
    double *temp_data
)  {
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);
  InstanceXtra  *instance_xtra;

  Type1         *dummy1;
  Type5         *dummy5;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      if (public_xtra->type == 1)
      {
        dummy1 = (Type1*)(public_xtra->data);
        if ((dummy1->data_from_file) == 1)
        {
          FreeVector(dummy1->n_values);
          FreeVector(dummy1->alpha_values);
          FreeVector(dummy1->s_res_values);
          FreeVector(dummy1->s_sat_values);
        }
      }
      if (public_xtra->type == 5)
      {
        dummy5 = (Type5*)(public_xtra->data);
        FreeVector(dummy5->satRF);
      }
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    /* Uses a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);
      if ((dummy1->data_from_file) == 1)
      {
        dummy1->n_values = NewVectorType(grid, 1, 1, vector_cell_centered);
        dummy1->alpha_values = NewVectorType(grid, 1, 1, vector_cell_centered);
        dummy1->s_res_values = NewVectorType(grid, 1, 1, vector_cell_centered);
        dummy1->s_sat_values = NewVectorType(grid, 1, 1, vector_cell_centered);
      }
    }
    if (public_xtra->type == 5)
    {
      dummy5 = (Type5*)(public_xtra->data);
      (dummy5->satRF) = NewVectorType(grid, 1, 1, vector_cell_centered);
    }
  }


  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;

    /* Uses a spatially varying field */
    if (public_xtra->type == 1)
    {
      dummy1 = (Type1*)(public_xtra->data);
      if ((dummy1->data_from_file) == 1)
      {
        ReadPFBinary((dummy1->alpha_file),
                     (dummy1->alpha_values));
        ReadPFBinary((dummy1->n_file),
                     (dummy1->n_values));
        ReadPFBinary((dummy1->s_res_file),
                     (dummy1->s_res_values));
        ReadPFBinary((dummy1->s_sat_file),
                     (dummy1->s_sat_values));
      }
    }
    if (public_xtra->type == 5)
    {
      dummy5 = (Type5*)(public_xtra->data);

      ReadPFBinary((dummy5->filename),
                   (dummy5->satRF));
    }
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;

  return this_module;
}

/*-------------------------------------------------------------------------
 * VanGenuchtenFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  vanGenuchtenFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * VanGenuchtenNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *vanGenuchtenNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  //Type0         *dummy0;
  Type1         *dummy1;
  //Type2         *dummy2;
  //Type3         *dummy3;
  //Type4         *dummy4;
  //Type5         *dummy5;

  int num_regions, ir;

  char *switch_name;
  char *region;

  char key[IDB_MAX_KEY_LEN];

  NameArray type_na;

  type_na = NA_NewNameArray("Constant VanGenuchten Haverkamp Data Polynomial PFBFile");

  public_xtra = ctalloc(PublicXtra, 1);

  switch_name = GetStringDefault("Phase.Saturation.Type", "Constant");  //BB Testing single_default error
  public_xtra->type = NA_NameToIndex(type_na, switch_name);

  double s_sat;

  dummy1 = ctalloc(Type1, 1);

  if ((public_xtra->type) == 1 || (public_xtra->type) == 5) {  // BB only do this if van_genuchten is used. What do we need PFBFile for??
    switch_name = GetString("Phase.Saturation.GeomNames");
    public_xtra->regions = NA_NewNameArray(switch_name);

    num_regions = NA_Sizeof(public_xtra->regions);

    sprintf(key, "Phase.Saturation.VanGenuchten.File");
    dummy1->data_from_file = GetIntDefault(key, 0);

    if ((dummy1->data_from_file) == 0) {
      dummy1->num_regions = num_regions;

      (dummy1->region_indices) = ctalloc(int, num_regions);
      (dummy1->alphas) = ctalloc(double, num_regions);
      (dummy1->ns) = ctalloc(double, num_regions);
      (dummy1->s_ress) = ctalloc(double, num_regions);
      (dummy1->s_difs) = ctalloc(double, num_regions);

      for (ir = 0; ir < num_regions; ir++) {
        region = NA_IndexToName(public_xtra->regions, ir);

        dummy1->region_indices[ir] =
            NA_NameToIndex(GlobalsGeomNames, region);

        if (dummy1->region_indices[ir] < 0) {
          InputError("Error: invalid geometry name <%s> for key <%s>\n",
                     region, "Phase.Saturation.GeomNames");
        }


        sprintf(key, "Geom.%s.Saturation.Alpha", region);
        dummy1->alphas[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.Saturation.N", region);
        dummy1->ns[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.Saturation.SRes", region);
        dummy1->s_ress[ir] = GetDouble(key);

        sprintf(key, "Geom.%s.Saturation.SSat", region);
        s_sat = GetDouble(key);

        (dummy1->s_difs[ir]) = s_sat - (dummy1->s_ress[ir]);
      }

      dummy1->alpha_file = NULL;
      dummy1->n_file = NULL;
      dummy1->s_res_file = NULL;
      dummy1->s_sat_file = NULL;
      dummy1->alpha_values = NULL;
      dummy1->n_values = NULL;
      dummy1->s_res_values = NULL;
      dummy1->s_sat_values = NULL;
    } else {
      sprintf(key, "Geom.%s.Saturation.Alpha.Filename", "domain");
      dummy1->alpha_file = GetString(key);
      sprintf(key, "Geom.%s.Saturation.N.Filename", "domain");
      dummy1->n_file = GetString(key);
      sprintf(key, "Geom.%s.Saturation.SRes.Filename", "domain");
      dummy1->s_res_file = GetString(key);
      sprintf(key, "Geom.%s.Saturation.SSat.Filename", "domain");
      dummy1->s_sat_file = GetString(key);

      dummy1->num_regions = 0;
      dummy1->region_indices = NULL;
      dummy1->alphas = NULL;
      dummy1->ns = NULL;
      dummy1->s_ress = NULL;
      dummy1->s_difs = NULL;
    }
  }
  (public_xtra->data) = (void*)dummy1;

  NA_FreeNameArray(type_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*--------------------------------------------------------------------------
 * VanGenuchtenFreePublicXtra
 *--------------------------------------------------------------------------*/

void  vanGenuchtenFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  //Type0       *dummy0;
  Type1       *dummy1;
  //Type2       *dummy2;
  //Type3       *dummy3;
  //Type4       *dummy4;
  //Type5       *dummy5;

  if (public_xtra)
  {
    if ((public_xtra->type) == 1 || (public_xtra->type) == 5) {  // BB only do this if van_genuchten is used. What do we need PFBFile for??

      NA_FreeNameArray(public_xtra->regions);
      dummy1 = (Type1 *) (public_xtra->data);

      if (dummy1->data_from_file == 1) {
        FreeVector(dummy1->alpha_values);
        FreeVector(dummy1->n_values);
        FreeVector(dummy1->s_res_values);
        FreeVector(dummy1->s_sat_values);
      }

      tfree(dummy1->region_indices);
      tfree(dummy1->alphas);
      tfree(dummy1->ns);
      tfree(dummy1->s_ress);
      tfree(dummy1->s_difs);
      tfree(dummy1);
      tfree(public_xtra);
    }
  }
}

/*--------------------------------------------------------------------------
 * VanGenuchtenSizeOfTempData
 *--------------------------------------------------------------------------*/

int  vanGenuchtenSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type1         *dummy1;

  int sz = 0;
  dummy1 = (Type1*)(public_xtra->data);
  if ((public_xtra->type) == 1 || (public_xtra->type) == 5) {  // BB only do this if van_genuchten is used. What do we need PFBFile for??
    if ((dummy1->data_from_file) == 1) {
      /* add local TempData size to `sz' */
      sz += SizeOfVector(dummy1->n_values);
      sz += SizeOfVector(dummy1->alpha_values);
      sz += SizeOfVector(dummy1->s_res_values);
      sz += SizeOfVector(dummy1->s_sat_values);
    }
  }
  return sz;
}


/* should define Invoke be here? */
