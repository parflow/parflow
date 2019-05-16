/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2019, Lawrence Livermore National Security,
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

#include "clustering.h"
#include "index_space.h"

#include "parflow.h"
#include "llnlmath.h"

typedef struct 
{
  Box box;
  int *histogram[DIM];
} HistogramBox;

/**
 * Reset the histogram.
 */
HistogramBox* NewHistogramBox(Box *box)
{
  HistogramBox* histogram_box = ctalloc(HistogramBox, 1);

  BoxCopy(&(histogram_box -> box), box);

  int size =  BoxSize(box);  

  for (int dim = 0; dim < DIM; dim++)
  {
    histogram_box -> histogram[dim] = ctalloc(int, size);
  }
}

/**
 * Reset the histogram.
 */
void ResetHistogram(HistogramBox *histogram_box)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    int size =  BoxSize(&(histogram_box -> box));
    for (int index = 0; index < size; index++) 
    {
      histogram_box -> histogram[dim][index] = 0;
    }
  }
}

/**
 * Bound the non-zero histogram elements within the interval
 * (box_lo, box_hi) in the given coordinate direction. Note that
 * computed bounding interval must be larger than given min_size.
 */
void BoundTagHistogram(HistogramBox *histogram_box,
		       int* box_lo,
		       int* box_up,
		       int dim,
		       int min_size)
{
  int hist_lo = histogram_box -> box.lo[dim];
  int hist_up = histogram_box -> box.up[dim];

  *box_lo = hist_lo;
  *box_up = hist_up;
  
  int width = (*box_up)-(*box_lo)+1;  

   if (width > min_size) {

     while ( ((*box_lo) <= (*box_up)) &&
	     (histogram_box -> histogram[dim][(*box_lo)] == 0) )
     {
       (*box_lo)++;
     }

     while ( ((*box_up) >= (*box_lo)) &&
	     (histogram_box -> histogram[dim][(*box_up)] == 0) ) 
     {
       (*box_up)--;
     }

     width = (*box_up)-(*box_lo)+1;  

     int pad = min_size - width;

     if (pad > 0) {
       (*box_lo) -= (pad+1)/2;
       if ((*box_lo) < hist_lo) 
       {
	 (*box_lo) = hist_lo;
       }

       (*box_up) = (*box_lo)+min_size-1;
       if ((*box_up) > hist_up) {
	 (*box_up) = hist_up;
	 (*box_lo) = MAX(hist_lo, (*box_up)-min_size+1);
       }
     }
   }
}

/**
 * Compute smallest box bounding non-zero histogram elements in
 * histogram box.
 */
void FindBoundBoxForTags(HistogramBox* histogram_box, Box* bound_box, Index min_box)
{
  Index box_lo;
  Index box_hi;

  /*
   * Compute extent of bounding box in each coordinate direction.
   */
  for (int dim = 0; dim < DIM; dim++) 
  {
    BoundTagHistogram(histogram_box, &box_lo[dim], &box_hi[dim], dim, min_box[dim]);
  }

  BoxSet(bound_box, box_lo, box_hi);
}

void ReduceTags(HistogramBox *histogram_box, Vector *vector, int dim)
{

  Box intersection;

  BoxCopy(&intersection, &(histogram_box -> box));
  
  int ilo = intersection.lo[dim];
  int ihi = intersection.up[dim];


  for(int ic_sb = ilo; ic_sb <= ihi; ic_sb++)
  {

    int tag_count = 0;

    Box src_box;
    BoxCopy(&src_box, &intersection);

    src_box.lo[dim] = ic_sb;
    src_box.up[dim] = ic_sb;

    {
      Grid* grid = VectorGrid(vector);
      Subvector* v_sub;
      double     *vp;
    
      Subgrid* subgrid;
    
      int ix, iy, iz;
      int nx, ny, nz;
      int nx_v, ny_v, nz_v;

      int i_s;
      int i, j, k, iv;

      Box bounding_box;
      Index lo, up;

      // SGS TODO this loop is not very efficient, should move the
      // src_box intersection check into loop bounds.
      ForSubgridI(i_s, GridSubgrids(grid))
      {
	subgrid = GridSubgrid(grid, i_s);
	
	v_sub = VectorSubvector(vector, i_s);
	
	ix = SubgridIX(subgrid);
	iy = SubgridIY(subgrid);
	iz = SubgridIZ(subgrid);
	
	nx = SubgridNX(subgrid);
	ny = SubgridNY(subgrid);
	nz = SubgridNZ(subgrid);
	
	nx_v = SubvectorNX(v_sub);
	ny_v = SubvectorNY(v_sub);
	nz_v = SubvectorNZ(v_sub);
	
	lo[0] = ix;
	lo[1] = iy;
	lo[2] = iz;
	
	up[0] = ix + nx - 1;
	up[1] = iy + ny - 1;
	up[2] = iz + nz - 1;
	
	vp = SubvectorElt(v_sub, ix, iy, iz);
	
	iv = 0;
	BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		  iv, nx_v, ny_v, nz_v, 1, 1, 1,
	{
	  if(vp[iv] != 0)
	  {
	    if( ( (src_box.lo[0] <= i) &&  (i <= src_box.up[0]) ) &&
		( (src_box.lo[1] <= j) &&  (j <= src_box.up[1]) ) &&
		( (src_box.lo[2] <= k) &&  (k <= src_box.up[2]) ) )
	    {
	      tag_count++;
	    }
	  }
	});
      }
	  
      histogram_box -> histogram[dim][ic_sb] += tag_count;
    }
  }
}

/**
 * Compute Tag Histogram 
 */
int ComputeTagHistogram(HistogramBox *histogram_box, Vector* vector)
{
   ResetHistogram(histogram_box);

   for(int dim = 0; dim < DIM; dim++)
   {
     ReduceTags(histogram_box, vector, dim);
   }

   int num_tags = 0;

   Index num_cells;
   
   BoxNumberCells(&(histogram_box -> box), &num_cells);

   int dim = 0;
   for (int index = 0; index < num_cells[dim]; index++) 
   {
     num_tags += (histogram_box->histogram)[dim][index];
   }

   return(num_tags);
}

/*
*************************************************************************
*                                                                       *
* Attempt to find a zero histogram value near the middle of the index   *
* interval (lo, hi) in the given coordinate direction. Note that the    *
* cut_pt is kept more than a minimium distance from the endpoints of    *
* of the index interval. Since box indices are cell-centered, the cut   *
* point value corresponds to the right edge of the cell whose index     *
* is equal to the cut point.                                            *
*                                                                       *
* Note that it is assumed that box indices are cell indices.            *
*                                                                       *
*************************************************************************
*/

int FindZeroCutPoint( int* cut_pt, 
		      int dim, 
		      int lo, 
		      int hi, 
		      HistogramBox* hist_box, 
		      int min_size)
{
  int cut_lo = lo + min_size - 1;
  int cut_hi = hi - min_size;
  int cut_mid = (lo + hi)/2;

  for (int ic = 0; ((cut_mid-ic>=cut_lo) && (cut_mid+ic<=cut_hi)); ic++) 
  {
    if (hist_box -> histogram[dim, cut_mid-ic] == 0) 
    {
      (*cut_pt) = cut_mid-ic;
      return TRUE;
    }
    if (hist_box -> histogram[dim, cut_mid+ic] == 0) 
    {
      (*cut_pt) = cut_mid+ic;
      return TRUE;
    }
  }
   
  return FALSE;
}

/*
***************************************************************************
*                                                                         *
* Attempt to find a point in the given coordinate direction near an       *
* inflection point in the histogram for that direction. Note that the     *
* cut point is kept more than a minimium distance from the endpoints      *
* of the index interval (lo, hi).  Also, the box must have at least       *
* three cells along a side to apply the Laplacian test.  If no            *
* inflection point is found, the mid-point of the interval is returned    *
* as the cut point.                                                       *
*                                                                         *
* Note that it is assumed that box indices are cell indices.              *
*                                                                         *
***************************************************************************
*/

void CutAtLaplacian(int* cut_pt, 
		    int dim,
		    int lo, 
		    int hi, 
		    HistogramBox* hist_box,
		    int min_size)
{
  int loc_zero = (lo + hi)/2;

  if ( (hi - lo + 1) >= 3 ) 
  {
      int max_zero = 0;
      int infpt_lo = MIN(lo + min_size - 1, lo + 1);  
      int infpt_hi = MIN(hi - min_size, hi - 2);  

      int last_lap = hist_box -> histogram[dim][infpt_lo - 1]
	- 2 * hist_box->histogram[dim][infpt_lo]
	+ hist_box -> histogram[dim][infpt_lo + 1];

      for (int ic = infpt_lo+1; ic <= infpt_hi+1; ic++) 
      {
	int new_lap = hist_box -> histogram[dim][ic - 1]
	  - 2 * hist_box->histogram[dim][ic]
	  + hist_box -> histogram[dim][ic + 1];

	if ( ((new_lap < 0) && (last_lap >= 0)) ||
	     ((new_lap >= 0) && (last_lap < 0)) ) {

	  int delta = new_lap - last_lap;

	  if ( delta < 0 ) {
	    delta = -delta; 
	  }

	  if ( delta > max_zero ) {
	    loc_zero = ic - 1;
	    max_zero = delta;
	  }
	}
     
	last_lap = new_lap;
      }
   }

  (*cut_pt) = loc_zero;
}

/*
*************************************************************************
*                                                                       *
* Attempt to split the box bounding a collection of tagged cells into   *
* two boxes. If an appropriate splitting is found, the two smaller      *
* boxes are returned and the return value of the function is true.      *
* Otherwise, false is returned. Note that the bounding box must be      *
* contained in the histogram box and that the two splitting boxes must  *
* be larger than some smallest box size.                                *
*                                                                       *
* Note that it is assumed that box indices are cell indices.            *
*                                                                       *
*************************************************************************
*/

int SplitTagBoundBox(Box* box_lft, 
		     Box* box_rgt,
		     Box* bound_box, 
		     HistogramBox* hist_box, 
		     Index min_box)
{
  int cut_pt = INT_MIN;
  int tmp_dim = -1; 
  int id = -1;
  
  Index num_cells;
  
  BoxNumberCells(bound_box, &num_cells);
  
  Index box_lo; 
  Index box_up; 
  
  IndexCopy(box_lo, bound_box -> lo);
  IndexCopy(box_up, bound_box -> up);
  
  /*
   * Sort the bound box dimensions from largest to smallest.
   */
  Index sorted;
  for (int dim = 0; dim < DIM; dim++) 
  {
    sorted[dim] = dim;
  }

  for (int dim0 = 0; dim0 < DIM-1; dim0++) 
  {
    for (int dim1 = dim0+1; dim1 < DIM; dim1++) 
    {
      if ( num_cells[sorted[dim0]] < num_cells[sorted[dim1]] ) 
      {
	tmp_dim = sorted[dim0];
	sorted[dim0] = sorted[dim1];
	sorted[dim1] = tmp_dim; 
      }
    }
  }

  /*
   * Determine number of coordinate directions in bounding box
   * that are splittable according to the minimum box size restriction.
   */
  int nsplit;
  for (nsplit = 0; nsplit < DIM; nsplit++) 
  {
    tmp_dim = sorted[nsplit];
    // Which test??
    if ( num_cells[tmp_dim] < 2*min_box[tmp_dim] ) 
    {
      break;
    }
    //    if ( (num_cells(tmp_dim) < 2*min_box(tmp_dim)) ||
    //         (num_cells(tmp_dim) < num_cells(sorted(0))/2) ) {
    //       break;
    //    }
  }

  if (nsplit == 0) 
  {
    return FALSE;
  }

  /*
   * Attempt to split box at a zero interior point in the histogram.
   * Check each splittable direction, from largest to smallest, until
   * zero point found.
   */
  for (int dim = 0; dim < nsplit; dim++) 
  {
    tmp_dim = sorted[dim];
    if ( FindZeroCutPoint(&cut_pt, 
			  tmp_dim, box_lo[tmp_dim], box_up[tmp_dim], 
			  hist_box, min_box[tmp_dim]) )
    {
      break;
    }
  }

  /*
   * If no zero point found, try Laplacian on longest side of bound box.
   */

  if (id == nsplit) 
  {
    tmp_dim = sorted[0];

    CutAtLaplacian(&cut_pt, 
		   tmp_dim, box_lo[tmp_dim], box_up[tmp_dim], 
		   hist_box, min_box[tmp_dim]);
  }

  /*
   * Split bound box at cut_pt; tmp_dim is splitting dimension.
   */
  Index lft_hi;
  IndexCopy(lft_hi, box_up); 
  Index rgt_lo;
  IndexCopy(rgt_lo, box_lo);
  lft_hi[tmp_dim] = cut_pt;
  rgt_lo[tmp_dim] = cut_pt + 1;

  BoxSet(box_lft, box_lo, lft_hi);
  BoxSet(box_rgt, rgt_lo, box_up);

  return 1;
}

void FindBoxesContainingTags(BoxList* boxes, 
			     Vector* vector,
			     Box* bound_box, 
			     Index min_box,
			     double efficiency_tol, 
			     double combine_tol)
{
  HistogramBox* hist_box = NewHistogramBox(bound_box);  

  int num_tags = ComputeTagHistogram(hist_box, vector);

  if ( num_tags == 0 ) 
  {
    BoxListClearItems(boxes);
  } 
  else 
  {
    Box tag_bound_box;
    FindBoundBoxForTags(hist_box, &tag_bound_box, min_box);

     int num_cells = BoxSize(&tag_bound_box);

     double efficiency = ( num_cells == 0 ? 1.e0 :
			   ((double) num_tags)/((double) num_cells) );

     if (efficiency <= efficiency_tol) 
     {
       Box box_lft;
       Box box_rgt;

       int is_split = SplitTagBoundBox(&box_lft, &box_rgt, &tag_bound_box, 
				       hist_box, min_box);
       
       if ( is_split ) 
       {
	 /*
	  * The bounding box "tag_bound_box" has been split into two
	  * boxes, "box_lft" and "box_rgt".  Now, attempt to recursively
	  * split these boxes further.
	  */
	 BoxList box_list_lft;
	 BoxList box_list_rgt;

	 FindBoxesContainingTags(&box_list_lft, 
				 vector,
				 &box_lft, min_box,
				 efficiency_tol, combine_tol);
	 FindBoxesContainingTags(&box_list_rgt, 
				 vector,
				 &box_rgt, min_box,
				 efficiency_tol, combine_tol);

	 if ( (( BoxListSize(&box_list_lft) > 1) ||
	       ( BoxListSize(&box_list_rgt) > 1)) ||
	      ( (double) (BoxSize(BoxListFront(&box_list_lft))
			  + BoxSize(BoxListFront(&box_list_rgt)))
		< ((double) BoxSize(&tag_bound_box))*combine_tol ) )
	 {
	   BoxListConcatenate(boxes, &box_list_lft);
	   BoxListConcatenate(boxes, &box_list_rgt);
	 }
       }
     }

     /*
      * If no good splitting is found, add bounding box to list.
      */
     if ( BoxListIsEmpty(boxes) ) 
     {
       BoxListAppend(boxes, &tag_bound_box);
     }
  }
}

void BergerRigoutsos(Vector* vector,
		     Index min_box,
		     double efficiency_tol, 
		     double combine_tol,
		     BoxList* boxes)
{
  Grid* grid = VectorGrid(vector);
  Subvector* v_sub;
  Subgrid* subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s;

  Box bounding_box;
  Index lo, up;

  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = VectorSubvector(vector, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(v_sub);
    ny_v = SubvectorNY(v_sub);
    nz_v = SubvectorNZ(v_sub);

    lo[0] = ix;
    lo[1] = iy;
    lo[2] = iz;

    up[0] = ix + nx - 1;
    up[1] = iy + ny - 1;
    up[2] = iz + nz - 1;
  }

  BoxSet(&bounding_box, lo, up);
  HistogramBox* histogram_box = NewHistogramBox(&bounding_box);

  int num_tags = ComputeTagHistogram(histogram_box, vector);

   if ( num_tags == 0 ) {
      BoxListClearItems(boxes);
   } 
   else
   {
     Box tag_bound_box;

     FindBoundBoxForTags(histogram_box, &tag_bound_box, min_box);

     int num_cells = BoxSize(&tag_bound_box);

     double efficiency = ( num_cells == 0 ? 1.e0 :
                          ((double) num_tags)/((double) num_cells) );
     
     if (efficiency <= efficiency_tol) 
     {
       Box box_lft;
       Box box_rgt;

       int is_split = SplitTagBoundBox(&box_lft, &box_rgt, &tag_bound_box, 
				       histogram_box, min_box);

       if ( is_split ) 
       {
	 /*
	  * The bounding box "tag_bound_box" has been split into two
	  * boxes, "box_lft" and "box_rgt".  Now, attempt to recursively
	  * split these boxes further.
	  */
	 BoxList box_list_lft;
	 BoxList box_list_rgt;

	 FindBoxesContainingTags(&box_list_lft, 
				 vector,
				 &box_lft, min_box,
				 efficiency_tol, combine_tol);
	 FindBoxesContainingTags(&box_list_rgt, 
				 vector,
				 &box_rgt, min_box,
				 efficiency_tol, combine_tol);

	 if ( (( BoxListSize(&box_list_lft) > 1) ||
	       ( BoxListSize(&box_list_rgt) > 1)) ||
	      ( (double) (BoxSize(BoxListFront(&box_list_lft))
			  + BoxSize(BoxListFront(&box_list_rgt)))
		< ((double) BoxSize(&tag_bound_box))*combine_tol ) )
	 {
	   BoxListConcatenate(boxes, &box_list_lft);
	   BoxListConcatenate(boxes, &box_list_rgt);
	 }
       }
     }

     /*
      * If no good splitting is found, add bounding box to list.
      */
     if ( BoxListIsEmpty(boxes) ) 
     {
       BoxListAppend(boxes, &tag_bound_box);
     }
   }
}

/**
 * Compute set of boxes that exactly cover the GeomSolid.
 *
 * Runs the Berger-Rigoutsos algorithm to compute an 
 * set of boxes that cover the iterations spaces 
 * in the octree for more efficient iteration.   The 
 * Octree boxes are not anywhere near a minimal set 
 * of boxes.
 *
 * This assumes Octree's are in background grid space.
 */
void ComputeBoxes(GrGeomSolid *geom_solid)
{

  Grid *grid =  CreateGrid(GlobalsUserGrid);

  printf("SGS OctreeComputeBoxes\n");
  Vector* indicator =  NewVectorType(grid, 1, 0, vector_cell_centered);
  InitVectorAll(indicator, 0.0);

  {
    Grid        *grid = VectorGrid(indicator);
    Subgrid     *subgrid;

    Subvector   *d_sub;

    int ip;

    int i, j, k, r, is;
    int ix, iy, iz;
    int nx, ny, nz;
    double  dx, dy, dz;

    double *dp;

    int tag_count = 0;

    ForSubgridI(is, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, is);

      d_sub = VectorSubvector(indicator, is);

      /* RDF: assumes resolutions are the same in all 3 directions */
      r = SubgridRX(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
      
      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
      
      dp = SubvectorData(d_sub);

      GrGeomInLoop(i, j, k, geom_solid, r, ix, iy, iz, nx, ny, nz,
      {
	ip = SubvectorEltIndex(d_sub, i, j, k);

	dp[ip] = 1;
	tag_count++;
      });
    }
    
    printf("SGSDEBUG: Tag Count set %d\n", tag_count);
  }

  Index min_box;
  min_box[0] = 1;
  min_box[1] = 1;
  min_box[2] = 1;

  double efficiency_tol = 0.99;
  double combine_tol = 0.1;

  BoxList* boxes = NewBoxList();

  BergerRigoutsos(indicator,
		  min_box,
		  efficiency_tol, 
		  combine_tol,
		  boxes);

  printf("SGS OctreeComputeBoxes BR clustering : \n");
  BoxListPrint(boxes);
  printf("SGS End\n");

  
}

