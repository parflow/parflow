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

#include "parflow.h"
#include "clustering.h"
#include "index_space.h"
#include "llnlmath.h"

#include <string.h>

/**
 * This implementation is derived from the SAMRAI Berger-Rigoutsos
 * implementation developed by LLNL.
 *
 * https://computation.llnl.gov/projects/samrai
 *
 * The Berger-Rigoutsos algorithm is commonly used in AMR applications
 * to cluster tagged cells into boxes.  This implementation was
 * modified to compute an exact covering of tagged cells.  The traditional
 * BR algorithm allows for boxes that cover tagged cells and some untagged cells to
 * some percentage.  For the looping the boxes should cover exactly the tagged cells.
 *
 * The algorithm is described in Berger and Rigoutsos, IEEE Trans. on
 * Sys, Man, and Cyber (21)5:1278-1286.
 */

/**
 * Tag count.
 */
static int tag_count;

/**
 * Maximum number of ghost layers.
 *
 * This needs to be the maximum number of ghost layers in any vector
 * in the problem in order to correctly find the iteration spaces.
 *
 * @TODO Obviously this is not a great constant to have; not sure how
 * to address this.  The geometries are defined before all the vectors
 * have been created.
 */
static const int num_ghost = 4;

/**
 * Store tags in a double.
 *
 * Vectors can only store doubles but want to store tag values so
 * use union to overlay tags on the double.
 */
typedef union {
  double as_double;
  unsigned int as_tags;
} DoubleTags;

/**
 * Structure to store histogram for BR algorithm.
 */
typedef struct {
  Box box;
  int *histogram[DIM];
} HistogramBox;

/**
 * Get tags along the provided dimension for the global index along that dimension.
 */
int HistogramBoxGetTags(HistogramBox *histogram_box, int dim, int global_index)
{
  return histogram_box->histogram[dim][global_index - histogram_box->box.lo[dim]];
}

/**
 * Add tags along the provided dimension for the global index along that dimension.
 */
void HistogramBoxAddTags(HistogramBox *histogram_box, int dim, int global_index, int tag_count)
{
  histogram_box->histogram[dim][global_index - histogram_box->box.lo[dim]] += tag_count;
}

/**
 * Create a new histogram box for the index space spanned by the provided box.
 */
HistogramBox* NewHistogramBox(Box *box)
{
  HistogramBox* histogram_box = talloc(HistogramBox, 1);

  memset(histogram_box, 0, sizeof(HistogramBox));

  BoxCopy(&(histogram_box->box), box);

  int size = BoxSize(&(histogram_box->box));

  for (int dim = 0; dim < DIM; dim++)
  {
    histogram_box->histogram[dim] = talloc(int, size);
    memset(histogram_box->histogram[dim], 0, size * sizeof(int));
  }

  return histogram_box;
}

/**
 * Free the histogram box.
 */
void FreeHistogramBox(HistogramBox* histogram_box)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    tfree(histogram_box->histogram[dim]);
  }

  tfree(histogram_box);
}

/**
 * Reset the histogram.
 *
 * Clears tag counts in the histogram box.
 */
void ResetHistogram(HistogramBox *histogram_box)
{
  for (int dim = 0; dim < DIM; dim++)
  {
    int size = BoxSize(&(histogram_box->box));
    for (int index = 0; index < size; index++)
    {
      histogram_box->histogram[dim][index] = 0;
    }
  }
}

/**
 * Bound the non-zero histogram elements within the interval
 * (box_lo, box_hi) in the given coordinate direction. Note that
 * computed bounding interval must be larger than given min_size.
 */
void BoundTagHistogram(HistogramBox *histogram_box,
                       int*          box_lo,
                       int*          box_up,
                       int           dim,
                       int           min_size)
{
  int hist_lo = histogram_box->box.lo[dim];
  int hist_up = histogram_box->box.up[dim];

  *box_lo = hist_lo;
  *box_up = hist_up;

  int width = (*box_up) - (*box_lo) + 1;

  if (width > min_size)
  {
    while (((*box_lo) <= (*box_up)) &&
           (HistogramBoxGetTags(histogram_box, dim, *box_lo) == 0))
    {
      (*box_lo)++;
    }

    while (((*box_up) >= (*box_lo)) &&
           (HistogramBoxGetTags(histogram_box, dim, *box_up) == 0))
    {
      (*box_up)--;
    }

    width = (*box_up) - (*box_lo) + 1;

    int pad = min_size - width;

    if (pad > 0)
    {
      (*box_lo) -= (pad + 1) / 2;
      if ((*box_lo) < hist_lo)
      {
        (*box_lo) = hist_lo;
      }

      (*box_up) = (*box_lo) + min_size - 1;
      if ((*box_up) > hist_up)
      {
        (*box_up) = hist_up;
        (*box_lo) = MAX(hist_lo, (*box_up) - min_size + 1);
      }
    }
  }
}

/**
 * Compute smallest box bounding for the non-zero histogram elements
 * in histogram box.
 */
void FindBoundBoxForTags(HistogramBox* histogram_box, Box* bound_box, Point min_box)
{
  Point box_lo;
  Point box_hi;

  /*
   * Compute extent of bounding box in each coordinate direction.
   */
  for (int dim = 0; dim < DIM; dim++)
  {
    BoundTagHistogram(histogram_box, &box_lo[dim], &box_hi[dim], dim, min_box[dim]);
  }

  BoxSet(bound_box, box_lo, box_hi);
}

/**
 * Reduces tag counts along an axis.
 *
 * Reduce (count) the number of cells that have the specified tag
 * along the provided dimension and store into the provided histogram
 * box.
 */
void ReduceTags(HistogramBox *histogram_box, Vector *vector, int dim, DoubleTags tag)
{
  Box intersection;

  BoxClear(&intersection);

  BoxCopy(&intersection, &(histogram_box->box));

  int ilo = intersection.lo[dim];
  int ihi = intersection.up[dim];


  for (int ic_sb = ilo; ic_sb <= ihi; ic_sb++)
  {
    tag_count = 0;

    Box src_box;
    BoxClear(&src_box);

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
      BoxClear(&bounding_box);
      //      Point lo, up;

      ForSubgridI(i_s, GridSubgrids(grid))
      {
        subgrid = GridSubgrid(grid, i_s);

        v_sub = VectorSubvector(vector, i_s);

        ix = SubgridIX(subgrid) - num_ghost;
        iy = SubgridIY(subgrid) - num_ghost;
        iz = SubgridIZ(subgrid) - num_ghost;

        nx = SubgridNX(subgrid) + 2 * num_ghost;
        ny = SubgridNY(subgrid) + 2 * num_ghost;
        nz = SubgridNZ(subgrid) + 2 * num_ghost;

        nx_v = SubvectorNX(v_sub);
        ny_v = SubvectorNY(v_sub);
        nz_v = SubvectorNZ(v_sub);

        ix = pfmax(ix, src_box.lo[0]);
        iy = pfmax(iy, src_box.lo[1]);
        iz = pfmax(iz, src_box.lo[2]);

        nx = pfmin(nx, src_box.up[0] - src_box.lo[0] + 1);
        ny = pfmin(ny, src_box.up[1] - src_box.lo[1] + 1);
        nz = pfmin(nz, src_box.up[2] - src_box.lo[2] + 1);
        vp = SubvectorElt(v_sub, ix, iy, iz);

        iv = 0;
        BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                  iv, nx_v, ny_v, nz_v, 1, 1, 1,
        {
          DoubleTags v;
          v.as_double = vp[iv];

          if (v.as_tags & tag.as_tags)
          {
            if (((src_box.lo[0] <= i) && (i <= src_box.up[0])) &&
                ((src_box.lo[1] <= j) && (j <= src_box.up[1])) &&
                ((src_box.lo[2] <= k) && (k <= src_box.up[2])))
            {
              PlusEquals(tag_count, 1);
            }
          }
        });
      }
      HistogramBoxAddTags(histogram_box, dim, ic_sb, tag_count);
    }
  }
}

/**
 * Compute Tag Histogram along all dimensions.
 */
int ComputeTagHistogram(HistogramBox *histogram_box, Vector* vector, DoubleTags tag)
{
  int num_tags = 0;
  Point num_cells;

  ResetHistogram(histogram_box);

  for (int dim = 0; dim < DIM; dim++)
  {
    ReduceTags(histogram_box, vector, dim, tag);
  }

  BoxNumberCells(&(histogram_box->box), &num_cells);

  int dim = 0;
  for (int index = 0; index < num_cells[dim]; index++)
  {
    num_tags += (histogram_box->histogram)[dim][index];
  }

  return(num_tags);
}

/**
 * Attempt to find zero cut point.
 *
 * Attempt to find a zero histogram value near the middle of the index
 * interval (lo, hi) in the given coordinate direction. Note that the
 * cut_pt is kept more than a minimium distance from the endpoints of
 * of the index interval. Since box indices are cell-centered, the cut
 * point value corresponds to the right edge of the cell whose index
 * is equal to the cut point.
 */
int FindZeroCutPoint(int*          cut_pt,
                     int           dim,
                     int           lo,
                     int           hi,
                     HistogramBox* hist_box,
                     int           min_size)
{
  int cut_lo = lo + min_size - 1;
  int cut_hi = hi - min_size;
  int cut_mid = (lo + hi) / 2;

  for (int ic = 0; ((cut_mid - ic >= cut_lo) && (cut_mid + ic <= cut_hi)); ic++)
  {
    if (HistogramBoxGetTags(hist_box, dim, cut_mid - ic) == 0)
    {
      (*cut_pt) = cut_mid - ic;
      return TRUE;
    }
    if (HistogramBoxGetTags(hist_box, dim, cut_mid + ic) == 0)
    {
      (*cut_pt) = cut_mid + ic;
      return TRUE;
    }
  }

  return FALSE;
}

/**
 * Attempt to find cut point.
 *
 * Attempt to find a point in the given coordinate direction near an
 * inflection point in the histogram for that direction. Note that the
 * cut point is kept more than a minimium distance from the endpoints
 * of the index interval (lo, hi).  Also, the box must have at least
 * three cells along a side to apply the Laplacian test.  If no
 * inflection point is found, the mid-point of the interval is
 * returned as the cut point.
 */
void CutAtLaplacian(int*          cut_pt,
                    int           dim,
                    int           lo,
                    int           hi,
                    HistogramBox* hist_box,
                    int           min_size)
{
  int loc_zero = (lo + hi) / 2;

  if ((hi - lo + 1) >= 3)
  {
    int max_zero = 0;
    int infpt_lo = MAX(lo + min_size - 1, lo + 1);
    int infpt_hi = MIN(hi - min_size, hi - 2);

    int last_lap = HistogramBoxGetTags(hist_box, dim, infpt_lo - 1)
                   - 2 * HistogramBoxGetTags(hist_box, dim, infpt_lo)
                   + HistogramBoxGetTags(hist_box, dim, infpt_lo + 1);

    for (int ic = infpt_lo + 1; ic <= infpt_hi + 1; ic++)
    {
      int new_lap = HistogramBoxGetTags(hist_box, dim, ic - 1)
                    - 2 * HistogramBoxGetTags(hist_box, dim, ic)
                    + HistogramBoxGetTags(hist_box, dim, ic + 1);

      if (((new_lap < 0) && (last_lap >= 0)) ||
          ((new_lap >= 0) && (last_lap < 0)))
      {
        int delta = new_lap - last_lap;

        if (delta < 0)
        {
          delta = -delta;
        }

        if (delta > max_zero)
        {
          loc_zero = ic - 1;
          max_zero = delta;
        }
      }

      last_lap = new_lap;
    }
  }

  (*cut_pt) = loc_zero;
}

/**
 * Attempt to split box into two boxes.
 *
 * Attempt to split the box bounding a collection of tagged cells into
 * two boxes. If an appropriate splitting is found, the two smaller
 * boxes are returned and the return value of the function is true.
 * Otherwise, false is returned. Note that the bounding box must be
 * contained in the histogram box and that the two splitting boxes
 * must be larger than some smallest box size.
 */
int SplitTagBoundBox(Box*          box_lft,
                     Box*          box_rgt,
                     Box*          bound_box,
                     HistogramBox* hist_box,
                     Point         min_box)
{
  int cut_pt = INT_MIN;
  int tmp_dim = -1;
  int dim = -1;

  Point num_cells;

  BoxNumberCells(bound_box, &num_cells);

  Point box_lo;
  Point box_up;

  PointCopy(box_lo, bound_box->lo);
  PointCopy(box_up, bound_box->up);

  /*
   * Sort the bound box dimensions from largest to smallest.
   */
  Point sorted;
  for (dim = 0; dim < DIM; dim++)
  {
    sorted[dim] = dim;
  }

  for (int dim0 = 0; dim0 < DIM - 1; dim0++)
  {
    for (int dim1 = dim0 + 1; dim1 < DIM; dim1++)
    {
      if (num_cells[sorted[dim0]] < num_cells[sorted[dim1]])
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
    if (num_cells[tmp_dim] < 2 * min_box[tmp_dim])
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
  for (dim = 0; dim < nsplit; dim++)
  {
    tmp_dim = sorted[dim];
    if (FindZeroCutPoint(&cut_pt,
                         tmp_dim, box_lo[tmp_dim], box_up[tmp_dim],
                         hist_box, min_box[tmp_dim]))
    {
      break;
    }
  }

  /*
   * If no zero point found, try Laplacian on longest side of bound box.
   */

  if (dim == nsplit)
  {
    tmp_dim = sorted[0];

    CutAtLaplacian(&cut_pt,
                   tmp_dim, box_lo[tmp_dim], box_up[tmp_dim],
                   hist_box, min_box[tmp_dim]);
  }

  /*
   * Split bound box at cut_pt; tmp_dim is splitting dimension.
   */
  Point lft_hi;
  PointCopy(lft_hi, box_up);
  Point rgt_lo;
  PointCopy(rgt_lo, box_lo);
  lft_hi[tmp_dim] = cut_pt;
  rgt_lo[tmp_dim] = cut_pt + 1;

  BoxSet(box_lft, box_lo, lft_hi);
  BoxSet(box_rgt, rgt_lo, box_up);

  return TRUE;
}

/**
 * Recursive function to compute boxes that cover cells with the provided tag.
 *
 * Create a list of boxes that exactly cover all tags that match the
 * specified tag value.
 *
 */
void FindBoxesContainingTags(BoxList*   boxes,
                             Vector*    vector,
                             Box*       bound_box,
                             Point      min_box,
                             DoubleTags tag)
{
  HistogramBox* hist_box = NewHistogramBox(bound_box);

  int num_tags = ComputeTagHistogram(hist_box, vector, tag);

  if (num_tags == 0)
  {
    BoxListClearItems(boxes);
  }
  else
  {
    Box tag_bound_box;
    BoxClear(&tag_bound_box);

    FindBoundBoxForTags(hist_box, &tag_bound_box, min_box);

    int num_cells = BoxSize(&tag_bound_box);

    if (num_tags < num_cells)
    {
      Box box_lft;
      Box box_rgt;
      BoxClear(&box_lft);
      BoxClear(&box_rgt);

      int is_split = SplitTagBoundBox(&box_lft, &box_rgt, &tag_bound_box,
                                      hist_box, min_box);

      if (is_split)
      {
        /*
         * The bounding box "tag_bound_box" has been split into two
         * boxes, "box_lft" and "box_rgt".  Now, attempt to recursively
         * split these boxes further.
         */
        BoxList* box_list_lft = NewBoxList();
        BoxList* box_list_rgt = NewBoxList();

        FindBoxesContainingTags(box_list_lft,
                                vector,
                                &box_lft, min_box,
                                tag);
        FindBoxesContainingTags(box_list_rgt,
                                vector,
                                &box_rgt, min_box,
                                tag);

        if (((BoxListSize(box_list_lft) > 1) ||
             (BoxListSize(box_list_rgt) > 1)) ||
            ((double)(BoxSize(BoxListFront(box_list_lft))
                      + BoxSize(BoxListFront(box_list_rgt)))
             < ((double)BoxSize(&tag_bound_box))))
        {
          BoxListConcatenate(boxes, box_list_lft);
          BoxListConcatenate(boxes, box_list_rgt);
        }

        FreeBoxList(box_list_lft);
        FreeBoxList(box_list_rgt);
      }
    }

    /*
     * If no good splitting is found, add bounding box to list.
     */
    if (BoxListIsEmpty(boxes))
    {
      BoxListAppend(boxes, &tag_bound_box);
    }
  }

  FreeHistogramBox(hist_box);
}

/**
 * Compute boxes that cover cells with the provided tag.
 *
 * Create a list of boxes that exactly cover all tags that match the
 * specified tag value.
 *
 */
void BergerRigoutsos(Vector*    vector,
                     Point      min_box,
                     DoubleTags tag,
                     BoxList*   boxes)
{
  Grid* grid = VectorGrid(vector);
  Subgrid* subgrid;

  int ix, iy, iz;
  int nx, ny, nz;

  int i_s;

  Box bounding_box;

  BoxClear(&bounding_box);

  Point lo, up;

  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid) - num_ghost;
    iy = SubgridIY(subgrid) - num_ghost;
    iz = SubgridIZ(subgrid) - num_ghost;

    nx = SubgridNX(subgrid) + 2 * num_ghost;
    ny = SubgridNY(subgrid) + 2 * num_ghost;
    nz = SubgridNZ(subgrid) + 2 * num_ghost;

    lo[0] = ix;
    lo[1] = iy;
    lo[2] = iz;

    up[0] = ix + nx - 1;
    up[1] = iy + ny - 1;
    up[2] = iz + nz - 1;
  }

  BoxSet(&bounding_box, lo, up);

  HistogramBox* histogram_box = NewHistogramBox(&bounding_box);

  int num_tags = ComputeTagHistogram(histogram_box, vector, tag);

  if (num_tags == 0)
  {
    BoxListClearItems(boxes);
  }
  else
  {
    Box tag_bound_box;
    BoxClear(&tag_bound_box);

    FindBoundBoxForTags(histogram_box, &tag_bound_box, min_box);

    int num_cells = BoxSize(&tag_bound_box);

    if (num_tags < num_cells)
    {
      Box box_lft;
      Box box_rgt;

      BoxClear(&box_lft);
      BoxClear(&box_rgt);

      int is_split = SplitTagBoundBox(&box_lft, &box_rgt, &tag_bound_box,
                                      histogram_box, min_box);

      if (is_split)
      {
        /*
         * The bounding box "tag_bound_box" has been split into two
         * boxes, "box_lft" and "box_rgt".  Now, attempt to recursively
         * split these boxes further.
         */
        BoxList* box_list_lft = NewBoxList();
        BoxList* box_list_rgt = NewBoxList();

        FindBoxesContainingTags(box_list_lft,
                                vector,
                                &box_lft, min_box,
                                tag);
        FindBoxesContainingTags(box_list_rgt,
                                vector,
                                &box_rgt, min_box,
                                tag);

        if (((BoxListSize(box_list_lft) > 1) ||
             (BoxListSize(box_list_rgt) > 1)) ||
            ((double)(BoxSize(BoxListFront(box_list_lft))
                      + BoxSize(BoxListFront(box_list_rgt)))
             < ((double)BoxSize(&tag_bound_box))))
        {
          BoxListConcatenate(boxes, box_list_lft);
          BoxListConcatenate(boxes, box_list_rgt);
        }

        FreeBoxList(box_list_lft);
        FreeBoxList(box_list_rgt);
      }
    }

    /*
     * If no good splitting is found, add bounding box to list.
     */
    if (BoxListIsEmpty(boxes))
    {
      BoxListAppend(boxes, &tag_bound_box);
    }
  }

  FreeHistogramBox(histogram_box);
}


/**
 * Compute the patch loop iteration boxes.
 *
 * Compute the patch loop iteration boxes for the specified patch.  Patches
 * are looped over in each face direction so a box array is generated for
 * each face direction.
 *
 * The computed box arrays are stored in the geom_solid.
 */
void ComputePatchBoxes(GrGeomSolid *geom_solid, int patch)
{
  Grid *grid = CreateGrid(GlobalsUserGrid);

  Vector* indicator = NewVectorType(grid, 1, num_ghost, vector_cell_centered);

  InitVectorAll(indicator, 0.0);

  DoubleTags tag;

  {
    Grid        *grid = VectorGrid(indicator);
    Subgrid     *subgrid;

    Subvector   *d_sub;

    int i, j, k, r, is;
    int ix, iy, iz;
    int nx, ny, nz;

    double *dp;

    tag_count = 0;

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

      dp = SubvectorData(d_sub);

      int *fdir;
      GrGeomPatchLoop(i, j, k, fdir, geom_solid, patch,
                      r, ix, iy, iz, nx, ny, nz,
      {
        int ip = SubvectorEltIndex(d_sub, i, j, k);
        int this_face_tag = 1 << PV_f;

        DoubleTags v;
        v.as_double = dp[ip];
        v.as_tags = v.as_tags | this_face_tag;
        dp[ip] = v.as_double;

        PlusEquals(tag_count, 1);
      });
    }
  }

  {
    VectorUpdateCommHandle   *handle;
    handle = InitVectorUpdate(indicator, VectorUpdateAll);
    FinalizeVectorUpdate(handle);
  }

  Point min_box;
  min_box[0] = 1;
  min_box[1] = 1;
  min_box[2] = 1;

  for (int face = 0; face < GrGeomOctreeNumFaces; face++)
  {
    BoxList* boxes = NewBoxList();

    tag.as_tags = 1 << face;

    BergerRigoutsos(indicator,
                    min_box,
                    tag,
                    boxes);

    GrGeomSolidPatchBoxes(geom_solid, patch, face) = NewBoxArray(boxes);

    FreeBoxList(boxes);
  }

  FreeVector(indicator);
  FreeGrid(grid);
}

/**
 * Compute the surface loop iteration boxes.
 *
 * Compute the surface loop iteration boxes.  Surface loops are looped
 * over in each face direction so a box array is generated for each
 * face direction.
 *
 * The computed box arrays are stored in the geom_solid.
 */
void ComputeSurfaceBoxes(GrGeomSolid *geom_solid)
{
  Grid *grid = CreateGrid(GlobalsUserGrid);

  Vector* indicator = NewVectorType(grid, 1, num_ghost, vector_cell_centered);

  InitVectorAll(indicator, 0.0);

  DoubleTags tag;

  {
    Grid        *grid = VectorGrid(indicator);
    Subgrid     *subgrid;

    Subvector   *d_sub;

    int i, j, k, r, is;
    int ix, iy, iz;
    int nx, ny, nz;

    double *dp;

    tag_count = 0;

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

      dp = SubvectorData(d_sub);

      int *fdir;
      GrGeomSurfLoop(i, j, k, fdir, geom_solid, r, ix, iy, iz, nx, ny, nz,
      {
        int ip = SubvectorEltIndex(d_sub, i, j, k);
        int this_face_tag = 1 << PV_f;

        DoubleTags v;
        v.as_double = dp[ip];
        v.as_tags = v.as_tags | this_face_tag;
        dp[ip] = v.as_double;

        PlusEquals(tag_count, 1);
      });
    }
  }

  {
    VectorUpdateCommHandle   *handle;
    handle = InitVectorUpdate(indicator, VectorUpdateAll);
    FinalizeVectorUpdate(handle);
  }

  Point min_box;
  min_box[0] = 1;
  min_box[1] = 1;
  min_box[2] = 1;

  for (int face = 0; face < GrGeomOctreeNumFaces; face++)
  {
    BoxList* boxes = NewBoxList();

    tag.as_tags = 1 << face;

    BergerRigoutsos(indicator,
                    min_box,
                    tag,
                    boxes);

    GrGeomSolidSurfaceBoxes(geom_solid, face) = NewBoxArray(boxes);

    FreeBoxList(boxes);
  }

  FreeVector(indicator);
  FreeGrid(grid);
}

/**
 * Compute the interior loop iteration boxes.
 *
 * Compute the interior loop iteration boxes.  Boxes will exactly cover
 * all of interior of geom_solid in index space.
 *
 * The computed box array is stored in the geom_solid.
 */
void ComputeInteriorBoxes(GrGeomSolid *geom_solid)
{
  DoubleTags tag;

  tag.as_tags = 1;

  Grid *grid = CreateGrid(GlobalsUserGrid);

  Vector* indicator = NewVectorType(grid, 1, num_ghost, vector_cell_centered);
  InitVectorAll(indicator, 0.0);

  {
    Grid        *grid = VectorGrid(indicator);
    Subgrid     *subgrid;

    Subvector   *d_sub;

    int i, j, k, r, is;
    int ix, iy, iz;
    int nx, ny, nz;

    double *dp;

    tag_count = 0;

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

      dp = SubvectorData(d_sub);

      GrGeomInLoop(i, j, k, geom_solid, r, ix, iy, iz, nx, ny, nz,
      {
        int ip = SubvectorEltIndex(d_sub, i, j, k);

        dp[ip] = tag.as_double;
        PlusEquals(tag_count, 1);
      });
    }
  }

  {
    VectorUpdateCommHandle   *handle;
    handle = InitVectorUpdate(indicator, VectorUpdateAll);
    FinalizeVectorUpdate(handle);
  }

  Point min_box;
  min_box[0] = 1;
  min_box[1] = 1;
  min_box[2] = 1;

  BoxList* boxes = NewBoxList();

  BergerRigoutsos(indicator,
                  min_box,
                  tag,
                  boxes);

  GrGeomSolidInteriorBoxes(geom_solid) = NewBoxArray(boxes);

  FreeBoxList(boxes);
  FreeVector(indicator);
  FreeGrid(grid);
}

void ComputeBoxes(GrGeomSolid *geom_solid)
{
  BeginTiming(ClusteringTimingIndex);

  ComputeInteriorBoxes(geom_solid);

  ComputeSurfaceBoxes(geom_solid);

  for (int patch = 0; patch < GrGeomSolidNumPatches(geom_solid); patch++)
  {
    ComputePatchBoxes(geom_solid, patch);
  }

  EndTiming(ClusteringTimingIndex);
}
