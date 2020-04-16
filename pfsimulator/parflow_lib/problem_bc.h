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
*****************************************************************************/

#ifndef _PROBLEM_BC_HEADER
#define _PROBLEM_BC_HEADER

#define DirichletBC  0
#define FluxBC       1
#define OverlandBC   2   //sk
#define SeepageFaceBC   3   //rmm

/* @MCB: Additional overlandflow cases per LEC */
#define OverlandKinematicBC 4
#define OverlandDiffusiveBC 5

/*----------------------------------------------------------------
 * BCStruct structure
 *----------------------------------------------------------------*/

typedef struct {
  SubgridArray    *subgrids;   /* subgrids that BC data is defined on */

  GrGeomSolid     *gr_domain;

  int num_patches;
  int             *patch_indexes;  /* num_patches patch indexes */
  int             *bc_types;          /* num_patches BC types */

  double        ***values;   /* num_patches x num_subgrids data arrays */
} BCStruct;


/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define BCStructSubgrids(bc_struct)           ((bc_struct)->subgrids)
#define BCStructNumSubgrids(bc_struct) \
  SubgridArrayNumSubgrids(BCStructSubgrids(bc_struct))
#define BCStructGrDomain(bc_struct)           ((bc_struct)->gr_domain)
#define BCStructNumPatches(bc_struct)         ((bc_struct)->num_patches)
#define BCStructPatchIndexes(bc_struct)       ((bc_struct)->patch_indexes)
#define BCStructBCTypes(bc_struct)            ((bc_struct)->bc_types)
#define BCStructValues(bc_struct)             ((bc_struct)->values)
#define BCStructSubgrid(bc_struct, i) \
  SubgridArraySubgrid(BCStructSubgrids(bc_struct), i)
#define BCStructPatchIndex(bc_struct, p)      ((bc_struct)->patch_indexes[p])
#define BCStructBCType(bc_struct, p)          ((bc_struct)->bc_types[p])
#define BCStructPatchValues(bc_struct, p, s)  ((bc_struct)->values[p][s])

#define ForBCStructNumPatches(ipatch, bc_struct)										\
  for(ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)

/*--------------------------------------------------------------------------
 * Looping macro:
 *--------------------------------------------------------------------------*/

#define BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is, body) \
  {                                                                         \
    GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);               \
    int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);             \
    Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);              \
                                                                            \
    int PV_r = SubgridRX(PV_subgrid);                                       \
    int PV_ix = SubgridIX(PV_subgrid);                                      \
    int PV_iy = SubgridIY(PV_subgrid);                                      \
    int PV_iz = SubgridIZ(PV_subgrid);                                      \
    int PV_nx = SubgridNX(PV_subgrid);                                      \
    int PV_ny = SubgridNY(PV_subgrid);                                      \
    int PV_nz = SubgridNZ(PV_subgrid);                                      \
                                                                            \
    ival = 0;                                                               \
    GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,            \
                    PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,         \
    {                                                                       \
      body;                                                                 \
      ival++;                                                               \
    });                                                                     \
  }

#define BCStructPatchLoopOvrlnd(i, j, k, fdir, ival, bc_struct, ipatch, is, body) \
  {                                                                               \
    GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);                     \
    int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);                   \
    Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);                    \
                                                                                  \
    int PV_r = SubgridRX(PV_subgrid);                                             \
    int PV_ix = SubgridIX(PV_subgrid) - 1;                                        \
    int PV_iy = SubgridIY(PV_subgrid) - 1;                                        \
    int PV_iz = SubgridIZ(PV_subgrid) - 1;                                        \
    int PV_nx = SubgridNX(PV_subgrid) + 2;                                        \
    int PV_ny = SubgridNY(PV_subgrid) + 2;                                        \
    int PV_nz = SubgridNZ(PV_subgrid) + 2;                                        \
                                                                                  \
    ival = 0;                                                                     \
    GrGeomPatchLoop(i, j, k, fdir, PV_gr_domain, PV_patch_index,                  \
                    PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz,               \
    {                                                                             \
      body;                                                                       \
      ival++;                                                                     \
    });                                                                           \
  }

#define BCStructPatchLoopNoFdir(i, j, k, ival, bc_struct, ipatch, is,		\
                                locals, setup,                          \
                                f_left, f_right,                        \
                                f_down, f_up,                           \
                                f_back, f_front,                        \
                                finalize)                               \
  {                                                                     \
    GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);           \
    int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);         \
    Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);          \
                                                                        \
    int PV_r = SubgridRX(PV_subgrid);                                   \
    int PV_ix = SubgridIX(PV_subgrid);                                  \
    int PV_iy = SubgridIY(PV_subgrid);                                  \
    int PV_iz = SubgridIZ(PV_subgrid);                                  \
    int PV_nx = SubgridNX(PV_subgrid);                                  \
    int PV_ny = SubgridNY(PV_subgrid);                                  \
    int PV_nz = SubgridNZ(PV_subgrid);                                  \
                                                                        \
    ival = 0;                                                           \
    GrGeomPatchLoopNoFdir(i, j, k, PV_gr_domain, PV_patch_index,        \
                          PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz, \
                          locals, setup,                                \
                          f_left, f_right,                              \
                          f_down, f_up,                                 \
                          f_back, f_front,                              \
    {                                                                   \
      finalize;                                                         \
      ival++;                                                           \
    });                                                                 \
  }

#define BCStructPatchLoopOvrlndNoFdir(i, j, k, ival, bc_struct, ipatch, is, \
                                      locals, setup,                    \
                                      f_left, f_right,                  \
                                      f_down, f_up,                     \
                                      f_back, f_front,                  \
                                      finalize)                         \
  {                                                                     \
    GrGeomSolid  *PV_gr_domain = BCStructGrDomain(bc_struct);           \
    int PV_patch_index = BCStructPatchIndex(bc_struct, ipatch);         \
    Subgrid      *PV_subgrid = BCStructSubgrid(bc_struct, is);          \
                                                                        \
    int PV_r = SubgridRX(PV_subgrid);                                   \
    int PV_ix = SubgridIX(PV_subgrid) - 1;                              \
    int PV_iy = SubgridIY(PV_subgrid) - 1;                              \
    int PV_iz = SubgridIZ(PV_subgrid) - 1;                              \
    int PV_nx = SubgridNX(PV_subgrid) + 2;                              \
    int PV_ny = SubgridNY(PV_subgrid) + 2;                              \
    int PV_nz = SubgridNZ(PV_subgrid) + 2;                              \
                                                                        \
    ival = 0;                                                           \
    GrGeomPatchLoopNoFdir(i, j, k, PV_gr_domain, PV_patch_index,        \
                          PV_r, PV_ix, PV_iy, PV_iz, PV_nx, PV_ny, PV_nz, \
                          locals, setup,                                \
                          f_left, f_right,                              \
                          f_down, f_up,                                 \
                          f_back, f_front,                              \
    {                                                                   \
      finalize;                                                         \
      ival++;                                                           \
    });                                                                 \
  }

/*--------------------------------------------------------------------------
 * ForPatch loops and macros
 *--------------------------------------------------------------------------*/

/* Used to allow any BC type in a loop.
   Because all BC types are actually macro defines to integers,
   the compiler will optimize the necessary checks away.
*/
#define ALL -1
#define DoNothing

#define LeftFace GrGeomOctreeFaceL
#define RightFace GrGeomOctreeFaceR
#define DownFace GrGeomOctreeFaceD
#define UpFace GrGeomOctreeFaceU
#define BackFace GrGeomOctreeFaceB
#define FrontFace GrGeomOctreeFaceF

//#define CellSetup(...) DEFER3(_CellSetup)(__VA_ARGS__)
//#define _CellSetup(...) __VA_ARGS__
#define CellSetup(body) { body; };
#define CellFinalize(body) { body; };
#define BeforeAllCells(body) { body; };
#define AfterAllCells(body) { body; }
#define LoopVars(...) __VA_ARGS__

/* @MCB:
   Locals will pack everything into parens, which captures commas.
   This will treat it as a single parameter in other macros.
   It can then be exapnded at the correct place using UNPACK.
 */
#define Locals(...) (__VA_ARGS__)
#define NoLocals ()
#define UNPACK(locals) _UNPACK locals
#define _UNPACK(...) __VA_ARGS__ ;


#define FACE(fdir, body)      \
  case fdir:                  \
  {                           \
    body;                     \
    break;                    \
  }

#define _GetCurrentPatch(i, j, k, ival, bc_struct, ipatch, is) \
  BCStructBCType(bc_struct, ipatch)


#if 0
/* Template for copy+pasting for new BC loops */
ForPatchCellsPerFace(InsertBCTypeHere,
                     BeforeAllCells(DoNothing),
                     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                     NoLocals,
                     CellSetup({}),
                     FACE(LeftFace, {}),
                     FACE(RightFace, {}),
                     FACE(DownFace, {}),
                     FACE(UpFace, {}),
                     FACE(BackFace, {}),
                     FACE(FrontFace, {}),
                     CellFinalize({}),
                     AfterAllCells(DoNothing)
                     );
#endif

#define ForPatchCellsPerFace(bctype,													\
                             before_loop,											\
														 loopvars, locals,								\
                             setup,                           \
                             f_left, f_right,                 \
                             f_down, f_up,                    \
                             f_back, f_front,                 \
                             finalize,                        \
                             after_loop)                      \
  {                                                           \
    if ( ((bctype) == ALL) ||                                 \
         ((bctype) == _GetCurrentPatch(loopvars)))            \
    {                                                         \
      before_loop;                                            \
      BCStructPatchLoopNoFdir(loopvars,                       \
                              locals, setup,                  \
                              f_left, f_right,                \
                              f_down, f_up,                   \
                              f_back, f_front,                \
                              finalize);                      \
      after_loop;                                             \
    }                                                         \
  }

#define ForPatchCellsPerFaceWithGhost(bctype, \
                                      before_loop, loopvars,          \
																			locals,													\
                                      setup,                          \
                                      f_left, f_right,                \
                                      f_down, f_up,                   \
                                      f_back, f_front,                \
                                      finalize,                       \
                                      after_loop)                     \
  {                                                                   \
    if ( ((bctype) == ALL) ||                                         \
         ((bctype) == _GetCurrentPatch(loopvars)))                    \
    {                                                                 \
      before_loop;                                                    \
      BCStructPatchLoopOvrlndNoFdir(loopvars,                         \
																		locals,														\
                                    setup,                            \
                                    f_left, f_right,                  \
                                    f_down, f_up,                     \
                                    f_back, f_front,                  \
                                    finalize);                        \
      after_loop;                                                     \
    }                                                                 \
  }

/* Calls the loop directly
   Puts "body" into what is normally CellSetup
   Replaces locals, faces, and finalize with DoNothing
*/
#define ForEachPatchCell(i, j, k, ival, bc_struct, ipatch, is, body)		\
  BCStructPatchLoopNoFdir(i, j, k, ival, bc_struct, ipatch, is,					\
													NoLocals,																			\
													body,																					\
                          DoNothing, DoNothing,													\
                          DoNothing, DoNothing,													\
                          DoNothing, DoNothing,													\
                          DoNothing);

#endif
