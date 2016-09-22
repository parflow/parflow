#ifndef PARLOW_P4EST_H
#define PARLOW_P4EST_H

#include <parflow.h>
#include <sc_containers.h>

SC_EXTERN_C_BEGIN;

typedef Subregion Subgrid;
typedef struct parflow_p4est_grid parflow_p4est_grid_t;
typedef struct parflow_p4est_qiter parflow_p4est_qiter_t;

typedef enum parflow_p4est_iter_type {
    PARFLOW_P4EST_QUAD = 0x01,
    PARFLOW_P4EST_GHOST = 0x02
} parflow_p4est_iter_type_t;

typedef struct parflow_p4est_quad_data {

    Subgrid *pf_subgrid;

} parflow_p4est_quad_data_t;

typedef parflow_p4est_quad_data_t parflow_p4est_ghost_data_t;

typedef struct parflow_p4est_sg_param{

  /** These values are set just once **/
  int       N[3];   /** Input number of grid points per coord. direction */
  int       P[3];   /** Computed number of subgrids per coord. direction */
  int       m[3];   /** Input number of subgrid points per coord. direction */
  int       l[3];   /** Residual N % m **/

  /** These values are to be updated
   ** when looping over p4est quadrants */
  int       p[3];         /** Computed number of subgrid points per coord. direction */
  int       icorner[3];   /** Bottom left corner in index space */

}parflow_p4est_sg_param_t;


/** Init parameter structure */
void
parflow_p4est_sg_param_init(parflow_p4est_sg_param_t *sp);

/** Update parameter structure */
void
parflow_p4est_sg_param_update(parflow_p4est_qiter_t * qiter,
                              parflow_p4est_sg_param_t *sp);

/** Create a parflow_p4est_grid structure.
 *  A globals structure must exist prior calling this function.
 *  TODO: Explain how Px, Py and Pz are computed.
 * \param [in] Px    Number of subgrids in the x coordinate direction.
 * \param [in] Py    Number of subgrids in the y coordinate direction.
 * \param [in] Pz    Number of subgrids in the z coordinate direction.
 *
 * \return Freshly allocated parflow_p4est_grid structure.
 */
parflow_p4est_grid_t *parflow_p4est_grid_new(int Px, int Py, int Pz);

/** Destroy a parflow_p4est_grid structure */
void            parflow_p4est_grid_destroy(parflow_p4est_grid_t * pfgrid);

void            parflow_p4est_grid_mesh_init(parflow_p4est_grid_t *
                                             pfgrid);

void            parflow_p4est_grid_mesh_destroy(parflow_p4est_grid_t *
                                                pfgrid);

void            parflow_p4est_get_zneigh(Subgrid * subgrid,
                                         parflow_p4est_qiter_t * qiter,
                                         parflow_p4est_grid_t * pfgrid);

/** Create an iterator over all local or ghost quadrants.
 * \param [in] pfg      Pointer to a valid parflow_p4est_grid structure.
 * \param [in] itype    Flag determining the type of iterator.
 *
 * \return Complete internal state of iterator.
 */
parflow_p4est_qiter_t *parflow_p4est_qiter_init(parflow_p4est_grid_t *
                                                pfg,
                                                parflow_p4est_iter_type_t
                                                itype);
/** Advance the iterator.
 *  \param[in] qiter   Valid interator.
 *
 *  \return After procesing one quadrant, return an interator with
 *          information of the next quadrant. (resp. ghost).
 *          If this has been the last quadrant return NULL.
 */
parflow_p4est_qiter_t *parflow_p4est_qiter_next(parflow_p4est_qiter_t *
                                                qiter);

/** Get coorner of a quadrant in the brick coordinate system
 * \param [in] pfg      Pointer to a valid iterator structure.
 * \param [out] v       Coordinates of the quadrant in the passed iterator.
 */
void            parflow_p4est_qiter_qcorner(parflow_p4est_qiter_t * qiter,
                                            double v[3]);

/** Retrieve a pointer to the information placed on each quadrant */
parflow_p4est_quad_data_t
    * parflow_p4est_get_quad_data(parflow_p4est_qiter_t * qiter);

/** Get owner rank of the quadrant in the passed iterator */
int             parflow_p4est_qiter_get_owner_rank(parflow_p4est_qiter_t *
                                                   qiter);

/** Get index in the owner processor of quadrant in the passed iterator */
int             parflow_p4est_qiter_get_local_idx(parflow_p4est_qiter_t *
                                                   qiter);

/** Get owner tree of the quadrant in the passed iterator */
int             parflow_p4est_qiter_get_tree(parflow_p4est_qiter_t * qiter);

/** Get index of quadrant in the owner tree in passed iterator */
int             parflow_p4est_qiter_get_idx_in_tree(parflow_p4est_qiter_t * qiter);

/** Retrieve a pointer to the information placed on each ghost quadrant */
parflow_p4est_ghost_data_t
   * parflow_p4est_get_ghost_data(parflow_p4est_grid_t * pfgrid,
                                  parflow_p4est_qiter_t * qiter);

/** Returns 1 if there are no trees in this processor, 0 otherwise */
int             parflow_p4est_rank_is_empty(parflow_p4est_grid_t * pfgrid);

/** Get owner of the projection of this subgrid in a target xy plane.
 * \param [in] subgrid       Subgrid to be projected.
 * \param [in] z_level       Height of the xy plane we whish to project to.
 * \param [in] pfgrid        Pointer to a valid parflow_p4est_grid structure.
 * \param [in, out] info     An array of two integers. The first one is the
 *                           owner rank of the subgrid coordinates (x,y,z_level)
 *                           where (x,y) are taken from the pased subgrid. The
 *                           second one is the morton code of the
 *                           projection, it can be used as a tag to
 *                           distiguish columns in mpi communication.
 */
void
parflow_p4est_get_projection_info (Subgrid *subgrid, int z_level,
                                   parflow_p4est_grid_t *pfgrid, int info[2]);

/** Get array with number of subgrids per rank.
 * \param [in] pfgrid        Pointer to a valid parflow_p4est_grid structure.
 * \param [in, out] quads_per_rank
 *                           Pointer to a int array of mpisize elements.
 */
void             parflow_p4est_nquads_per_rank(parflow_p4est_grid_t *pfgrid,
                                               int *quads_per_rank);
SC_EXTERN_C_END;

#endif                          /* !PARLOW_P4EST_H */
