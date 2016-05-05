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


/*
 * Functions
 */
void
parflow_p4est_sg_param_init(parflow_p4est_sg_param_t *sp);

void
parflow_p4est_sg_param_update(parflow_p4est_qiter_t * qiter,
                              parflow_p4est_sg_param_t *sp);

parflow_p4est_grid_t *parflow_p4est_grid_new(int Px, int Py, int Pz);

void            parflow_p4est_grid_destroy(parflow_p4est_grid_t * pfgrid);

void            parflow_p4est_grid_mesh_init(parflow_p4est_grid_t *
                                             pfgrid);

void            parflow_p4est_grid_mesh_destroy(parflow_p4est_grid_t *
                                                pfgrid);

parflow_p4est_qiter_t *parflow_p4est_qiter_init(parflow_p4est_grid_t *
                                                pfg,
                                                parflow_p4est_iter_type_t
                                                itype);

parflow_p4est_qiter_t *parflow_p4est_qiter_next(parflow_p4est_qiter_t *
                                                qiter);

void            parflow_p4est_qiter_qcorner(parflow_p4est_qiter_t * qiter,
                                            double v[3]);

parflow_p4est_quad_data_t
    * parflow_p4est_get_quad_data(parflow_p4est_qiter_t * qiter);

int             parflow_p4est_qiter_get_owner_rank(parflow_p4est_qiter_t *
                                                   qiter);

int             parflow_p4est_qiter_get_local_idx(parflow_p4est_qiter_t *
                                                   qiter);
parflow_p4est_ghost_data_t
   * parflow_p4est_get_ghost_data(parflow_p4est_grid_t * pfgrid,
                                  parflow_p4est_qiter_t * qiter);

SC_EXTERN_C_END;

#endif                          /* !PARLOW_P4EST_H */
