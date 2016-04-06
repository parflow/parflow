#ifndef PARLOW_P4EST_H
#define PARLOW_P4EST_H


typedef struct parflow_p4est_grid parflow_p4est_grid_t;

/*Accesor macros*/
#define PARFLOW_P4EST_GET_GRID_DIM(pfgrid) ((pfgrid)->dim)

/*Functions*/
parflow_p4est_grid_t *parflow_p4est_grid_new (int Px, int Py, int Pz);


void                parflow_p4est_grid_destroy (parflow_p4est_grid_t *
                                                pfgrid);
#if 0
void                parflow_p4est_qcoord_to_vertex (parflow_p4est_grid_t *
                                                    pfgrid,
                                                    p4est_topidx_t treeid,
                                                    p4est_quadrant_t * quad,
                                                    double v[3]);
p4est_topidx_t
parflow_p4est_gquad_owner_tree (p4est_quadrant_t   *quad);
#endif

#endif // !PARLOW_P4EST_H
