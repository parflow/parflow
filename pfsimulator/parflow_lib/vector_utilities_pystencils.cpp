#include "parflow.h"

extern "C++" {

#include "KINSpgmrAtimesDQ.h"

void Pystencils_VProd(
/* Prod : z_i = x_i * y_i   */
        Vector *x,
        Vector *y,
        Vector *z) {

    Grid *grid = VectorGrid(x);
    Subgrid *subgrid;

    Subvector *x_sub;
    Subvector *y_sub;
    Subvector *z_sub;

    double *__restrict__ xp;
    double *__restrict__ yp;
    double *__restrict__ zp;

    int ix, iy, iz;
    int nx, ny, nz;

    int sg;

    grid = VectorGrid(x);
    ForSubgridI(sg, GridSubgrids(grid)) {
        subgrid = GridSubgrid(grid, sg);

        z_sub = VectorSubvector(z, sg);
        x_sub = VectorSubvector(x, sg);
        y_sub = VectorSubvector(y, sg);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        zp = SubvectorElt(z_sub, ix, iy, iz);
        xp = SubvectorElt(x_sub, ix, iy, iz);
        yp = SubvectorElt(y_sub, ix, iy, iz);

#ifdef PARFLOW_HAVE_PYSTENCILS
        PyCodegen_VProd(xp, yp, zp,
                        nx, ny, nz,      /* _size_x_0, _size_x_1, _size_x_2 */
                        1, nx, nx * ny,  /* _stride_x_0, _stride_x_1, _stride_x_2 */
                        1, nx, nx * ny,  /* _stride_y_0, _stride_y_1, _stride_y_2 */
                        1, nx, nx * ny); /* _stride_z_0, _stride_z_1, _stride_z_2 */
#endif
    }
}

void Pystencils_VLinearSum(
/* LinearSum : z = a * x + b * y              */
        double a,
        Vector *x,
        double b,
        Vector *y,
        Vector *z) {
    Grid *grid = VectorGrid(x);
    Subgrid *subgrid;

    Subvector *x_sub;
    Subvector *y_sub;
    Subvector *z_sub;

    double *__restrict__ xp;
    double *__restrict__ yp;
    double *__restrict__ zp;

    int ix, iy, iz;
    int nx, ny, nz;

    int sg;

    ForSubgridI(sg, GridSubgrids(grid)) {
        subgrid = GridSubgrid(grid, sg);

        z_sub = VectorSubvector(z, sg);
        x_sub = VectorSubvector(x, sg);
        y_sub = VectorSubvector(y, sg);

        ix = SubgridIX(subgrid);
        iy = SubgridIY(subgrid);
        iz = SubgridIZ(subgrid);

        nx = SubgridNX(subgrid);
        ny = SubgridNY(subgrid);
        nz = SubgridNZ(subgrid);

        zp = SubvectorElt(z_sub, ix, iy, iz);
        xp = SubvectorElt(x_sub, ix, iy, iz);
        yp = SubvectorElt(y_sub, ix, iy, iz);

#ifdef PARFLOW_HAVE_PYSTENCILS
        PyCodegen_VLinearSum(xp, yp, zp,
                             nx, ny, nz,      /* _size_x_0, _size_x_1, _size_x_2 */
                             1, nx, nx * ny, /* _stride_x_0, _stride_x_1, _stride_x_2 */
                             1, nx, nx * ny, /* _stride_y_0, _stride_y_1, _stride_y_2 */
                             1, nx, nx * ny, /* _stride_z_0, _stride_z_1, _stride_z_2 */
                             a, b);
#endif
    }
    IncFLOPCount(3 * VectorSize(z));
}
}

// TODO
double Pystencils_VDotProd(Vector *x, Vector *y) {
    return 0.;
}

// TODO
double Pystencils_VL1Norm(Vector *x) {
    return 0.;
}