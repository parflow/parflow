typedef PFModule * (*NewDefault)(void);

typedef void (*AdvectionConcentrationInvoke) (ProblemData *problem_data, int phase, int concentration, Vector *old_concentration, Vector *new_concentration, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, Vector *solid_mass_factor, double time, double deltat, int order);
typedef PFModule *(*AdvectionConcentrationInitInstanceXtraType) (Problem *problem, Grid *grid, double *temp_data);

/* advection_godunov.c */
void Godunov(ProblemData *problem_data, int phase, int concentration, Vector *old_concentration, Vector *new_concentration, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, Vector *solid_mass_factor, double time, double deltat, int order);
PFModule *GodunovInitInstanceXtra(Problem *problem, Grid *grid, double *temp_data);
void GodunovFreeInstanceXtra(void);
PFModule *GodunovNewPublicXtra(void);
void GodunovFreePublicXtra(void);
int GodunovSizeOfTempData(void);

/* axpy.c */
void Axpy(double alpha, Vector *x, Vector *y);

/* background.c */
Background *ReadBackground(void);
void FreeBackground(Background *background);
void SetBackgroundBounds(Background *background, Grid *grid);

/* bc_lb.c */
void LBInitializeBC(Lattice *lattice, Problem *problem, ProblemData *problem_data);

/* bc_pressure.c */
BCPressureData *NewBCPressureData(void);
void FreeBCPressureData(BCPressureData *bc_pressure_data);
void PrintBCPressureData(BCPressureData *bc_pressure_data);

typedef void (*BCPressurePackageInvoke) (ProblemData *problem_data);
typedef PFModule *(*BCPressurePackageInitInstanceXtraInvoke) (Problem *problem);
typedef PFModule *(*BCPressurePackageNewPublicXtraInvoke) (int num_phases);

/* bc_pressure_package.c */
void BCPressurePackage(ProblemData *problem_data);
PFModule *BCPressurePackageInitInstanceXtra(Problem *problem);
void BCPressurePackageFreeInstanceXtra(void);
PFModule *BCPressurePackageNewPublicXtra(int num_phases);
void BCPressurePackageFreePublicXtra(void);
int BCPressurePackageSizeOfTempData(void);

/* calc_elevations.c */
double **CalcElevations(GeomSolid *geom_solid, int ref_patch, SubgridArray *subgrids, ProblemData  *problem_data);

typedef void (*LinearSolverInvoke) (Vector *x, Vector *b, double tol, int zero);
typedef PFModule *(*LinearSolverInitInstanceXtraInvoke) (Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
typedef PFModule *(*LinearSolverNewPublicXtraInvoke) (char *name);

/* cghs.c */
void CGHS(Vector *x, Vector *b, double tol, int zero);
PFModule *CGHSInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
void CGHSFreeInstanceXtra(void);
PFModule *CGHSNewPublicXtra(char *name);
void CGHSFreePublicXtra(void);
int CGHSSizeOfTempData(void);

/* char_vector.c */
CommPkg *NewCharVectorUpdatePkg(CharVector *charvector, int update_mode);
CommHandle *InitCharVectorUpdate(CharVector *charvector, int update_mode);
void FinalizeCharVectorUpdate(CommHandle *handle);
CharVector *NewTempCharVector(Grid *grid, int nc, int num_ghost);
void SetTempCharVectorData(CharVector *charvector, char *data);
CharVector *NewCharVector(Grid *grid, int nc, int num_ghost);
void FreeTempCharVector(CharVector *charvector);
void FreeCharVector(CharVector *charvector);
void InitCharVector(CharVector *v, char value);
void InitCharVectorAll(CharVector *v, char value);
void InitCharVectorInc(CharVector *v, char value, int inc);

typedef void (*ChebyshevInvoke) (Vector *x, Vector *b, double tol, int zero, double ia, double ib, int num_iter);
typedef PFModule *(*ChebyshevInitInstanceXtraInvoke) (Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
typedef PFModule *(*ChebyshevNewPublicXtraInvoke) (char *name);

/* chebyshev.c */
void Chebyshev(Vector *x, Vector *b, double tol, int zero, double ia, double ib, int num_iter);
PFModule *ChebyshevInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
void ChebyshevFreeInstanceXtra(void);
PFModule *ChebyshevNewPublicXtra(char *name);
void ChebyshevFreePublicXtra(void);
int ChebyshevSizeOfTempData(void);

/* comm_pkg.c */
void ProjectRegion(Region *region, int sx, int sy, int sz, int ix, int iy, int iz);
Region *ProjectRBPoint(Region *region, int rb[4 ][3 ]);
void CreateComputePkgs(Grid *grid);
void FreeComputePkgs(Grid *grid);

/* communication.c */
int NewCommPkgInfo(Subregion *data_sr, Subregion *comm_sr, int index, int num_vars, int *loop_array);
CommPkg *NewCommPkg(Region *send_region, Region *recv_region, SubregionArray *data_space, int num_vars, double *data);
void FreeCommPkg(CommPkg *pkg);
// SGS what's up with this?
CommHandle *InitCommunication(CommPkg *comm_pkg);
void FinalizeCommunication(CommHandle *handle);

/* computation.c */
ComputePkg *NewComputePkg(Region *send_reg, Region *recv_reg, Region *dep_reg, Region *ind_reg);
void FreeComputePkg(ComputePkg *compute_pkg);

/* compute_maximums.c */
double ComputePhaseMaximum(double phase_u_max, double dx, double phase_v_max, double dy, double phase_w_max, double dz);
double ComputeTotalMaximum(Problem *problem, EvalStruct *eval_struct, double s_lower, double s_upper, double total_u_max, double dx, double total_v_max, double dy, double total_w_max, double beta_max, double dz);

/* compute_total_concentration.c */
double ComputeTotalConcen(GrGeomSolid *gr_domain, Grid *grid, Vector *substance);

typedef void (*ConstantRFInvoke) (GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata);
typedef PFModule *(*ConstantRFInitInstanceXtraInvoke) (Grid *grid, double *temp_data);

/* constantRF.c */
void ConstantRF(GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata);
PFModule *ConstantRFInitInstanceXtra(Grid *grid, double *temp_data);
void ConstantRFFreeInstanceXtra(void);
PFModule *ConstantRFNewPublicXtra(char *geom_name);
void ConstantRFFreePublicXtra(void);
int ConstantRFSizeOfTempData(void);

typedef void (*PorosityFieldInvoke) (GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field);
typedef PFModule *(*PorosityFieldInitInstanceXtraInvoke) (Grid *grid, double *temp_data);
typedef PFModule *(*PorosityFieldNewPublicXtraInvoke) (char *geom_name);

/* constant_porosity.c */
void ConstantPorosity(GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field);
PFModule *ConstantPorosityInitInstanceXtra(Grid *grid, double *temp_data);
void ConstantPorosityFreeInstanceXtra(void);
PFModule *ConstantPorosityNewPublicXtra(char *geom_name);
void ConstantPorosityFreePublicXtra(void);
int ConstantPorositySizeOfTempData(void);

/* copy.c */
void Copy(Vector *x, Vector *y);

/* create_grid.c */
SubgridArray *GetGridSubgrids(SubgridArray *all_subgrids);
Grid *CreateGrid(Grid *user_grid);

typedef void (*DiagScaleInvoke) (Vector *x, Matrix *A, Vector *b, Vector *d);

/* diag_scale.c */
void DiagScale(Vector *x, Matrix *A, Vector *b, Vector *d);

/* diffuse_lb.c */
void DiffuseLB(Lattice *lattice, Problem *problem, int max_iterations, char *file_prefix);
void LatticeFlowInit(Lattice *lattice, Problem *problem);
double MaxVectorValue(Vector *field);
double MaxVectorDividend(Vector *field1, Vector *field2);

typedef void (*DiscretizePressureInvoke) (Matrix **ptr_to_A, Vector **ptr_to_f, ProblemData *problem_data, double time, Vector *total_mobility_x, Vector *total_mobility_y, Vector *total_mobility_z, Vector **phase_saturations);
typedef PFModule *(*DiscretizePressureInitInstanceXtraInvoke) (Problem *problem, Grid *grid, double *temp_data);
/* discretize_pressure.c */
void DiscretizePressure(Matrix **ptr_to_A, Vector **ptr_to_f, ProblemData *problem_data, double time, Vector *total_mobility_x, Vector *total_mobility_y, Vector *total_mobility_z, Vector **phase_saturations);

PFModule *DiscretizePressureInitInstanceXtra(Problem *problem, Grid *grid, double *temp_data);
void DiscretizePressureFreeInstanceXtra(void);
PFModule *DiscretizePressureNewPublicXtra(void);
void DiscretizePressureFreePublicXtra(void);
int DiscretizePressureSizeOfTempData(void);

/* distribute_usergrid.c */
SubgridArray *DistributeUserGrid(Grid *user_grid);

/* dpofa.c */
int dpofa_(double *a, int *lda, int *n, int *info);
double ddot_(int *n, double *dx, int *incx, double *dy, int *incy);

/* dposl.c */
int dposl_(double *a, int *lda, int *n, double *b);
int daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);

/* gauinv.c */
int gauinv_(double *p, double *xp, int *ierr);

/* general.c */
char *malloc_chk(int size, char *file, int line);
char *calloc_chk(int count, int elt_size, char *file, int line);
int Exp2(int p);
void printMemoryInfo(FILE *log_file);
void recordMemoryInfo();
void printMaxMemory(FILE *log_file);

/* geom_t_solid.c */
GeomTSolid *GeomNewTSolid(GeomTIN *surface, int **patches, int num_patches, int *num_patch_triangles);
void GeomFreeTSolid(GeomTSolid *solid);
int GeomReadTSolids(GeomTSolid ***solids_data_ptr, char *geom_input_name);
GeomTSolid *GeomTSolidFromBox(double xl, double yl, double zl, double xu, double yu, double zu);

/* geometry.c */
GeomVertexArray *GeomNewVertexArray(GeomVertex **vertices, int nV);
void GeomFreeVertexArray(GeomVertexArray *vertex_array);
GeomTIN *GeomNewTIN(GeomVertexArray *vertex_array, GeomTriangle **triangles, int nT);
void GeomFreeTIN(GeomTIN *surface);
GeomSolid *GeomNewSolid(void *data, int type);
void GeomFreeSolid(GeomSolid *solid);
int GeomReadSolids(GeomSolid ***solids_ptr, char *geom_input_name, int type);
GeomSolid *GeomSolidFromBox(double xl, double yl, double zl, double xu, double yu, double zu, int type);
void IntersectLineWithTriangle(unsigned int line_direction, double coord_0, double coord_1, double v0_x, double v0_y, double v0_z, double v1_x, double v1_y, double v1_z, double v2_x, double v2_y, double v2_z, int *intersects, double *point, int *normal_component);

/* globals.c */
void NewGlobals(char *run_name);
void FreeGlobals(void);
void LogGlobals(void);

/* grgeom_list.c */
ListMember *NewListMember(double value, int normal_component, int triangle_id);
void FreeListMember(ListMember *member);
void ListInsert(ListMember **head, ListMember *member);
int ListDelete(ListMember **head, ListMember *member);
ListMember *ListSearch(ListMember *head, double value, int normal_component, int triangle_id);
ListMember *ListValueSearch(ListMember *head, double value);
ListMember *ListValueNormalComponentSearch(ListMember *head, double value, int normal_component);
ListMember *ListTriangleIDSearch(ListMember *head, int triangle_id);
void ListFree(ListMember **head);
int ListLength(ListMember *head);
void ListPrint(ListMember *head);

/* grgeom_octree.c */
int GrGeomCheckOctree(GrGeomOctree *grgeom_octree);
void GrGeomFixOctree(GrGeomOctree *grgeom_octree, GrGeomOctree **patch_octrees, int num_patches, int level, int num_indices);
GrGeomOctree *GrGeomNewOctree(void);
void GrGeomNewOctreeChildren(GrGeomOctree *grgeom_octree);
void GrGeomFreeOctree(GrGeomOctree *grgeom_octree);
GrGeomOctree *GrGeomOctreeFind(int *new_level, GrGeomOctree *grgeom_octree_root, int ix, int iy, int iz, int level);
GrGeomOctree *GrGeomOctreeAddCell(GrGeomOctree *grgeom_octree_root, unsigned int cell, int ix, int iy, int iz, int level);
GrGeomOctree *GrGeomOctreeAddFace(GrGeomOctree *grgeom_octree_root, int line_direction, int cell_index0, int cell_index1, int face_index, int extent_lower, int extent_upper, int level, int normal_in_direction);
void GrGeomOctreeFromTIN(GrGeomOctree **solid_octree_ptr, GrGeomOctree ***patch_octrees_ptr, GeomTIN *solid, int **patches, int num_patches, int *num_patch_triangles, GrGeomExtentArray *extent_array, double xlower, double ylower, double zlower, double xupper, double yupper, double zupper, int min_level, int max_level);
void GrGeomOctreeFromInd(GrGeomOctree **solid_octree_ptr, Vector *indicator_field, int indicator, double xlower, double ylower, double zlower, double xupper, double yupper, double zupper, int octree_bg_level, int octree_ix, int octree_iy, int octree_iz);
void GrGeomPrintOctreeStruc(amps_File file, GrGeomOctree *grgeom_octree);
int GrGeomPrintOctreeLevel(amps_File file, GrGeomOctree *grgeom_octree, int level, int current_level);
void GrGeomPrintOctree(char *filename, GrGeomOctree *grgeom_octree_root);
void GrGeomPrintOctreeCells(char *filename, GrGeomOctree *octree, int last_level);
void GrGeomOctreeFree(GrGeomOctree *grgeom_octree_root);

/* grgeometry.c */
int GrGeomGetOctreeInfo(double *xlp, double *ylp, double *zlp, double *xup, double *yup, double *zup, int *ixp, int *iyp, int *izp);
GrGeomExtentArray *GrGeomNewExtentArray(GrGeomExtents *extents, int size);
void GrGeomFreeExtentArray(GrGeomExtentArray *extent_array);
GrGeomExtentArray *GrGeomCreateExtentArray(SubgridArray *subgrids, int xl_ghost, int xu_ghost, int yl_ghost, int yu_ghost, int zl_ghost, int zu_ghost);
GrGeomSolid *GrGeomNewSolid(GrGeomOctree *data, GrGeomOctree **patches, int num_patches, int octree_bg_level, int octree_ix, int octree_iy, int octree_iz);
void GrGeomFreeSolid(GrGeomSolid *solid);
void GrGeomSolidFromInd(GrGeomSolid **solid_ptr, Vector *indicator_field, int indicator);
void GrGeomSolidFromGeom(GrGeomSolid **solid_ptr, GeomSolid *geom_solid, GrGeomExtentArray *extent_array);

/* grid.c */
Grid *NewGrid(SubgridArray *subgrids, SubgridArray *all_subgrids);
void FreeGrid(Grid *grid);
int ProjectSubgrid(Subgrid *subgrid, int sx, int sy, int sz, int ix, int iy, int iz);
Subgrid *ConvertToSubgrid(Subregion *subregion);
Subgrid *ExtractSubgrid(int rx, int ry, int rz, Subgrid *subgrid);
Subgrid *IntersectSubgrids(Subgrid *subgrid1, Subgrid *subgrid2);
SubgridArray *SubtractSubgrids(Subgrid *subgrid1, Subgrid *subgrid2);
SubgridArray *UnionSubgridArray(SubgridArray *subgrids);

/* hbt.c */
HBT *HBT_new(
             int (*compare_method)(void *, void *),
             void (*free_method)(void *),
             void (*printf_method)(FILE *, void *),
             int (*scanf_method)(FILE *, void **),
             int malloc_flag);
HBT_element *_new_HBT_element(HBT *tree, void *object, int sizeof_obj);
void _free_HBT_element(HBT *tree, HBT_element *el);
void _HBT_free(HBT *tree, HBT_element *subtree);
void HBT_free(HBT *tree);
void *HBT_lookup(HBT *tree, void *obj);
void *HBT_replace(HBT *tree, void *obj, int sizeof_obj);
int HBT_insert(HBT *tree, void *obj, int sizeof_obj);
void *HBT_delete(HBT *tree, void *obj);
void *HBT_successor(HBT *tree, void *obj);

void HBT_printf(FILE *file, HBT *tree);
void HBT_scanf(FILE *file, HBT *tree);

/* infinity_norm.c */
double InfinityNorm(Vector *x);

/* innerprod.c */
double InnerProd(Vector *x, Vector *y);

/* input_porosity.c */
void InputPorosity(GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field);
PFModule *InputPorosityInitInstanceXtra(Grid *grid, double *temp_data);
void InputPorosityFreeInstanceXtra(void);
PFModule *InputPorosityNewPublicXtra(char *geom_name);
void InputPorosityFreePublicXtra(void);
int InputPorositySizeOfTempData(void);


/* inputRF.c */
void InputRF(GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata);
PFModule *InputRFInitInstanceXtra(Grid *grid, double *temp_data);
void InputRFFreeInstanceXtra(void);
PFModule *InputRFNewPublicXtra(char *geom_name);
void InputRFFreePublicXtra(void);
int InputRFSizeOfTempData(void);

typedef int (*NonlinSolverInvoke) (Vector *pressure, Vector *density, Vector *old_density, Vector *saturation, Vector *old_saturation, double t, double dt, ProblemData *problem_data, Vector *old_pressure, Vector *evap_trans, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity);
typedef PFModule *(*NonlinSolverInitInstanceXtraInvoke) (Problem *problem, Grid *grid, ProblemData *problem_data, double *temp_data);

/* kinsol_nonlin_solver.c */
int KINSolInitPC(int neq, N_Vector pressure, N_Vector uscale, N_Vector fval, N_Vector fscale, N_Vector vtemp1, N_Vector vtemp2, void *nl_function, double uround, long int *nfePtr, void *current_state);
int KINSolCallPC(int neq, N_Vector pressure, N_Vector uscale, N_Vector fval, N_Vector fscale, N_Vector vtem, N_Vector ftem, void *nl_function, double uround, long int *nfePtr, void *current_state);
void PrintFinalStats(FILE *out_file, long int *integer_outputs_now, long int *integer_outputs_total);
int KinsolNonlinSolver(Vector *pressure, Vector *density, Vector *old_density, Vector *saturation, Vector *old_saturation, double t, double dt, ProblemData *problem_data, Vector *old_pressure, Vector *evap_trans, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity);
PFModule *KinsolNonlinSolverInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, double *temp_data);
void KinsolNonlinSolverFreeInstanceXtra(void);
PFModule *KinsolNonlinSolverNewPublicXtra(void);
void KinsolNonlinSolverFreePublicXtra(void);
int KinsolNonlinSolverSizeOfTempData(void);

typedef void (*KinsolPCInvoke) (Vector *rhs);
typedef PFModule * (*KinsolPCInitInstanceXtraInvoke) (Problem *problem, Grid *grid, ProblemData *problem_data, double *temp_data, Vector *pressure, Vector *old_pressure, Vector *saturation, Vector *density, double dt, double time);
typedef PFModule *(*KinsolPCNewPublicXtraInvoke) (char *name, char *pc_name);

/* kinsol_pc.c */
void KinsolPC(Vector *rhs);
PFModule *KinsolPCInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, double *temp_data, Vector *pressure, Vector *old_pressure, Vector *saturation, Vector *density, double dt, double time);
void KinsolPCFreeInstanceXtra(void);
PFModule *KinsolPCNewPublicXtra(char *name, char *pc_name);
void KinsolPCFreePublicXtra(void);
int KinsolPCSizeOfTempData(void);


typedef void (*L2ErrorNormInvoke) (double time, Vector *pressure, ProblemData *problem_data, double *l2_error_norm);
/* kokkos.cpp */
void kokkosInit();
void kokkosFinalize();

/* l2_error_norm.c */
void L2ErrorNorm(double time, Vector *pressure, ProblemData *problem_data, double *l2_error_norm);
PFModule *L2ErrorNormInitInstanceXtra(void);
void L2ErrorNormFreeInstanceXtra(void);
PFModule *L2ErrorNormNewPublicXtra(void);
void L2ErrorNormFreePublicXtra(void);
int L2ErrorNormSizeOfTempData(void);

/* line_process.c */
void LineProc(double *Z, double phi, double theta, double dzeta, int izeta, int nzeta, double Kmax, double dK);

/* logging.c */
void NewLogging(void);
void FreeLogging(void);
FILE *OpenLogFile(char *module_name);
int CloseLogFile(FILE *log_file);
void PrintVersionInfo(FILE *log_file);


typedef void (*MatrixDiagScaleInvoke) (Vector *x, Matrix *A, Vector *b, int flag);
typedef PFModule *(*MatrixDiagScaleInitInstanceXtraInvoke) (Grid *grid);
typedef PFModule *(*MatrixDiagScaleNewPublicXtraInvoke) (char *name);

/* matdiag_scale.c */
void MatDiagScale(Vector *x, Matrix *A, Vector *b, int flag);
PFModule *MatDiagScaleInitInstanceXtra(Grid *grid);
void MatDiagScaleFreeInstanceXtra(void);
PFModule *MatDiagScaleNewPublicXtra(char *name);
void MatDiagScaleFreePublicXtra(void);
int MatDiagScaleSizeOfTempData(void);

/* matrix.c */
Stencil *NewStencil(int shape [][3 ], int sz);
CommPkg *NewMatrixUpdatePkg(Matrix *matrix, Stencil *ghost);
CommHandle *InitMatrixUpdate(Matrix *matrix);
void FinalizeMatrixUpdate(CommHandle *handle);
Matrix *NewMatrix(Grid *grid, SubregionArray *range, Stencil *stencil, int symmetry, Stencil *ghost);
Matrix *NewMatrixType(Grid *grid, SubregionArray *range, Stencil *stencil, int symmetry, Stencil *ghost, enum matrix_type type);
void FreeStencil(Stencil *stencil);
void FreeMatrix(Matrix *matrix);
void InitMatrix(Matrix *A, double value);

/* matvec.c */
void Matvec(double alpha, Matrix *A, Vector *x, double beta, Vector *y);

/* matvecSubMat.c */
void MatvecSubMat(void *  current_state,
                  double  alpha,
                  Matrix *JB,
                  Matrix *JC,
                  Vector *x,
                  double  beta,
                  Vector *y);

/* MatvecJacF */
void            MatvecJacF(
                           ProblemData *problem_data,
                           double       alpha,
                           Matrix *     JF,
                           Vector *     x,
                           double       beta,
                           Vector *     y);

/* MatvecJacE */
void            MatvecJacE(
                           ProblemData *problem_data,
                           double       alpha,
                           Matrix *     JE,
                           Vector *     x,
                           double       beta,
                           Vector *     y);

/* max_field_value.c */
double MaxFieldValue(Vector *field, Vector *phi, int dir);
double MaxPhaseFieldValue(Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, Vector *phi);
double MaxTotalFieldValue(Problem *problem, EvalStruct *eval_struct, Vector *saturation, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, Vector *beta, Vector *phi);


typedef void (*PrecondInvoke) (Vector *x, Vector *b, double tol, int zero);
typedef PFModule * (*PrecondInitInstanceXtraInvoke) (Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, Matrix *B, double *temp_data);
typedef PFModule *(*PrecondNewPublicXtra) (char *name);

/* mg_semi.c */
void MGSemi(Vector *x, Vector *b, double tol, int zero);
void SetupCoarseOps(Matrix **A_l, Matrix **P_l, int num_levels, SubregionArray **f_sra_l, SubregionArray **c_sra_l);
PFModule *MGSemiInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
void MGSemiFreeInstanceXtra(void);
PFModule *MGSemiNewPublicXtra(char *name);
void MGSemiFreePublicXtra(void);
int MGSemiSizeOfTempData(void);

/* mg_semi_prolong.c */
void MGSemiProlong(Matrix *A_f, Vector *e_f, Vector *e_c, Matrix *P, SubregionArray *f_sr_array, SubregionArray *c_sr_array, ComputePkg *compute_pkg, CommPkg *e_f_comm_pkg);
ComputePkg *NewMGSemiProlongComputePkg(Grid *grid, Stencil *stencil, int sx, int sy, int sz, int c_index, int f_index);

/* mg_semi_restrict.c */
void MGSemiRestrict(Matrix *A_f, Vector *r_f, Vector *r_c, Matrix *P, SubregionArray *f_sr_array, SubregionArray *c_sr_array, ComputePkg *compute_pkg, CommPkg *r_f_comm_pkg);
ComputePkg *NewMGSemiRestrictComputePkg(Grid *grid, Stencil *stencil, int sx, int sy, int sz, int c_index, int f_index);

/* n_vector.c */
void SetPf2KinsolData(Grid *grid, int num_ghost);
void N_VPrint(N_Vector x);
void FreeTempVector(Vector *vector);

/* Kinsol API is in C. */
#ifdef __cplusplus
extern "C" {
#endif

N_Vector N_VNew(int N, void *machEnv);
void N_VFree(N_Vector x);

#ifdef __cplusplus
}
#endif

/* new_endpts.c */
void NewEndpts(double *alpha, double *beta, double *pp, int *size_ptr, int n, double *a_ptr, double *b_ptr, double *cond_ptr, double ereps);

typedef void (*NlFunctionEvalInvoke) (Vector *pressure, Vector *fval, ProblemData *problem_data, Vector *saturation, Vector *old_saturation, Vector *density, Vector *old_density, double dt, double time, Vector *old_pressure, Vector *evap_trans, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity);
typedef PFModule *(*NlFunctionEvalInitInstanceXtraInvoke) (Problem *problem, Grid *grid, double *temp_data);

/* nl_function_eval.c */
void KINSolFunctionEval(int size, N_Vector pressure, N_Vector fval, void *current_state);
void NlFunctionEval(Vector *pressure, Vector *fval, ProblemData *problem_data, Vector *saturation, Vector *old_saturation, Vector *density, Vector *old_density, double dt, double time, Vector *old_pressure, Vector *evap_trans, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity);
PFModule *NlFunctionEvalInitInstanceXtra(Problem *problem, Grid *grid, double *temp_data);
void NlFunctionEvalFreeInstanceXtra(void);
PFModule *NlFunctionEvalNewPublicXtra(char *name);

void NlFunctionEvalFreePublicXtra(void);
int NlFunctionEvalSizeOfTempData(void);

/* nodiag_scale.c */
void NoDiagScale(Vector *x, Matrix *A, Vector *b, int flag);
PFModule *NoDiagScaleInitInstanceXtra(Grid *grid);
void NoDiagScaleFreeInstanceXtra(void);
PFModule *NoDiagScaleNewPublicXtra(char *name);
void NoDiagScaleFreePublicXtra(void);
int NoDiagScaleSizeOfTempData(void);

/* pcg.c */
void PCG(Vector *x, Vector *b, double tol, int zero);
PFModule *PCGInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, Matrix *C, double *temp_data);
void PCGFreeInstanceXtra(void);
PFModule *PCGNewPublicXtra(char *name);
void PCGFreePublicXtra(void);
int PCGSizeOfTempData(void);

typedef void (*PermeabilityFaceInvoke) (Vector *zperm, Vector *permeability);
typedef PFModule *(*PermeabilityFaceInitInstanceXtraInvoke) (Grid *z_grid);

/* permeability_face.c */
void PermeabilityFace(Vector *zperm, Vector *permeability);
PFModule *PermeabilityFaceInitInstanceXtra(Grid *z_grid);
void PermeabilityFaceFreeInstanceXtra(void);
PFModule *PermeabilityFaceNewPublicXtra(void);
void PermeabilityFaceFreePublicXtra(void);
int PermeabilityFaceSizeOfTempData(void);

/* perturb_lb.c */
void PerturbSystem(Lattice *lattice, Problem *problem);

/* pf_module.c */
PFModule *NewPFModule(void *call, void *init_instance_xtra, void *free_instance_xtra, void *new_public_xtra, void *free_public_xtra, void *sizeof_temp_data, void *instance_xtra, void *public_xtra);
PFModule *NewPFModuleExtended(void *call, void *init_instance_xtra, void *free_instance_xtra, void *new_public_xtra, void *free_public_xtra, void *sizeof_temp_data, void *output, void *output_static, void *instance_xtra, void *public_xtra);
PFModule *DupPFModule(PFModule *pf_module);
void FreePFModule(PFModule *pf_module);

/* pf_pfmg.c */
void PFMG(Vector *soln, Vector *rhs, double tol, int zero);
//PFModule *PFMGInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *pf_Bmat , Matrix *pf_Cmat, double *temp_data );
PFModule *PFMGInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *pf_Bmat, Matrix *pf_Cmat, double *temp_data);
void PFMGFreeInstanceXtra(void);
PFModule *PFMGNewPublicXtra(char *name);
void PFMGFreePublicXtra(void);
int PFMGSizeOfTempData(void);

/* pf_pfmg_octree.c */

void PFMGOctree(Vector *soln, Vector *rhs, double tol, int zero);
PFModule  *PFMGOctreeInitInstanceXtra(
                                      Problem *    problem,
                                      Grid *       grid,
                                      ProblemData *problem_data,
                                      Matrix *     pf_Bmat,
                                      Matrix *     pf_Cmat,
                                      double *     temp_data);
void PFMGOctreeFreeInstanceXtra(void);
PFModule *PFMGOctreeNewPublicXtra(char *name);
void PFMGOctreeFreePublicXtra(void);
int PFMGOctreeSizeOfTempData(void);

/* pf_smg.c */
void SMG(Vector *soln, Vector *rhs, double tol, int zero);
PFModule  *SMGInitInstanceXtra(
                               Problem *    problem,
                               Grid *       grid,
                               ProblemData *problem_data,
                               Matrix *     pf_Bmat,
                               Matrix *     pf_Cmat,
                               double *     temp_data);
void SMGFreeInstanceXtra(void);
PFModule *SMGNewPublicXtra(char *name);
void SMGFreePublicXtra(void);
int SMGSizeOfTempData(void);

/* pfield.c */
void PField(Grid *grid, GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata, Statistics *stats);

/* pgsRF.c */
void PGSRF(GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata);
PFModule *PGSRFInitInstanceXtra(Grid *grid, double *temp_data);
void PGSRFFreeInstanceXtra(void);
PFModule *PGSRFNewPublicXtra(char *geom_name);
void PGSRFFreePublicXtra(void);
int PGSRFSizeOfTempData(void);

typedef void (*PhaseVelocityFaceInvoke) (Vector *xvel, Vector *yvel, Vector *zvel, ProblemData *problem_data, Vector *pressure, Vector **saturations, int phase, double time);
typedef PFModule *(*PhaseVelocityFaceInitInstanceXtraInvoke) (Problem *problem, Grid *grid, Grid *x_grid, Grid *y_grid, Grid *z_grid, double *temp_data);

/* phase_velocity_face.c */
void PhaseVelocityFace(Vector *xvel, Vector *yvel, Vector *zvel, ProblemData *problem_data, Vector *pressure, Vector **saturations, int phase, double time);
PFModule *PhaseVelocityFaceInitInstanceXtra(Problem *problem, Grid *grid, Grid *x_grid, Grid *y_grid, Grid *z_grid, double *temp_data);
void PhaseVelocityFaceFreeInstanceXtra(void);
PFModule *PhaseVelocityFaceNewPublicXtra(void);
void PhaseVelocityFaceFreePublicXtra(void);
int PhaseVelocityFaceSizeOfTempData(void);

/* ppcg.c */
void PPCG(Vector *x, Vector *b, double tol, int zero);
PFModule *PPCGInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
void PPCGFreeInstanceXtra(void);
PFModule *PPCGNewPublicXtra(char *name);
void PPCGFreePublicXtra(void);
int PPCGSizeOfTempData(void);

/* printgrid.c */
void PrintGrid(char *filename, Grid *grid);

/* printmatrix.c */
void PrintSubmatrixAll(amps_File file, Submatrix *submatrix, Stencil *stencil);
void PrintMatrixAll(char *filename, Matrix *A);
void PrintSubmatrix(amps_File file, Submatrix *submatrix, Subregion *subregion, Stencil *stencil);
void PrintMatrix(char *filename, Matrix *A);
void PrintSortMatrix(char *filename, Matrix *A, int all);

/* printvector.c */
void PrintSubvectorAll(amps_File file, Subvector *subvector);
void PrintVectorAll(char *filename, Vector *v);
void PrintSubvector(amps_File file, Subvector *subvector, Subgrid *subgrid);
void PrintVector(char *filename, Vector *v);

/* problem.c */
Problem *NewProblem(int solver);
void FreeProblem(Problem *problem, int solver);
ProblemData *NewProblemData(Grid *grid, Grid *grid2d);
void FreeProblemData(ProblemData *problem_data);

/* problem_bc.c */
BCStruct *NewBCStruct(SubgridArray *subgrids, GrGeomSolid *gr_domain, int num_patches, int *patch_indexes, int *bc_types, double ***values);
void FreeBCStruct(BCStruct *bc_struct);


typedef void (*BCInternalInvoke) (Problem *problem, ProblemData *problem_data, Matrix *A, Vector *f, double time);

/* problem_bc_internal.c */
void BCInternal(Problem *problem, ProblemData *problem_data, Matrix *A, Vector *f, double time);
PFModule *BCInternalInitInstanceXtra(void);
void BCInternalFreeInstanceXtra(void);
PFModule *BCInternalNewPublicXtra(void);
void BCInternalFreePublicXtra(void);
int BCInternalSizeOfTempData(void);

typedef void (*BCPhaseSaturationInvoke) (Vector *saturation, int phase, GrGeomSolid *gr_domain);
typedef PFModule *(*BCPhaseSaturationNewPublicXtraInvoke) (int num_phases);

/* problem_bc_phase_saturation.c */
void BCPhaseSaturation(Vector *saturation, int phase, GrGeomSolid *gr_domain);
PFModule *BCPhaseSaturationInitInstanceXtra(void);
void BCPhaseSaturationFreeInstanceXtra(void);
PFModule *BCPhaseSaturationNewPublicXtra(int num_phases);
void BCPhaseSaturationFreePublicXtra(void);
int BCPhaseSaturationSizeOfTempData(void);

typedef BCStruct *(*BCPressureInvoke) (ProblemData *problem_data, Grid *grid, GrGeomSolid *gr_domain, double time);
typedef PFModule *(*BCPressureInitInstanceXtraInvoke) (Problem *problem);
typedef PFModule *(*BCPressureNewPublicXtraInvoke) (int num_phases);

/* problem_bc_pressure.c */
BCStruct *BCPressure(ProblemData *problem_data, Grid *grid, GrGeomSolid *gr_domain, double time);
PFModule *BCPressureInitInstanceXtra(Problem *problem);
void BCPressureFreeInstanceXtra(void);
PFModule *BCPressureNewPublicXtra(int num_phases);
void BCPressureFreePublicXtra(void);
int BCPressureSizeOfTempData(void);

typedef void (*CapillaryPressureInvoke) (Vector *capillary_pressure, int phase_i, int phase_j, ProblemData *problem_data, Vector *phase_saturation);
typedef PFModule *(*CapillaryPressureNewPublicXtraInvoke) (int num_phases);

/* problem_capillary_pressure.c */
void CapillaryPressure(Vector *capillary_pressure, int phase_i, int phase_j, ProblemData *problem_data, Vector *phase_saturation);
PFModule *CapillaryPressureInitInstanceXtra(void);
void CapillaryPressureFreeInstanceXtra(void);
PFModule *CapillaryPressureNewPublicXtra(int num_phases);
void CapillaryPressureFreePublicXtra(void);
int CapillaryPressureSizeOfTempData(void);

typedef void (*DomainInvoke) (ProblemData *problem_data);
typedef PFModule *(*DomainInitInstanceXtraInvoke) (Grid *grid);
/* problem_domain.c */
void Domain(ProblemData *problem_data);
PFModule *DomainInitInstanceXtra(Grid *grid);
void DomainFreeInstanceXtra(void);
PFModule *DomainNewPublicXtra(void);
void DomainFreePublicXtra(void);
int DomainSizeOfTempData(void);

/* problem_eval.c */
EvalStruct *NewEvalStruct(Problem *problem);
void FreeEvalStruct(EvalStruct *eval_struct);

typedef void (*GeometriesInvoke) (ProblemData *problem_data);
typedef PFModule *(*GeometriesInitInstanceXtraInvoke) (Grid *grid);
/* problem_geometries.c */
void Geometries(ProblemData *problem_data);

PFModule *GeometriesInitInstanceXtra(Grid *grid);
void GeometriesFreeInstanceXtra(void);
PFModule *GeometriesNewPublicXtra(void);
void GeometriesFreePublicXtra(void);
int GeometriesSizeOfTempData(void);

typedef void (*ICPhaseConcenInvoke) (Vector *ic_phase_concen, int phase, int contaminant, ProblemData *problem_data);
typedef PFModule *(*ICPhaseConcenNewPublicXtraInvoke) (int num_phases, int num_contaminants);

/* problem_ic_phase_concen.c */
void ICPhaseConcen(Vector *ic_phase_concen, int phase, int contaminant, ProblemData *problem_data);
PFModule *ICPhaseConcenInitInstanceXtra(void);
void ICPhaseConcenFreeInstanceXtra(void);
PFModule *ICPhaseConcenNewPublicXtra(int num_phases, int num_contaminants);
void ICPhaseConcenFreePublicXtra(void);
int ICPhaseConcenSizeOfTempData(void);

typedef void (*ICPhasePressureInvoke) (Vector *ic_pressure, Vector *mask, ProblemData *problem_data, Problem *problem);
typedef PFModule *(*ICPhasePressureInitInstanceXtraInvoke) (Problem *problem, Grid *grid, double *temp_data);

/* problem_ic_phase_pressure.c */
void ICPhasePressure(Vector *ic_pressure, Vector *mask, ProblemData *problem_data, Problem *problem);
PFModule *ICPhasePressureInitInstanceXtra(Problem *problem, Grid *grid, double *temp_data);
void ICPhasePressureFreeInstanceXtra(void);
PFModule *ICPhasePressureNewPublicXtra(void);
void ICPhasePressureFreePublicXtra(void);
int ICPhasePressureSizeOfTempData(void);

typedef void (*ManningsInvoke) (ProblemData *problem_data, Vector *mann, Vector *dummy);
typedef PFModule *(*ManningsInitInstanceXtraInvoke) (Grid *grid3d, Grid *grid2d);

/* problem_mannings.c */
void Mannings(ProblemData *problem_data, Vector *mann, Vector *dummy);
PFModule *ManningsInitInstanceXtra(Grid *grid3d, Grid *grid2d);
void ManningsFreeInstanceXtra(void);
PFModule *ManningsNewPublicXtra(void);
void ManningsFreePublicXtra(void);
int ManningsSizeOfTempData(void);

typedef void (*SpecStorageInvoke) (ProblemData *problem_data, Vector *specific_storage);

/* problem_spec_storage.c */
void SpecStorage(ProblemData *problem_data, Vector *specific_storage);
PFModule *SpecStorageInitInstanceXtra(void);
void SpecStorageFreeInstanceXtra(void);
PFModule *SpecStorageNewPublicXtra(void);
void SpecStorageFreePublicXtra(void);
int SpecStorageSizeOfTempData(void);

/* @RMM new module for dz scaling factors */

typedef void (*dzScaleInvoke) (ProblemData *problem_data, Vector *dz_mult);

/* problem_dz_scale.c */
void dzScale(ProblemData *problem_data, Vector *dz_mult);
PFModule *dzScaleInitInstanceXtra(void);
void dzScaleFreeInstanceXtra(void);
PFModule *dzScaleNewPublicXtra(void);
void dzScaleFreePublicXtra(void);
int dzScaleSizeOfTempData(void);

/* RMM patterned FB (flow boundary) input from DZ scale,
 * three modules called  */
typedef void (*FBxInvoke) (ProblemData *problem_data, Vector *FBx);

/* problem_FBx.c */
void FBx(ProblemData *problem_data, Vector *FBx);
PFModule *FBxInitInstanceXtra(void);
void FBxFreeInstanceXtra(void);
PFModule *FBxNewPublicXtra(void);
void FBxFreePublicXtra(void);
int FBxSizeOfTempData(void);

typedef void (*FByInvoke) (ProblemData *problem_data, Vector *FBy);

/* problem_FBy.c */
void FBy(ProblemData *problem_data, Vector *FBy);
PFModule *FByInitInstanceXtra(void);
void FByFreeInstanceXtra(void);
PFModule *FByNewPublicXtra(void);
void FByFreePublicXtra(void);
int FBySizeOfTempData(void);

typedef void (*FBzInvoke) (ProblemData *problem_data, Vector *FBz);

/* problem_FBz.c */
void FBz(ProblemData *problem_data, Vector *FBz);
PFModule *FBzInitInstanceXtra(void);
void FBzFreeInstanceXtra(void);
PFModule *FBzNewPublicXtra(void);
void FBzFreePublicXtra(void);
int FBzSizeOfTempData(void);



typedef void (*realSpaceZInvoke) (ProblemData *problem_data, Vector *rsz);

/* problem_real_space_z.c */
void realSpaceZ(ProblemData *problem_data, Vector *rsz);
PFModule *realSpaceZInitInstanceXtra(void);
void realSpaceZFreeInstanceXtra(void);
PFModule *realSpaceZNewPublicXtra(void);
void realSpaceZFreePublicXtra(void);
int realSpaceZSizeOfTempData(void);




/* DOK - overlandfloweval */
typedef void (*OverlandFlowEvalInvoke) (Grid *       grid,
                                        int          sg,
                                        BCStruct *   bc_struct,
                                        int          ipatch,
                                        ProblemData *problem_data,
                                        Vector *     pressure,
                                        Vector *     old_pressure,
                                        double *     ke_v,
                                        double *     kw_v,
                                        double *     kn_v,
                                        double *     ks_v,
                                        double *     qx_v,
                                        double *     qy_v,
                                        int          fcn);

void OverlandFlowEval(Grid *       grid,
                      int          sg,
                      BCStruct *   bc_struct,
                      int          ipatch,
                      ProblemData *problem_data,
                      Vector *     pressure,
                      Vector *     old_pressure,
                      double *     ke_v,
                      double *     kw_v,
                      double *     kn_v,
                      double *     ks_v,
                      double *     qx_v,
                      double *     qy_v,
                      int          fcn);
PFModule *OverlandFlowEvalInitInstanceXtra(void);
void OverlandFlowEvalFreeInstanceXtra(void);
PFModule *OverlandFlowEvalNewPublicXtra(void);
void OverlandFlowEvalFreePublicXtra(void);
int OverlandFlowEvalSizeOfTempData(void);

/* @RMM - overlandflowevaldiffusive */
typedef void (*OverlandFlowEvalDiffInvoke) (Grid *       grid,
                                            int          sg,
                                            BCStruct *   bc_struct,
                                            int          ipatch,
                                            ProblemData *problem_data,
                                            Vector *     pressure,
                                            Vector *     old_pressure,
                                            double *     ke_v,
                                            double *     kw_v,
                                            double *     kn_v,
                                            double *     ks_v,
                                            double *     ke_vns,
                                            double *     kw_vns,
                                            double *     kn_vns,
                                            double *     ks_vns,
                                            double *     qx_v,
                                            double *     qy_v,
                                            int          fcn);

void OverlandFlowEvalDiff(Grid *       grid,
                          int          sg,
                          BCStruct *   bc_struct,
                          int          ipatch,
                          ProblemData *problem_data,
                          Vector *     pressure,
                          Vector *     old_pressure,
                          double *     ke_v,
                          double *     kw_v,
                          double *     kn_v,
                          double *     ks_v,
                          double *     ke_vns,
                          double *     kw_vns,
                          double *     kn_vns,
                          double *     ks_vns,
                          double *     qx_v,
                          double *     qy_v,
                          int          fcn);
PFModule *OverlandFlowEvalDiffInitInstanceXtra(void);
void OverlandFlowEvalDiffFreeInstanceXtra(void);
PFModule *OverlandFlowEvalDiffNewPublicXtra(void);
void OverlandFlowEvalDiffFreePublicXtra(void);
int OverlandFlowEvalDiffSizeOfTempData(void);


/* overlandflow_eval_kin.c */
typedef void (*OverlandFlowEvalKinInvoke) (Grid *       grid,
                                           int          sg,
                                           BCStruct *   bc_struct,
                                           int          ipatch,
                                           ProblemData *problem_data,
                                           Vector *     pressure,
                                           double *     ke_v,
                                           double *     kw_v,
                                           double *     kn_v,
                                           double *     ks_v,
                                           double *     ke_vns,
                                           double *     kw_vns,
                                           double *     kn_vns,
                                           double *     ks_vns,
                                           double *     qx_v,
                                           double *     qy_v,
                                           int          fcn);

void OverlandFlowEvalKin(Grid *       grid,
                         int          sg,
                         BCStruct *   bc_struct,
                         int          ipatch,
                         ProblemData *problem_data,
                         Vector *     pressure,
                         double *     ke_v,
                         double *     kw_v,
                         double *     kn_v,
                         double *     ks_v,
                         double *     ke_vns,
                         double *     kw_vns,
                         double *     kn_vns,
                         double *     ks_vns,
                         double *     qx_v,
                         double *     qy_v,
                         int          fcn);
PFModule *OverlandFlowEvalKinInitInstanceXtra(void);
void OverlandFlowEvalKinFreeInstanceXtra(void);
PFModule *OverlandFlowEvalKinNewPublicXtra(void);
void OverlandFlowEvalKinFreePublicXtra(void);
int OverlandFlowEvalKinSizeOfTempData(void);

typedef void (*ICPhaseSaturInvoke) (Vector *ic_phase_satur, int phase, ProblemData *problem_data);
typedef PFModule *(*ICPhaseSaturNewPublicXtraInvoke) (int num_phases);

/* problem_ic_phase_satur.c */
void ICPhaseSatur(Vector *ic_phase_satur, int phase, ProblemData *problem_data);
PFModule *ICPhaseSaturInitInstanceXtra(void);
void ICPhaseSaturFreeInstanceXtra(void);
PFModule *ICPhaseSaturNewPublicXtra(int num_phases);
void ICPhaseSaturFreePublicXtra(void);
int ICPhaseSaturSizeOfTempData(void);

/* problem_phase_density.c */

typedef void (*PhaseDensityInvoke) (int phase, Vector *phase_pressure, Vector *density_v, double *pressure_d, double *density_d, int fcn);
typedef PFModule *(*PhaseDensityNewPublicXtraInvoke) (int num_phases);

void PhaseDensityConstants(int phase, int fcn, int *phase_type, double *constant, double *ref_den, double *comp_const);
void PhaseDensity(int phase, Vector *phase_pressure, Vector *density_v, double *pressure_d, double *density_d, int fcn);
PFModule *PhaseDensityInitInstanceXtra(void);
void PhaseDensityFreeInstanceXtra(void);
PFModule *PhaseDensityNewPublicXtra(int num_phases);
void PhaseDensityFreePublicXtra(void);
int PhaseDensitySizeOfTempData(void);

typedef void (*PhaseMobilityInvoke) (Vector *phase_mobility_x, Vector *phase_mobility_y, Vector *phase_mobility_z, Vector *perm_x, Vector *perm_y, Vector *perm_z, int phase, Vector *phase_saturation, double phase_viscosity);
typedef PFModule *(*PhaseMobilityNewPublicXtraInvoke) (int num_phases);

/* problem_phase_mobility.c */
void PhaseMobility(Vector *phase_mobility_x, Vector *phase_mobility_y, Vector *phase_mobility_z, Vector *perm_x, Vector *perm_y, Vector *perm_z, int phase, Vector *phase_saturation, double phase_viscosity);
PFModule *PhaseMobilityInitInstanceXtra(void);
void PhaseMobilityFreeInstanceXtra(void);
PFModule *PhaseMobilityNewPublicXtra(int num_phases);
void PhaseMobilityFreePublicXtra(void);
int PhaseMobilitySizeOfTempData(void);

typedef void (*PhaseRelPermInvoke) (Vector *phase_rel_perm, Vector *phase_pressure, Vector *phase_density, double gravity, ProblemData *problem_data, int fcn);
typedef PFModule *(*PhaseRelPermInitInstanceXtraInvoke) (Grid *grid, double *temp_data);

/* problem_phase_rel_perm.c */
void PhaseRelPerm(Vector *phase_rel_perm, Vector *phase_pressure, Vector *phase_density, double gravity, ProblemData *problem_data, int fcn);
PFModule *PhaseRelPermInitInstanceXtra(Grid *grid, double *temp_data);
void PhaseRelPermFreeInstanceXtra(void);
PFModule *PhaseRelPermNewPublicXtra(void);
void PhaseRelPermFreePublicXtra(void);
int PhaseRelPermSizeOfTempData(void);

typedef void (*PhaseSourceInvoke) (Vector *phase_source, int phase, Problem *problem, ProblemData *problem_data, double time);
typedef PFModule *(*PhaseSourceNewPublicXtraInvoke) (int num_phases);

/* problem_phase_source.c */
void PhaseSource(Vector *phase_source, int phase, Problem *problem, ProblemData *problem_data, double time);
PFModule *PhaseSourceInitInstanceXtra(void);
void PhaseSourceFreeInstanceXtra(void);
PFModule *PhaseSourceNewPublicXtra(int num_phases);
void PhaseSourceFreePublicXtra(void);
int PhaseSourceSizeOfTempData(void);

typedef void (*PorosityInvoke) (ProblemData *problem_data, Vector *porosity, int num_geounits, GeomSolid **geounits, GrGeomSolid **gr_geounits);
typedef PFModule *(*PorosityInitInstanceXtraInvoke) (Grid *grid, double *temp_data);

/* problem_porosity.c */
void Porosity(ProblemData *problem_data, Vector *porosity, int num_geounits, GeomSolid **geounits, GrGeomSolid **gr_geounits);
PFModule *PorosityInitInstanceXtra(Grid *grid, double *temp_data);
void PorosityFreeInstanceXtra(void);
PFModule *PorosityNewPublicXtra(void);
void PorosityFreePublicXtra(void);
int PorositySizeOfTempData(void);

typedef void (*RetardationInvoke) (Vector *solidmassfactor, int contaminant, ProblemData *problem_data);
typedef PFModule *(*RetardationInitInstanceXtraInvoke) (double *temp_data);
typedef PFModule *(*RetardationNewPublicXtraInvoke) (int num_contaminants);

/* problem_retardation.c */
void Retardation(Vector *solidmassfactor, int contaminant, ProblemData *problem_data);

PFModule *RetardationInitInstanceXtra(double *temp_data);
void RetardationFreeInstanceXtra(void);
PFModule *RetardationNewPublicXtra(int num_contaminants);
void RetardationFreePublicXtra(void);
int RetardationSizeOfTempData(void);


typedef void (*RichardsBCInternalInvoke) (Problem *problem, ProblemData *problem_data, Vector *f, Matrix *A, double time, Vector *pressure, int fcn);

/* problem_richards_bc_internal.c */
void RichardsBCInternal(Problem *problem, ProblemData *problem_data, Vector *f, Matrix *A, double time, Vector *pressure, int fcn);
PFModule *RichardsBCInternalInitInstanceXtra(void);
void RichardsBCInternalFreeInstanceXtra(void);
PFModule *RichardsBCInternalNewPublicXtra(void);
void RichardsBCInternalFreePublicXtra(void);
int RichardsBCInternalSizeOfTempData(void);

typedef void (*SaturationInvoke) (Vector *phase_saturation, Vector *phase_pressure, Vector *phase_density, double gravity, ProblemData *problem_data, int fcn);
typedef PFModule *(*SaturationInitInstanceXtraInvoke) (Grid *grid, double *temp_data);
typedef PFModule *(*SaturationOutputStaticInvoke) (char *file_prefix, ProblemData *problem_data);

/* problem_saturation.c */
void Saturation(Vector *phase_saturation, Vector *phase_pressure, Vector *phase_density, double gravity, ProblemData *problem_data, int fcn);
PFModule *SaturationInitInstanceXtra(Grid *grid, double *temp_data);
void SaturationFreeInstanceXtra(void);
PFModule *SaturationNewPublicXtra(void);
void SaturationFreePublicXtra(void);
int SaturationSizeOfTempData(void);
void  SaturationOutput(void);
void  SaturationOutputStatic(char *file_prefix, ProblemData *problem_data);

typedef void (*SaturationConstitutiveInvoke) (Vector **phase_saturations);
typedef PFModule *(*SaturationConstitutiveInitInstanceXtraInvoke) (Grid *grid);
typedef PFModule *(*SaturationConstitutiveNewPublicXtraInvoke) (int num_phases);
/* problem_saturation_constitutive.c */
void SaturationConstitutive(Vector **phase_saturations);
PFModule *SaturationConstitutiveInitInstanceXtra(Grid *grid);
void SaturationConstitutiveFreeInstanceXtra(void);
PFModule *SaturationConstitutiveNewPublicXtra(int num_phases);
void SaturationConstitutiveFreePublicXtra(void);
int SaturationConstitutiveSizeOfTempData(void);

typedef void (*SlopeInvoke) (ProblemData *problem_data, Vector *x_sl, Vector *dummy);
typedef PFModule *(*SlopeInitInstanceXtraInvoke) (Grid *grid3d, Grid *grid2d);
/* problem_toposlope_x.c */
void XSlope(ProblemData *problem_data, Vector *x_sl, Vector *dummy);
PFModule *XSlopeInitInstanceXtra(Grid *grid3d, Grid *grid2d);
void XSlopeFreeInstanceXtra(void);
PFModule *XSlopeNewPublicXtra(void);
void XSlopeFreePublicXtra(void);
int XSlopeSizeOfTempData(void);

/* problem_toposlope_y.c */
void YSlope(ProblemData *problem_data, Vector *y_slope, Vector *dummy);
PFModule *YSlopeInitInstanceXtra(Grid *grid3d, Grid *grid2d);
void YSlopeFreeInstanceXtra(void);
PFModule *YSlopeNewPublicXtra(void);
void YSlopeFreePublicXtra(void);
int YSlopeSizeOfTempData(void);

typedef void (*ChannelWidthInvoke) (ProblemData *problem_data, Vector *wc_x, Vector *dummy);
typedef PFModule *(*ChannelWidthInitInstanceXtraInvoke) (Grid *grid3d, Grid *grid2d);
/* problem_wc_x.c */
void XChannelWidth(ProblemData *problem_data, Vector *x_wc, Vector *dummy);
PFModule *XChannelWidthInitInstanceXtra(Grid *grid3d, Grid *grid2d);
void XChannelWidthFreeInstanceXtra(void);
PFModule *XChannelWidthNewPublicXtra(void);
void XChannelWidthFreePublicXtra(void);
int XChannelWidthSizeOfTempData(void);

/* problem_wc_y.c */
void YChannelWidth(ProblemData *problem_data, Vector *wc_y, Vector *dummy);
PFModule *YChannelWidthInitInstanceXtra(Grid *grid3d, Grid *grid2d);
void YChannelWidthFreeInstanceXtra(void);
PFModule *YChannelWidthNewPublicXtra(void);
void YChannelWidthFreePublicXtra(void);
int YChannelWidthSizeOfTempData(void);

/* random.c */
void SeedRand(int seed);
double Rand(void);

/* ratqr.c */
int ratqr_(int *n, double *eps1, double *d, double *e, double *e2, int *m, double *w, int *ind, double *bd, int *type, int *idef, int *ierr);
double epslon_(double *x);


/* rb_GS_point.c */
void RedBlackGSPoint(Vector *x, Vector *b, double tol, int zero);
PFModule *RedBlackGSPointInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
void RedBlackGSPointFreeInstanceXtra(void);
PFModule *RedBlackGSPointNewPublicXtra(char *name);
void RedBlackGSPointFreePublicXtra(void);
int RedBlackGSPointSizeOfTempData(void);

/* read_parflow_binary.c */
void ReadPFBinary_Subvector(amps_File file, Subvector *subvector, Subgrid *subgrid);
void ReadPFBinary(char *filename, Vector *v);

/* reg_from_stenc.c */
void ComputeRegFromStencil(Region **dep_reg_ptr, Region **ind_reg_ptr, SubregionArray *cr_array, Region *send_reg, Region *recv_reg, Stencil *stencil);
SubgridArray *GetGridNeighbors(SubgridArray *subgrids, SubgridArray *all_subgrids, Stencil *stencil);
void CommRegFromStencil(Region **send_region_ptr, Region **recv_region_ptr, Grid *grid, Stencil *stencil);

/* region.c */
Subregion *NewSubregion(int ix, int iy, int iz, int nx, int ny, int nz, int sx, int sy, int sz, int rx, int ry, int rz, int process);
SubregionArray *NewSubregionArray(void);
Region *NewRegion(int size);
void FreeSubregion(Subregion *subregion);
void FreeSubregionArray(SubregionArray *subregion_array);
void FreeRegion(Region *region);
Subregion *DuplicateSubregion(Subregion *subregion);
SubregionArray *DuplicateSubregionArray(SubregionArray *subregion_array);
Region *DuplicateRegion(Region *region);
void AppendSubregion(Subregion *subregion, SubregionArray *sr_array);
void DeleteSubregion(SubregionArray *sr_array, int index);
void AppendSubregionArray(SubregionArray *sr_array_0, SubregionArray *sr_array_1);


typedef void (*RichardsJacobianEvalInvoke) (Vector *pressure, Vector *old_pressure, Matrix **ptr_to_J, Matrix **ptr_to_JC, Vector *saturation, Vector *density, ProblemData *problem_data, double dt, double time, int symm_part);
typedef PFModule *(*RichardsJacobianEvalInitInstanceXtraInvoke) (Problem *problem, Grid *grid, ProblemData *problem_data, double *temp_data, int symmetric_jac);
typedef PFModule *(*RichardsJacobianEvalNewPublicXtraInvoke) (char *name);
/* richards_jacobian_eval.c */
int KINSolMatVec(void *current_state, N_Vector x, N_Vector y, int *recompute, N_Vector pressure);
void RichardsJacobianEval(Vector *pressure, Vector *old_pressure, Matrix **ptr_to_J, Matrix **ptr_to_JC, Vector *saturation, Vector *density, ProblemData *problem_data, double dt, double time, int symm_part);
PFModule *RichardsJacobianEvalInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, double *temp_data, int symmetric_jac);
void RichardsJacobianEvalFreeInstanceXtra(void);
PFModule *RichardsJacobianEvalNewPublicXtra(char *name);
void RichardsJacobianEvalFreePublicXtra(void);
int RichardsJacobianEvalSizeOfTempData(void);

typedef void (*AdvectionSaturationInvoke) (ProblemData *problem_data, int phase, Vector *old_saturation, Vector *new_saturation, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, Vector *z_permeability, Vector *solid_mass_factor, double *viscosity, double *density, double gravity, double time, double deltat, int order);
typedef PFModule *(*AdvectionSaturationInitInstanceXtraInvoke) (Problem *problem, Grid *grid, double *temp_data);

/* sadvection_godunov.c */
void SatGodunov(ProblemData *problem_data, int phase, Vector *old_saturation, Vector *new_saturation, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, Vector *z_permeability, Vector *solid_mass_factor, double *viscosity, double *density, double gravity, double time, double deltat, int order);
PFModule *SatGodunovInitInstanceXtra(Problem *problem, Grid *grid, double *temp_data);
void SatGodunovFreeInstanceXtra(void);
PFModule *SatGodunovNewPublicXtra(void);
void SatGodunovFreePublicXtra(void);
int SatGodunovSizeOfTempData(void);

/* scale.c */
void Scale(double alpha, Vector *y);

typedef void (*SelectTimeStepInvoke) (double *dt, char *dt_info, double time, Problem *problem, ProblemData *problem_data);

/* select_time_step.c */
void SelectTimeStep(double *dt, char *dt_info, double time, Problem *problem, ProblemData *problem_data);
PFModule *SelectTimeStepInitInstanceXtra(void);
void SelectTimeStepFreeInstanceXtra(void);
PFModule *SelectTimeStepNewPublicXtra(void);
void SelectTimeStepFreePublicXtra(void);
int SelectTimeStepSizeOfTempData(void);
PFModule  *WRFSelectTimeStepNewPublicXtra(
                                          double initial_step,
                                          double growth_factor,
                                          double max_step,
                                          double min_step);
void  WRFSelectTimeStepFreePublicXtra();

PFModule  *WRFSelectTimeStepInitInstanceXtra();
PFModule  *WRFSelectTimeStepNewPublicXtra(
                                          double initial_step,
                                          double growth_factor,
                                          double max_step,
                                          double min_step);
void  WRFSelectTimeStepFreePublicXtra();
PFModule  *CPLSelectTimeStepInitInstanceXtra();
void CPLSelectTimeStepFreeInstanceXtra();
PFModule  *CPLSelectTimeStepNewPublicXtra(
                                          double initial_step,
                                          double growth_factor,
                                          double max_step,
                                          double min_step);
void  CPLSelectTimeStepFreePublicXtra();
int  CPLSelectTimeStepSizeOfTempData();

typedef void (*SetProblemDataInvoke) (ProblemData *problem_data);
typedef PFModule *(*SetProblemDataInitInstanceXtraInvoke) (Problem *problem, Grid *grid, Grid *grid2d, double *temp_data);

/* set_problem_data.c */
void SetProblemData(ProblemData *problem_data);
PFModule *SetProblemDataInitInstanceXtra(Problem *problem, Grid *grid, Grid *grid2d, double *temp_data);
void SetProblemDataFreeInstanceXtra(void);
PFModule *SetProblemDataNewPublicXtra(void);
void SetProblemDataFreePublicXtra(void);
int SetProblemDataSizeOfTempData(void);

/* sim_shear.c */
double **SimShear(double **shear_min_ptr, double **shear_max_ptr, GeomSolid *geom_solid, SubgridArray *subgrids, int type);

/* solver.c */
void Solve(void);
void NewSolver(void);
void FreeSolver(void);

typedef void (*SolverInvoke)(void);
typedef PFModule *(*SolverImpesNewPublicXtraInvoke) (char *name);

/* solver_impes.c */
void SolverImpes(void);
PFModule *SolverImpesInitInstanceXtra(void);
void SolverImpesFreeInstanceXtra(void);
PFModule *SolverImpesNewPublicXtra(char *name);
void SolverImpesFreePublicXtra(void);
int SolverImpesSizeOfTempData(void);

/* solver_lb.c */
void SolverDiffusion(void);
PFModule *SolverDiffusionInitInstanceXtra(void);
void SolverDiffusionFreeInstanceXtra(void);
PFModule *SolverDiffusionNewPublicXtra(char *name);
void SolverDiffusionFreePublicXtra(void);
int SolverDiffusionSizeOfTempData(void);

typedef PFModule *(*SolverNewPublicXtraInvoke) (char *name);

/* solver_richards.c */
void SolverRichards(void);
PFModule *SolverRichardsInitInstanceXtra(void);
void SolverRichardsFreeInstanceXtra(void);
PFModule *SolverRichardsNewPublicXtra(char *name);
void SolverRichardsFreePublicXtra(void);
int SolverRichardsSizeOfTempData(void);
ProblemData *GetProblemDataRichards(PFModule *this_module);
Problem  *GetProblemRichards(PFModule *this_module);
PFModule *GetICPhasePressureRichards(PFModule *this_module);
Grid *GetGrid2DRichards(PFModule *this_module);
Vector *GetMaskRichards(PFModule *this_module);
void AdvanceRichards(PFModule *this_module,
                     double    start_time,   /* Starting time */
                     double    stop_time,    /* Stopping time */
                     PFModule *time_step_control, /* Use this module to control timestep if supplied */
                     Vector *  evap_trans,   /* Flux from land surface model */
                     Vector ** pressure_out, /* Output vars */
                     Vector ** porosity_out,
                     Vector ** saturation_out
                     );
void ExportRichards(PFModule *this_module,
                    Vector ** pressure_out,  /* Output vars */
                    Vector ** porosity_out,
                    Vector ** saturation_out
                    );
void SetupRichards(PFModule *this_module);


typedef void (*SubsrfSimInvoke) (ProblemData *problem_data, Vector *perm_x, Vector *perm_y, Vector *perm_z, int num_geounits, GeomSolid **geounits, GrGeomSolid **gr_geounits);
typedef PFModule *(*SubsrfSimInitInstanceXtraInvoke) (Grid *grid, double *temp_data);

/* subsrf_sim.c */
void SubsrfSim(ProblemData *problem_data, Vector *perm_x, Vector *perm_y, Vector *perm_z, int num_geounits, GeomSolid **geounits, GrGeomSolid **gr_geounits);
PFModule *SubsrfSimInitInstanceXtra(Grid *grid, double *temp_data);
void SubsrfSimFreeInstanceXtra(void);
PFModule *SubsrfSimNewPublicXtra(void);
void SubsrfSimFreePublicXtra(void);
int SubsrfSimSizeOfTempData(void);

/* time_cycle_data.c */
TimeCycleData *NewTimeCycleData(int number_of_cycles, int *number_of_intervals);
void FreeTimeCycleData(TimeCycleData *time_cycle_data);
void PrintTimeCycleData(TimeCycleData *time_cycle_data);
int TimeCycleDataComputeIntervalNumber(Problem *problem, double time, TimeCycleData *time_cycle_data, int cycle_number);
double TimeCycleDataComputeNextTransition(Problem *problem, double time, TimeCycleData *time_cycle_data);
void ReadGlobalTimeCycleData(void);
void FreeGlobalTimeCycleData(void);

/* timing.c */
#if defined(PF_TIMING)
void NewTiming(void);
int RegisterTiming(char *name);
void PrintTiming(void);
void FreeTiming(void);
#endif

typedef void (*TotalVelocityFaceInvoke) (Vector *xvel, Vector *yvel, Vector *zvel, ProblemData *problem_data, Vector *total_mobility_x, Vector *total_mobility_y, Vector *total_mobility_z, Vector *pressure, Vector **saturations);
typedef PFModule *(*TotalVelocityFaceInitInstanceXtraInvoke) (Problem *problem, Grid *grid, Grid *x_grid, Grid *y_grid, Grid *z_grid, double *temp_data);

/* total_velocity_face.c */
void TotalVelocityFace(Vector *xvel, Vector *yvel, Vector *zvel, ProblemData *problem_data, Vector *total_mobility_x, Vector *total_mobility_y, Vector *total_mobility_z, Vector *pressure, Vector **saturations);
PFModule *TotalVelocityFaceInitInstanceXtra(Problem *problem, Grid *grid, Grid *x_grid, Grid *y_grid, Grid *z_grid, double *temp_data);
void TotalVelocityFaceFreeInstanceXtra(void);
PFModule *TotalVelocityFaceNewPublicXtra(void);
void TotalVelocityFaceFreePublicXtra(void);
int TotalVelocityFaceSizeOfTempData(void);

/* turning_bands.c */
void Turn(Vector *field, void *vxtra);
typedef void (*TurnInvoke) (Vector *field, void *vxtra);
int InitTurn(void);
void *NewTurn(char *geom_name);
void FreeTurn(void *xtra);

typedef void (*KFieldSimulatorInvoke) (GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata);
typedef PFModule *(*KFieldSimulatorInitInstanceXtraInvoke) (Grid *grid, double *temp_data);
typedef PFModule *(*KFieldSimulatorNewPublicXtra) (char *geom_name);

/* turning_bandsRF.c */
void TurningBandsRF(GeomSolid *geounit, GrGeomSolid *gr_geounit, Vector *field, RFCondData *cdata);
PFModule *TurningBandsRFInitInstanceXtra(Grid *grid, double *temp_data);
void TurningBandsRFFreeInstanceXtra(void);
PFModule *TurningBandsRFNewPublicXtra(char *geom_name);
void TurningBandsRFFreePublicXtra(void);
int TurningBandsRFSizeOfTempData(void);

/* usergrid_input.c */
Subgrid *ReadUserSubgrid(void);
Grid *ReadUserGrid(void);
void FreeUserGrid(Grid *user_grid);

/* vector.c */
CommPkg *NewVectorCommPkg(Vector *vector, ComputePkg *compute_pkg);
VectorUpdateCommHandle  *InitVectorUpdate(
                                          Vector *vector,
                                          int     update_mode);
void         FinalizeVectorUpdate(
                                  VectorUpdateCommHandle *handle);
Vector  *NewVector(
                   Grid *grid,
                   int   nc,
                   int   num_ghost);
Vector  *NewVectorType(
                       Grid *           grid,
                       int              nc,
                       int              num_ghost,
                       enum vector_type type);
void FreeVector(Vector *vector);
void InitVector(Vector *v, double value);
void InitVectorAll(Vector *v, double value);
void InitVectorInc(Vector *v, double value, double inc);
void InitVectorRandom(Vector *v, long seed);

#ifdef __cplusplus
extern "C" {
#endif

/* vector_utilities.c */
void PFVLinearSum(double a, Vector *x, double b, Vector *y, Vector *z);
void PFVConstInit(double c, Vector *z);
void PFVProd(Vector *x, Vector *y, Vector *z);
void PFVDiv(Vector *x, Vector *y, Vector *z);
void PFVScale(double c, Vector *x, Vector *z);
void PFVAbs(Vector *x, Vector *z);
void PFVInv(Vector *x, Vector *z);
void PFVAddConst(Vector *x, double b, Vector *z);
double PFVDotProd(Vector *x, Vector *y);
double PFVMaxNorm(Vector *x);
double PFVWrmsNorm(Vector *x, Vector *w);
double PFVWL2Norm(Vector *x, Vector *w);
double PFVL1Norm(Vector *x);
double PFVMin(Vector *x);
double PFVMax(Vector *x);
int PFVConstrProdPos(Vector *c, Vector *x);
void PFVCompare(double c, Vector *x, Vector *z);
int PFVInvTest(Vector *x, Vector *z);
void PFVCopy(Vector *x, Vector *y);
void PFVSum(Vector *x, Vector *y, Vector *z);
void PFVDiff(Vector *x, Vector *y, Vector *z);
void PFVNeg(Vector *x, Vector *z);
void PFVScaleSum(double c, Vector *x, Vector *y, Vector *z);
void PFVScaleDiff(double c, Vector *x, Vector *y, Vector *z);
void PFVLin1(double a, Vector *x, Vector *y, Vector *z);
void PFVLin2(double a, Vector *x, Vector *y, Vector *z);
void PFVAxpy(double a, Vector *x, Vector *y);
void PFVScaleBy(double a, Vector *x);
void PFVLayerCopy(int a, int b, Vector *x, Vector *y);

#ifdef __cplusplus
}
#endif

/* w_jacobi.c */
void WJacobi(Vector *x, Vector *b, double tol, int zero);
PFModule *WJacobiInitInstanceXtra(Problem *problem, Grid *grid, ProblemData *problem_data, Matrix *A, double *temp_data);
void WJacobiFreeInstanceXtra(void);
PFModule *WJacobiNewPublicXtra(char *name);
void WJacobiFreePublicXtra(void);
int WJacobiSizeOfTempData(void);

/* well.c */
WellData *NewWellData(void);
void FreeWellData(WellData *well_data);
void PrintWellData(WellData *well_data, unsigned int print_mask);
void WriteWells(char *file_prefix, Problem *problem, WellData *well_data, double time, int write_header);

typedef void (*WellPackageInvoke) (ProblemData *problem_data);
typedef PFModule *(*WellPackageNewPublicXtraInvoke) (int num_phases, int num_contaminants);

/* well_package.c */
void WellPackage(ProblemData *problem_data);
PFModule *WellPackageInitInstanceXtra(void);
void WellPackageFreeInstanceXtra(void);
PFModule *WellPackageNewPublicXtra(int num_phases, int num_contaminants);
void WellPackageFreePublicXtra(void);
int WellPackageSizeOfTempData(void);

/* reservoir.c */
ReservoirData *NewReservoirData(void);
void FreeReservoirData(ReservoirData *reservoir_data);
void PrintReservoirData(ReservoirData *reservoir_data, unsigned int print_mask);
void WriteReservoirs(char *file_prefix, Problem *problem, ReservoirData *reservoir_data, double time, int write_header);

typedef void (*ReservoirPackageInvoke) (ProblemData *problem_data);
typedef PFModule *(*ReservoirPackageNewPublicXtraInvoke) (int num_phases, int num_contaminants);



/* wells_lb.c */
void LBWells(Lattice *lattice, Problem *problem, ProblemData *problem_data);

/* reservoir_package.c */
void ReservoirPackage(ProblemData *problem_data);
PFModule *ReservoirPackageInitInstanceXtra(void);
void ReservoirPackageFreeInstanceXtra(void);
PFModule *ReservoirPackageNewPublicXtra(void);
void ReservoirPackageFreePublicXtra(void);
int ReservoirPackageSizeOfTempData(void);

/* write_parflow_binary.c */
long SizeofPFBinarySubvector(Subvector *subvector, Subgrid *subgrid);
void WritePFBinary_Subvector(amps_File file, Subvector *subvector, Subgrid *subgrid);
void WritePFBinary(char *file_prefix, char *file_suffix, Vector *v);
long SizeofPFSBinarySubvector(Subvector *subvector, Subgrid *subgrid, double drop_tolerance);
void WritePFSBinary_Subvector(amps_File file, Subvector *subvector, Subgrid *subgrid, double drop_tolerance);
void WritePFSBinary(char *file_prefix, char *file_suffix, Vector *v, double drop_tolerance);

/* write_parflow_silo.c */
void WriteSilo(char *  file_prefix,
               char *  file_type,
               char *  file_suffix,
               Vector *v,
               double  time,
               int     step,
               char *  variable_name);
void WriteSiloInit(char *file_prefix);
void pf_mk_dir(char *filename);

/* write_parflow_silo_PMPIO.c */
void     WriteSiloPMPIO(char *  file_prefix,
                        char *  file_type,
                        char *  file_suffix,
                        Vector *v,
                        double  time,
                        int     step,
                        char *  variable_name);
void     WriteSiloPMPIOInit(char *file_prefix);


/* wrf_parflow.c */
void wrfparflowinit_(char *input_file);
void wrfparflowadvance_(double *current_time,
                        double *dt,
                        float * wrf_flux,
                        float * wrf_pressure,
                        float * wrf_porosity,
                        float * wrf_saturation,
                        int *   num_soil_layers,
                        int *   ghost_size_i_lower,  /* Number of ghost cells */
                        int *   ghost_size_j_lower,
                        int *   ghost_size_i_upper,
                        int *   ghost_size_j_upper);

void WRF2PF(float * wrf_array,
            int     wrf_depth,
            int     ghost_size_i_lower,  /* Number of ghost cells */
            int     ghost_size_j_lower,
            int     ghost_size_i_upper,
            int     ghost_size_j_upper,
            Vector *pf_vector,
            Vector *top);

void PF2WRF(Vector *pf_vector,
            float * wrf_array,
            int     wrf_depth,
            int     ghost_size_i_lower,    /* Number of ghost cells */
            int     ghost_size_j_lower,
            int     ghost_size_i_upper,
            int     ghost_size_j_upper,
            Vector *top);

/* cpl_parflow.c */
void cplparflowinit_(int *  fcom,
                     char * input_file,
                     int *  numprocs,
                     int *  subgridcount,
                     int *  nz,
                     int *  ierror);

void cplparflowadvance_(double * current_time,
                        double * dt,
                        float *  imp_flux,
                        float *  exp_pressure,
                        float *  exp_porosity,
                        float *  exp_saturation,
                        float *  exp_specific,
                        float *  exp_zmult,
                        int *    num_soil_layers,
                        int *    num_cpl_layers,
                        int *    ghost_size_i_lower, /* Number of ghost cells */
                        int *    ghost_size_j_lower,
                        int *    ghost_size_i_upper,
                        int *    ghost_size_j_upper,
                        int *    ierror);

void cplparflowexport_(float * exp_pressure,
                       float * exp_porosity,
                       float * exp_saturation,
                       float * exp_specific,
                       float * exp_zmult,
                       int *   num_soil_layers,
                       int *   num_cpl_layers,
                       int *   ghost_size_i_lower,  /* Number of ghost cells */
                       int *   ghost_size_j_lower,
                       int *   ghost_size_i_upper,
                       int *   ghost_size_j_upper,
                       int *   ierror);

void CPL2PF(float *  imp_array,
            int      imp_nz,
            int      cpy_layers,
            int      ghost_size_i_lower, /* Number of ghost cells */
            int      ghost_size_j_lower,
            int      ghost_size_i_upper,
            int      ghost_size_j_upper,
            Vector * pf_vector,
            Vector * top,
            Vector * mask);

void PF2CPL(Vector * pf_vector,
            float *  exp_array,
            int      exp_nz,
            int      ghost_size_i_lower, /* Number of ghost cells */
            int      ghost_size_j_lower,
            int      ghost_size_i_upper,
            int      ghost_size_j_upper,
            Vector * top,
            Vector * mask);

void cplparflowlcldecomp_(int * sg,
                          int * lowerx,
                          int * upperx,
                          int * lowery,
                          int * uppery,
                          int * ierror);

void cplparflowlclmask_(int * sg,
                        int * localmask,
                        int * ierror);

void cplparflowlclxyctr_(int *   sg,
                         float * localx,
                         float * localy,
                         int *   ierror);

void cplparflowlclxyedg_(int *   sg,
                         float * localx,
                         float * localy,
                         int *   ierror);

void ComputeTop(Problem *    problem,
                ProblemData *problem_data);

void ComputePatchTop(Problem *    problem,
                     ProblemData *problem_data);

int CheckTime(Problem *problem, char *key, double time);

/* evaptranssum.c */
void EvapTransSum(ProblemData *problem_data, double dt, Vector *evap_trans_sum, Vector *evap_trans);

void OverlandSum(ProblemData *problem_data,
                 Vector *     pressure,       /* Current pressure values */
                 double       dt,
                 Vector *     overland_sum);

Grid      *ReadProcessGrid();
