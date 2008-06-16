#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* Header.c */

/* advection_godunov.c */
void Godunov P((ProblemData *problem_data , int phase , int concentration , Vector *old_concentration , Vector *new_concentration , Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *solid_mass_factor , double time , double deltat , int order ));
PFModule *GodunovInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void GodunovFreeInstanceXtra P((void ));
PFModule *GodunovNewPublicXtra P((void ));
void GodunovFreePublicXtra P((void ));
int GodunovSizeOfTempData P((void ));

/* axpy.c */
void Axpy P((double alpha , Vector *x , Vector *y ));

/* background.c */
Background *ReadBackground P((void ));
void FreeBackground P((Background *background ));
void SetBackgroundBounds P((Background *background , Grid *grid ));

/* bc_lb.c */
void LBInitializeBC P((Lattice *lattice , Problem *problem , ProblemData *problem_data ));

/* bc_pressure.c */
BCPressureData *NewBCPressureData P((void ));
void FreeBCPressureData P((BCPressureData *bc_pressure_data ));
void PrintBCPressureData P((BCPressureData *bc_pressure_data ));

/* bc_temperature.c */
BCTemperatureData *NewBCTemperatureData P((void ));
void FreeBCTemperatureData P((BCTemperatureData *bc_temperature_data ));
void PrintBCTemperatureData P((BCTemperatureData *bc_temperature_data ));

/* bc_pressure_package.c */
void BCPressurePackage P((ProblemData *problem_data ));
PFModule *BCPressurePackageInitInstanceXtra P((Problem *problem ));
void BCPressurePackageFreeInstanceXtra P((void ));
PFModule *BCPressurePackageNewPublicXtra P((int num_phases ));
void BCPressurePackageFreePublicXtra P((void ));
int BCPressurePackageSizeOfTempData P((void ));

/* bc_temperature_package.c */
void BCTemperaturePackage P((ProblemData *problem_data ));
PFModule *BCTemperaturePackageInitInstanceXtra P((Problem *problem ));
void BCTemperaturePackageFreeInstanceXtra P((void ));
PFModule *BCTemperaturePackageNewPublicXtra P((int num_phases ));
void BCTemperaturePackageFreePublicXtra P((void ));
int BCTemperaturePackageSizeOfTempData P((void ));

/* calc_elevations.c */
double **CalcElevations P((GeomSolid *geom_solid , int ref_patch , SubgridArray *subgrids ));

/* cghs.c */
void CGHS P((Vector *x , Vector *b , double tol , int zero ));
PFModule *CGHSInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void CGHSFreeInstanceXtra P((void ));
PFModule *CGHSNewPublicXtra P((char *name ));
void CGHSFreePublicXtra P((void ));
int CGHSSizeOfTempData P((void ));

/* char_vector.c */
CommPkg *NewCharVectorUpdatePkg P((CharVector *charvector , int update_mode ));
CommHandle *InitCharVectorUpdate P((CharVector *charvector , int update_mode ));
void FinalizeCharVectorUpdate P((CommHandle *handle ));
CharVector *NewTempCharVector P((Grid *grid , int nc , int num_ghost ));
void SetTempCharVectorData P((CharVector *charvector , char *data ));
CharVector *NewCharVector P((Grid *grid , int nc , int num_ghost ));
void FreeTempCharVector P((CharVector *charvector ));
void FreeCharVector P((CharVector *charvector ));
void InitCharVector P((CharVector *v , int value ));
void InitCharVectorAll P((CharVector *v , int value ));
void InitCharVectorInc P((CharVector *v , int value , int inc ));

/* chebyshev.c */
void Chebyshev P((Vector *x , Vector *b , double tol , int zero , double ia , double ib , int num_iter ));
PFModule *ChebyshevInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void ChebyshevFreeInstanceXtra P((void ));
PFModule *ChebyshevNewPublicXtra P((char *name ));
void ChebyshevFreePublicXtra P((void ));
int ChebyshevSizeOfTempData P((void ));

/* comm_pkg.c */
void ProjectRegion P((Region *region , int sx , int sy , int sz , int ix , int iy , int iz ));
Region *ProjectRBPoint P((Region *region , int rb [4 ][3 ]));
void CreateComputePkgs P((Grid *grid ));
void FreeComputePkgs P((Grid *grid ));

/* communication.c */
int NewCommPkgInfo P((Subregion *data_sr , Subregion *comm_sr , int index , int num_vars , int *loop_array ));
CommPkg *NewCommPkg P((Region *send_region , Region *recv_region , SubregionArray *data_space , int num_vars , double *data ));
void FreeCommPkg P((CommPkg *pkg ));
CommHandle *InitCommunication P((CommPkg *comm_pkg ));
void FinalizeCommunication P((CommHandle *handle ));

/* computation.c */
ComputePkg *NewComputePkg P((Region *send_reg , Region *recv_reg , Region *dep_reg , Region *ind_reg ));
void FreeComputePkg P((ComputePkg *compute_pkg ));

/* compute_maximums.c */
double ComputePhaseMaximum P((double phase_u_max , double dx , double phase_v_max , double dy , double phase_w_max , double dz ));
double ComputeTotalMaximum P((Problem *problem , EvalStruct *eval_struct , double s_lower , double s_upper , double total_u_max , double dx , double total_v_max , double dy , double total_w_max , double beta_max , double dz ));

/* compute_total_concentration.c */
double ComputeTotalConcen P((GrGeomSolid *gr_domain , Grid *grid , Vector *substance ));

/* constantRF.c */
void ConstantRF P((GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata ));
PFModule *ConstantRFInitInstanceXtra P((Grid *grid , double *temp_data ));
void ConstantRFFreeInstanceXtra P((void ));
PFModule *ConstantRFNewPublicXtra P((char *geom_name ));
void ConstantRFFreePublicXtra P((void ));
int ConstantRFSizeOfTempData P((void ));

/* constant_porosity.c */
void ConstantPorosity P((GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field ));
PFModule *ConstantPorosityInitInstanceXtra P((Grid *grid , double *temp_data ));
void ConstantPorosityFreeInstanceXtra P((void ));
PFModule *ConstantPorosityNewPublicXtra P((char *geom_name ));
void ConstantPorosityFreePublicXtra P((void ));
int ConstantPorositySizeOfTempData P((void ));

/* copy.c */
void Copy P((Vector *x , Vector *y ));

/* create_grid.c */
SubgridArray *GetGridSubgrids P((SubgridArray *all_subgrids ));
Grid *CreateGrid P((Grid *user_grid ));

/* diag_scale.c */
void DiagScale P((Vector *x , Matrix *A , Vector *b , Vector *d ));

/* diffuse_lb.c */
void DiffuseLB P((Lattice *lattice , Problem *problem , int max_iterations , char *file_prefix ));
void LatticeFlowInit P((Lattice *lattice , Problem *problem ));
double MaxVectorValue P((Vector *field ));
double MaxVectorDividend P((Vector *field1 , Vector *field2 ));

/* discretize_pressure.c */
void DiscretizePressure P((Matrix **ptr_to_A , Vector **ptr_to_f , ProblemData *problem_data , double time , Vector *total_mobility_x , Vector *total_mobility_y , Vector *total_mobility_z , Vector **phase_saturations ));
PFModule *DiscretizePressureInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void DiscretizePressureFreeInstanceXtra P((void ));
PFModule *DiscretizePressureNewPublicXtra P((void ));
void DiscretizePressureFreePublicXtra P((void ));
int DiscretizePressureSizeOfTempData P((void ));

/* distribute_usergrid.c */
SubgridArray *DistributeUserGrid P((Grid *user_grid ));

/* dpofa.c */
int dpofa_ P((double *a , int *lda , int *n , int *info ));
double ddot_ P((int *n , double *dx , int *incx , double *dy , int *incy ));

/* dposl.c */
int dposl_ P((double *a , int *lda , int *n , double *b ));
int daxpy_ P((int *n , double *da , double *dx , int *incx , double *dy , int *incy ));

/* gauinv.c */
int gauinv_ P((double *p , double *xp , int *ierr ));

/* general.c */
char *malloc_chk P((int size , char *file , int line ));
char *calloc_chk P((int count , int elt_size , char *file , int line ));
int Exp2 P((int p ));

/* geom_t_solid.c */
GeomTSolid *GeomNewTSolid P((GeomTIN *surface , int **patches , int num_patches , int *num_patch_triangles ));
void GeomFreeTSolid P((GeomTSolid *solid ));
int GeomReadTSolids P((GeomTSolid ***solids_data_ptr , char *geom_input_name ));
GeomTSolid *GeomTSolidFromBox P((double xl , double yl , double zl , double xu , double yu , double zu ));

/* geometry.c */
GeomVertexArray *GeomNewVertexArray P((GeomVertex **vertices , int nV ));
void GeomFreeVertexArray P((GeomVertexArray *vertex_array ));
GeomTIN *GeomNewTIN P((GeomVertexArray *vertex_array , GeomTriangle **triangles , int nT ));
void GeomFreeTIN P((GeomTIN *surface ));
GeomSolid *GeomNewSolid P((void *data , int type ));
void GeomFreeSolid P((GeomSolid *solid ));
int GeomReadSolids P((GeomSolid ***solids_ptr , char *geom_input_name , int type ));
GeomSolid *GeomSolidFromBox P((double xl , double yl , double zl , double xu , double yu , double zu , int type ));
void IntersectLineWithTriangle P((unsigned int line_direction , double coord_0 , double coord_1 , double v0_x , double v0_y , double v0_z , double v1_x , double v1_y , double v1_z , double v2_x , double v2_y , double v2_z , int *intersects , double *point , int *normal_component ));

/* globals.c */
void NewGlobals P((char *run_name ));
void FreeGlobals P((void ));
void LogGlobals P((void ));

/* grgeom_list.c */
ListMember *NewListMember P((double value , int normal_component , int triangle_id ));
void FreeListMember P((ListMember *member ));
void ListInsert P((ListMember **head , ListMember *member ));
int ListDelete P((ListMember **head , ListMember *member ));
ListMember *ListSearch P((ListMember *head , double value , int normal_component , int triangle_id ));
ListMember *ListValueSearch P((ListMember *head , double value ));
ListMember *ListValueNormalComponentSearch P((ListMember *head , double value , int normal_component ));
ListMember *ListTriangleIDSearch P((ListMember *head , int triangle_id ));
void ListFree P((ListMember **head ));
int ListLength P((ListMember *head ));
void ListPrint P((ListMember *head ));

/* grgeom_octree.c */
int GrGeomCheckOctree P((GrGeomOctree *grgeom_octree ));
void GrGeomFixOctree P((GrGeomOctree *grgeom_octree , GrGeomOctree **patch_octrees , int num_patches , int level , int num_indices ));
GrGeomOctree *GrGeomNewOctree P((void ));
void GrGeomNewOctreeChildren P((GrGeomOctree *grgeom_octree ));
void GrGeomFreeOctree P((GrGeomOctree *grgeom_octree ));
GrGeomOctree *GrGeomOctreeFind P((int *new_level , GrGeomOctree *grgeom_octree_root , int ix , int iy , int iz , int level ));
GrGeomOctree *GrGeomOctreeAddCell P((GrGeomOctree *grgeom_octree_root , unsigned int cell , int ix , int iy , int iz , int level ));
GrGeomOctree *GrGeomOctreeAddFace P((GrGeomOctree *grgeom_octree_root , int line_direction , int cell_index0 , int cell_index1 , int face_index , int extent_lower , int extent_upper , int level , int normal_in_direction ));
void GrGeomOctreeFromTIN P((GrGeomOctree **solid_octree_ptr , GrGeomOctree ***patch_octrees_ptr , GeomTIN *solid , int **patches , int num_patches , int *num_patch_triangles , GrGeomExtentArray *extent_array , double xlower , double ylower , double zlower , double xupper , double yupper , double zupper , int min_level , int max_level ));
void GrGeomOctreeFromInd P((GrGeomOctree **solid_octree_ptr , Vector *indicator_field , int indicator , double xlower , double ylower , double zlower , double xupper , double yupper , double zupper , int octree_bg_level , int octree_ix , int octree_iy , int octree_iz ));
void GrGeomPrintOctreeStruc P((amps_File file , GrGeomOctree *grgeom_octree ));
int GrGeomPrintOctreeLevel P((amps_File file , GrGeomOctree *grgeom_octree , int level , int current_level ));
void GrGeomPrintOctree P((char *filename , GrGeomOctree *grgeom_octree_root ));
void GrGeomPrintOctreeCells P((char *filename , GrGeomOctree *octree , int last_level ));
void GrGeomOctreeFree P((GrGeomOctree *grgeom_octree_root ));

/* grgeometry.c */
int GrGeomGetOctreeInfo P((double *xlp , double *ylp , double *zlp , double *xup , double *yup , double *zup , int *ixp , int *iyp , int *izp ));
GrGeomExtentArray *GrGeomNewExtentArray P((GrGeomExtents *extents , int size ));
void GrGeomFreeExtentArray P((GrGeomExtentArray *extent_array ));
GrGeomExtentArray *GrGeomCreateExtentArray P((SubgridArray *subgrids , int xl_ghost , int xu_ghost , int yl_ghost , int yu_ghost , int zl_ghost , int zu_ghost ));
GrGeomSolid *GrGeomNewSolid P((GrGeomOctree *data , GrGeomOctree **patches , int num_patches , int octree_bg_level , int octree_ix , int octree_iy , int octree_iz ));
void GrGeomFreeSolid P((GrGeomSolid *solid ));
void GrGeomSolidFromInd P((GrGeomSolid **solid_ptr , Vector *indicator_field , int indicator ));
void GrGeomSolidFromGeom P((GrGeomSolid **solid_ptr , GeomSolid *geom_solid , GrGeomExtentArray *extent_array ));

/* grid.c */
Grid *NewGrid P((SubgridArray *subgrids , SubgridArray *all_subgrids ));
void FreeGrid P((Grid *grid ));
int ProjectSubgrid P((Subgrid *subgrid , int sx , int sy , int sz , int ix , int iy , int iz ));
Subgrid *ConvertToSubgrid P((Subregion *subregion ));
Subgrid *ExtractSubgrid P((int rx , int ry , int rz , Subgrid *subgrid ));
Subgrid *IntersectSubgrids P((Subgrid *subgrid1 , Subgrid *subgrid2 ));
SubgridArray *SubtractSubgrids P((Subgrid *subgrid1 , Subgrid *subgrid2 ));
SubgridArray *UnionSubgridArray P((SubgridArray *subgrids ));

/* hbt.c */
HBT *HBT_new P((int (*compare_method )(), void (*free_method )(), void (*printf_method )(), int (*scanf_method )(), int malloc_flag ));
HBT_element *_new_HBT_element P((HBT *tree , void *object , int sizeof_obj ));
void _free_HBT_element P((HBT *tree , HBT_element *el ));
void _HBT_free P((HBT *tree , HBT_element *subtree ));
void HBT_free P((HBT *tree ));
void *HBT_lookup P((HBT *tree , void *obj ));
void *HBT_replace P((HBT *tree , void *obj , int sizeof_obj ));
int HBT_insert P((HBT *tree , void *obj , int sizeof_obj ));
void *HBT_delete P((HBT *tree , void *obj ));
void *HBT_successor P((HBT *tree , void *obj ));
void _HBT_printf P((FILE *file , void (*printf_method )(), HBT_element *tree ));
void HBT_printf P((FILE *file , HBT *tree ));
void HBT_scanf P((FILE *file , HBT *tree ));

/* infinity_norm.c */
double InfinityNorm P((Vector *x ));

/* innerprod.c */
double InnerProd P((Vector *x , Vector *y ));

/* inputRF.c */
void InputRF P((GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata ));
PFModule *InputRFInitInstanceXtra P((Grid *grid , double *temp_data ));
void InputRFFreeInstanceXtra P((void ));
PFModule *InputRFNewPublicXtra P((char *geom_name ));
void InputRFFreePublicXtra P((void ));
int InputRFSizeOfTempData P((void ));

/* input_database.c */
void IDB_Print P((FILE *file , IDB_Entry *entry ));
int IDB_Compare P((IDB_Entry *a , IDB_Entry *b ));
void IDB_Free P((IDB_Entry *a ));
IDB_Entry *IDB_NewEntry P((char *key , char *value ));
IDB *IDB_NewDB P((char *filename ));
void IDB_FreeDB P((IDB *database ));
void IDB_PrintUsage P((FILE *file , IDB *database ));
char *IDB_GetString P((IDB *database , char *key ));
char *IDB_GetStringDefault P((IDB *database , char *key , char *default_value ));
double IDB_GetDoubleDefault P((IDB *database , char *key , double default_value ));
double IDB_GetDouble P((IDB *database , char *key ));
int IDB_GetIntDefault P((IDB *database , char *key , int default_value ));
int IDB_GetInt P((IDB *database , char *key ));
NameArray NA_NewNameArray P((char *string ));
int NA_AppendToArray P((NameArray name_array , char *string ));
void NA_FreeNameArray P((NameArray name_array ));
int NA_NameToIndex P((NameArray name_array , char *name ));
char *NA_IndexToName P((NameArray name_array , int index ));
int NA_Sizeof P((NameArray name_array ));
void InputError P((char *format , char *s1 , char *s2 ));

/*kinsol_function_eval.c*/
int KINSolFunctionEval P((N_Vector multispecies, N_Vector fval , void *current_state ));

/* kinsol_nonlin_solver.c */
int KINSolInitPC P((N_Vector multispecies, N_Vector uscale , N_Vector fval , N_Vector fscale , void *current_state, N_Vector vtemp1 , N_Vector vtemp2 ));
int KINSolCallPC P((N_Vector multispecies, N_Vector uscale , N_Vector fval , N_Vector fscale , N_Vector vtem , void *current_state, N_Vector ftem ));
void PrintFinalStats P((FILE *out_file ));
int KinsolNonlinSolver P((N_Vector multispecies, Vector *density , Vector *old_density , Vector *heat_capacity_water, Vector *heat_capacity_rock, Vector *viscosity, Vector *old_viscosity, Vector *saturation , Vector *old_saturation , double t , double dt , ProblemData *problem_data, Vector *old_pressure, Vector *old_temperature, double *outflow, Vector *evap_trans, Vector *clm_energy_source, Vector *forc_t, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity ));
PFModule *KinsolNonlinSolverInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , double *temp_data ));
void KinsolNonlinSolverFreeInstanceXtra P((void ));
PFModule *KinsolNonlinSolverNewPublicXtra P((void ));
void KinsolNonlinSolverFreePublicXtra P((void ));
int KinsolNonlinSolverSizeOfTempData P((void ));

/* kinsol_matvec.c */
int KINSolMatVec P((N_Vector x , N_Vector y , N_Vector multispecies, booleantype *recompute , void *current_state ));

/* kinsol_pc_pressure.c */
void KinsolPCPressure P((Vector *rhs ));
PFModule *KinsolPCPressureInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , double *temp_data , Vector *pressure , Vector *temperature, Vector *saturation , Vector *density , Vector *viscosity, double dt , double time));
void KinsolPCPressureFreeInstanceXtra P((void ));
PFModule *KinsolPCPressureNewPublicXtra P((char *name , char *pc_name ));
void KinsolPCPressureFreePublicXtra P((void ));
int KinsolPCPressureSizeOfTempData P((void ));

/* kinsol_pc_temperature.c */
void KinsolPCTemperature P((Vector *rhs ));
PFModule *KinsolPCTemperatureInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , double *temp_data , Vector *pressure , Vector *temperature, Vector *saturation , Vector *density , Vector *heat_capacity_water, Vector *heat_capacity_rock, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, double dt , double time));
void KinsolPCTemperatureFreeInstanceXtra P((void ));
PFModule *KinsolPCTemperatureNewPublicXtra P((char *name , char *pc_name ));
void KinsolPCTemperatureFreePublicXtra P((void ));
int KinsolPCTemperatureSizeOfTempData P((void ));

/* l2_error_norm.c */
void L2ErrorNorm P((double time , Vector *pressure , ProblemData *problem_data , double *l2_error_norm ));
PFModule *L2ErrorNormInitInstanceXtra P((void ));
void L2ErrorNormFreeInstanceXtra P((void ));
PFModule *L2ErrorNormNewPublicXtra P((void ));
void L2ErrorNormFreePublicXtra P((void ));
int L2ErrorNormSizeOfTempData P((void ));

/* line_process.c */
void LineProc P((double *Z , double phi , double theta , double dzeta , int izeta , int nzeta , double Kmax , double dK ));

/* logging.c */
void NewLogging P((void ));
void FreeLogging P((void ));
FILE *OpenLogFile P((char *module_name ));
int CloseLogFile P((FILE *log_file ));

/* matdiag_scale.c */
void MatDiagScale P((Vector *x , Matrix *A , Vector *b , int flag ));
PFModule *MatDiagScaleInitInstanceXtra P((Grid *grid ));
void MatDiagScaleFreeInstanceXtra P((void ));
PFModule *MatDiagScaleNewPublicXtra P((char *name ));
void MatDiagScaleFreePublicXtra P((void ));
int MatDiagScaleSizeOfTempData P((void ));

/* matrix.c */
Stencil *NewStencil P((int shape [][3 ], int sz ));
CommPkg *NewMatrixUpdatePkg P((Matrix *matrix , Stencil *ghost ));
CommHandle *InitMatrixUpdate P((Matrix *matrix ));
void FinalizeMatrixUpdate P((CommHandle *handle ));
Matrix *NewMatrix P((Grid *grid , SubregionArray *range , Stencil *stencil , int symmetry , Stencil *ghost ));
void FreeStencil P((Stencil *stencil ));
void FreeMatrix P((Matrix *matrix ));
void InitMatrix P((Matrix *A , double value ));

/* matvec.c */
void Matvec P((double alpha , Matrix *A , Vector *x , double beta , Vector *y ));

/* max_field_value.c */
double MaxFieldValue P((Vector *field , Vector *phi , int dir ));
double MaxPhaseFieldValue P((Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *phi ));
double MaxTotalFieldValue P((Problem *problem , EvalStruct *eval_struct , Vector *saturation , Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *beta , Vector *phi ));

/* mg_semi.c */
void MGSemi P((Vector *x , Vector *b , double tol , int zero ));
void SetupCoarseOps P((Matrix **A_l , Matrix **P_l , int num_levels , SubregionArray **f_sra_l , SubregionArray **c_sra_l ));
PFModule *MGSemiInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void MGSemiFreeInstanceXtra P((void ));
PFModule *MGSemiNewPublicXtra P((char *name ));
void MGSemiFreePublicXtra P((void ));
int MGSemiSizeOfTempData P((void ));

/* mg_semi_prolong.c */
void MGSemiProlong P((Matrix *A_f , Vector *e_f , Vector *e_c , Matrix *P , SubregionArray *f_sr_array , SubregionArray *c_sr_array , ComputePkg *compute_pkg , CommPkg *e_f_comm_pkg ));
ComputePkg *NewMGSemiProlongComputePkg P((Grid *grid , Stencil *stencil , int sx , int sy , int sz , int c_index , int f_index ));

/* mg_semi_restrict.c */
void MGSemiRestrict P((Matrix *A_f , Vector *r_f , Vector *r_c , Matrix *P , SubregionArray *f_sr_array , SubregionArray *c_sr_array , ComputePkg *compute_pkg , CommPkg *r_f_comm_pkg ));
ComputePkg *NewMGSemiRestrictComputePkg P((Grid *grid , Stencil *stencil , int sx , int sy , int sz , int c_index , int f_index ));

/* nvector_parflow.c */
N_Vector N_VNew_Parflow P((Grid *grid));
N_Vector N_VNewEmpty_Parflow P((int nspecies));
void N_VPrint P((N_Vector x));
N_Vector N_VCloneEmpty_Parflow(N_Vector w);
N_Vector N_VClone_Parflow(N_Vector w);
void N_VDestroy_Parflow(N_Vector v);
void N_VSpace_Parflow(N_Vector v, long int *lrw, long int *liw);
realtype *N_VGetArrayPointer_Parflow(N_Vector v);
void N_VSetArrayPointer_Parflow(realtype *v_data, N_Vector v);
void N_VLinearSum_Parflow(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z);
void N_VConst_Parflow(realtype c, N_Vector z);
void N_VProd_Parflow(N_Vector x, N_Vector y, N_Vector z);
void N_VDiv_Parflow(N_Vector x, N_Vector y, N_Vector z);
void N_VScale_Parflow(realtype c, N_Vector x, N_Vector z);
void N_VAbs_Parflow(N_Vector x, N_Vector y);
void N_VInv_Parflow(N_Vector x, N_Vector y);
void N_VAddConst_Parflow(N_Vector x, realtype b, N_Vector z);
realtype N_VDotProd_Parflow(N_Vector x, N_Vector y);
realtype N_VMaxNorm_Parflow(N_Vector x);
realtype N_VWrmsNorm_Parflow(N_Vector x, N_Vector w);
realtype N_VWrmsNormMask_Parflow(N_Vector x, N_Vector w, N_Vector id);
realtype N_VMin_Parflow(N_Vector x);
realtype N_VWL2Norm_Parflow(N_Vector x, N_Vector w);
realtype N_VL1Norm_Parflow(N_Vector x);
void N_VCompare_Parflow(realtype c, N_Vector x, N_Vector z);
booleantype N_VInvTest_Parflow(N_Vector x, N_Vector z);
booleantype N_VConstrMask_Parflow(N_Vector c, N_Vector x, N_Vector m);
realtype N_VMinQuotient_Parflow(N_Vector num, N_Vector denom);
void N_VVector_Parflow(N_Vector templ, Vector *pressure, Vector *temperature);
N_Vector N_VMake_Parflow(Vector *pressure, Vector *temperature);

/* new_endpts.c */
void NewEndpts P((double *alpha , double *beta , double *pp , int *size_ptr , int n , double *a_ptr , double *b_ptr , double *cond_ptr , double ereps ));

/* press_function_eval.c */
void PressFunctionEval P((Vector *pressure, Vector *fval , ProblemData *problem_data, Vector *temperature, Vector *saturation , Vector *old_saturation , Vector *density , Vector *old_density, Vector *viscosity, double dt , double time, Vector *old_pressure, double *outflow , Vector *evap_trans, Vector *ovrl_bc_flx, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity));
PFModule *PressFunctionEvalInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void PressFunctionEvalFreeInstanceXtra P((void ));
PFModule *PressFunctionEvalNewPublicXtra P((void ));
void PressFunctionEvalFreePublicXtra P((void ));
int PressFunctionEvalSizeOfTempData P((void ));

/* temp_function_eval.c */
void TempFunctionEval P((Vector *temperature, Vector *fval , ProblemData *problem_data , Vector *pressure, Vector *old_pressure, Vector *saturation , Vector *old_saturation , Vector *density , Vector *old_density , Vector *heat_capacity_water, Vector *heat_capacity_rock, Vector *viscosity, double dt , double time, Vector *old_temperature, Vector *evap_trans, Vector *energy_source, Vector *forc_t, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity));
PFModule *TempFunctionEvalInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void TempFunctionEvalFreeInstanceXtra P((void ));
PFModule *TempFunctionEvalNewPublicXtra P((void ));
void TempFunctionEvalFreePublicXtra P((void ));
int TempFunctionEvalSizeOfTempData P((void ));

/* nodiag_scale.c */
void NoDiagScale P((Vector *x , Matrix *A , Vector *b , int flag ));
PFModule *NoDiagScaleInitInstanceXtra P((Grid *grid ));
void NoDiagScaleFreeInstanceXtra P((void ));
PFModule *NoDiagScaleNewPublicXtra P((char *name ));
void NoDiagScaleFreePublicXtra P((void ));
int NoDiagScaleSizeOfTempData P((void ));

/* parflow.c */
int main P((int argc , char *argv []));

/* pcg.c */
void PCG P((Vector *x , Vector *b , double tol , int zero ));
PFModule *PCGInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void PCGFreeInstanceXtra P((void ));
PFModule *PCGNewPublicXtra P((char *name ));
void PCGFreePublicXtra P((void ));
int PCGSizeOfTempData P((void ));

/* permeability_face.c */
void PermeabilityFace P((Vector *zperm , Vector *permeability ));
PFModule *PermeabilityFaceInitInstanceXtra P((Grid *z_grid ));
void PermeabilityFaceFreeInstanceXtra P((void ));
PFModule *PermeabilityFaceNewPublicXtra P((void ));
void PermeabilityFaceFreePublicXtra P((void ));
int PermeabilityFaceSizeOfTempData P((void ));

/* perturb_lb.c */
void PerturbSystem P((Lattice *lattice , Problem *problem ));

/* pf_module.c */
PFModule *NewPFModule P((void *call , void *init_instance_xtra , void *free_instance_xtra , void *new_public_xtra , void *free_public_xtra , void *sizeof_temp_data , void *instance_xtra , void *public_xtra ));
PFModule *DupPFModule P((PFModule *pf_module ));
void FreePFModule P((PFModule *pf_module ));

/* pf_pfmg.c */
void PFMG P((Vector *soln , Vector *rhs , double tol , int zero ));
PFModule *PFMGInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *pf_matrix , double *temp_data ));
void PFMGFreeInstanceXtra P((void ));
PFModule *PFMGNewPublicXtra P((char *name ));
void PFMGFreePublicXtra P((void ));
int PFMGSizeOfTempData P((void ));

/* pf_smg.c */
void SMG P((Vector *soln , Vector *rhs , double tol , int zero ));
PFModule *SMGInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *pf_matrix , double *temp_data ));
void SMGFreeInstanceXtra P((void ));
PFModule *SMGNewPublicXtra P((char *name ));
void SMGFreePublicXtra P((void ));
int SMGSizeOfTempData P((void ));

/* pfield.c */
void PField P((Grid *grid , GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata , Statistics *stats ));

/* pgsRF.c */
void PGSRF P((GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata ));
PFModule *PGSRFInitInstanceXtra P((Grid *grid , double *temp_data ));
void PGSRFFreeInstanceXtra P((void ));
PFModule *PGSRFNewPublicXtra P((char *geom_name ));
void PGSRFFreePublicXtra P((void ));
int PGSRFSizeOfTempData P((void ));

/* phase_velocity_face.c */
void PhaseVelocityFace P((Vector *xvel , Vector *yvel , Vector *zvel , ProblemData *problem_data , Vector *pressure , Vector **saturations , int phase ));
PFModule *PhaseVelocityFaceInitInstanceXtra P((Problem *problem , Grid *grid , Grid *x_grid , Grid *y_grid , Grid *z_grid , double *temp_data ));
void PhaseVelocityFaceFreeInstanceXtra P((void ));
PFModule *PhaseVelocityFaceNewPublicXtra P((void ));
void PhaseVelocityFaceFreePublicXtra P((void ));
int PhaseVelocityFaceSizeOfTempData P((void ));

/* ppcg.c */
void PPCG P((Vector *x , Vector *b , double tol , int zero ));
PFModule *PPCGInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void PPCGFreeInstanceXtra P((void ));
PFModule *PPCGNewPublicXtra P((char *name ));
void PPCGFreePublicXtra P((void ));
int PPCGSizeOfTempData P((void ));

/* printgrid.c */
void PrintGrid P((char *filename , Grid *grid ));

/* printmatrix.c */
void PrintSubmatrixAll P((amps_File file , Submatrix *submatrix , Stencil *stencil ));
void PrintMatrixAll P((char *filename , Matrix *A ));
void PrintSubmatrix P((amps_File file , Submatrix *submatrix , Subregion *subregion , Stencil *stencil ));
void PrintMatrix P((char *filename , Matrix *A ));
void PrintSortMatrix P((char *filename , Matrix *A , int all ));

/* printvector.c */
void PrintSubvectorAll P((amps_File file , Subvector *subvector ));
void PrintVectorAll P((char *filename , Vector *v ));
void PrintSubvector P((amps_File file , Subvector *subvector , Subgrid *subgrid ));
void PrintVector P((char *filename , Vector *v ));

/* problem.c */
Problem *NewProblem P((int solver ));
void FreeProblem P((Problem *problem , int solver ));
ProblemData *NewProblemData P((Grid *grid, Grid *grid2d ));
void FreeProblemData P((ProblemData *problem_data ));

/* problem_bc.c */
BCStruct *NewBCStruct P((SubgridArray *subgrids , GrGeomSolid *gr_domain , int num_patches , int *patch_indexes , int *bc_types , double ***values ));
void FreeBCStruct P((BCStruct *bc_struct ));

/* problem_bc_internal.c */
void BCInternal P((Problem *problem , ProblemData *problem_data , Matrix *A , Vector *f , double time ));
PFModule *BCInternalInitInstanceXtra P((void ));
void BCInternalFreeInstanceXtra P((void ));
PFModule *BCInternalNewPublicXtra P((void ));
void BCInternalFreePublicXtra P((void ));
int BCInternalSizeOfTempData P((void ));

/* problem_bc_phase_saturation.c */
void BCPhaseSaturation P((Vector *saturation , int phase , GrGeomSolid *gr_domain ));
PFModule *BCPhaseSaturationInitInstanceXtra P((void ));
void BCPhaseSaturationFreeInstanceXtra P((void ));
PFModule *BCPhaseSaturationNewPublicXtra P((int num_phases ));
void BCPhaseSaturationFreePublicXtra P((void ));
int BCPhaseSaturationSizeOfTempData P((void ));

/* problem_bc_pressure.c */
BCStruct *BCPressure P((ProblemData *problem_data , Grid *grid , GrGeomSolid *gr_domain , double time ));
PFModule *BCPressureInitInstanceXtra P((Problem *problem ));
void BCPressureFreeInstanceXtra P((void ));
PFModule *BCPressureNewPublicXtra P((int num_phases ));
void BCPressureFreePublicXtra P((void ));
int BCPressureSizeOfTempData P((void ));

/* problem_bc_temperature.c */
BCStruct *BCTemperature P((ProblemData *problem_data , Grid *grid , GrGeomSolid *gr_domain , double time ));
PFModule *BCTemperatureInitInstanceXtra P((Problem *problem ));
void BCTemperatureFreeInstanceXtra P((void ));
PFModule *BCTemperatureNewPublicXtra P((int num_phases ));
void BCTemperatureFreePublicXtra P((void ));
int BCTemperatureSizeOfTempData P((void ));

/* problem_capillary_pressure.c */
void CapillaryPressure P((Vector *capillary_pressure , int phase_i , int phase_j , ProblemData *problem_data , Vector *phase_saturation ));
PFModule *CapillaryPressureInitInstanceXtra P((void ));
void CapillaryPressureFreeInstanceXtra P((void ));
PFModule *CapillaryPressureNewPublicXtra P((int num_phases ));
void CapillaryPressureFreePublicXtra P((void ));
int CapillaryPressureSizeOfTempData P((void ));

/* problem_domain.c */
void Domain P((ProblemData *problem_data ));
PFModule *DomainInitInstanceXtra P((Grid *grid ));
void DomainFreeInstanceXtra P((void ));
PFModule *DomainNewPublicXtra P((void ));
void DomainFreePublicXtra P((void ));
int DomainSizeOfTempData P((void ));

/* problem_eval.c */
EvalStruct *NewEvalStruct P((Problem *problem ));
void FreeEvalStruct P((EvalStruct *eval_struct ));

/* problem_geometries.c */
void Geometries P((ProblemData *problem_data ));
PFModule *GeometriesInitInstanceXtra P((Grid *grid));
void GeometriesFreeInstanceXtra P((void ));
PFModule *GeometriesNewPublicXtra P((void ));
void GeometriesFreePublicXtra P((void ));
int GeometriesSizeOfTempData P((void ));

/* problem_phase_heat_capacity.c */
PFModule *PhaseHeatCapacityInitInstanceXtra P((void ));
/* void PhaseHeatCapacity P((int phase, Vector *heat_capacity, Vector *heat_capacity_rock, ProblemData *problem_data));
*/
void PhaseHeatCapacity P((int phase, Vector *heat_capacity, ProblemData *problem_data));
void PhaseHeatCapacityFreeInstanceXtra P((void ));
PFModule *PhaseHeatCapacityNewPublicXtra P((int num_phases));
void PhaseHeatCapacityFreePublicXtra P((void ));
int PhaseHeatCapacitySizeOfTempData P((void ));

/* problem_ic_phase_concen.c */
void ICPhaseConcen P((Vector *ic_phase_concen , int phase , int contaminant , ProblemData *problem_data ));
PFModule *ICPhaseConcenInitInstanceXtra P((void ));
void ICPhaseConcenFreeInstanceXtra P((void ));
PFModule *ICPhaseConcenNewPublicXtra P((int num_phases , int num_contaminants ));
void ICPhaseConcenFreePublicXtra P((void ));
int ICPhaseConcenSizeOfTempData P((void ));

/* problem_ic_phase_pressure.c */
void ICPhasePressure P((Vector *ic_pressure , Vector *ic_temperature, Vector *mask, ProblemData *problem_data , Problem *problem ));
PFModule *ICPhasePressureInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void ICPhasePressureFreeInstanceXtra P((void ));
PFModule *ICPhasePressureNewPublicXtra P((void ));
void ICPhasePressureFreePublicXtra P((void ));
int ICPhasePressureSizeOfTempData P((void ));

/* problem_ic_phase_temperature.c */
void ICPhaseTemperature P((Vector *ic_temperature, ProblemData *problem_data , Problem *problem ));
PFModule *ICPhaseTemperatureInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void ICPhaseTemperatureFreeInstanceXtra P((void ));
PFModule *ICPhaseTemperatureNewPublicXtra P((void ));
void ICPhaseTemperatureFreePublicXtra P((void ));
int ICPhaseTemperatureSizeOfTempData P((void ));

/* problem_mannings.c */
void Mannings P((ProblemData *problem_data, Vector *mann));
PFModule *ManningsInitInstanceXtra P((Grid *grid));
void ManningsFreeInstanceXtra P((void ));
PFModule *ManningsNewPublicXtra P((void));
void ManningsFreePublicXtra P((void ));
int ManningsSizeOfTempData P((void ));

/* problem_spec_storage.c */
void SpecStorage P((ProblemData *problem_data, Vector *specific_storage ));
PFModule *SpecStorageInitInstanceXtra P((void ));
void SpecStorageFreeInstanceXtra P((void ));
PFModule *SpecStorageNewPublicXtra P((void ));
void SpecStorageFreePublicXtra P((void ));
int SpecStorageSizeOfTempData P((void ));

/* problem_ic_phase_satur.c */
void ICPhaseSatur P((Vector *ic_phase_satur , int phase , ProblemData *problem_data ));
PFModule *ICPhaseSaturInitInstanceXtra P((void ));
void ICPhaseSaturFreeInstanceXtra P((void ));
PFModule *ICPhaseSaturNewPublicXtra P((int num_phases ));
void ICPhaseSaturFreePublicXtra P((void ));
int ICPhaseSaturSizeOfTempData P((void ));

/* problem_phase_density.c */
void PhaseDensity P((int phase , Vector *phase_pressure , Vector *temperature, Vector *density_v , double *pressure_d , double *density_d , int fcn ));
PFModule *PhaseDensityInitInstanceXtra P((void ));
void PhaseDensityFreeInstanceXtra P((void ));
PFModule *PhaseDensityNewPublicXtra P((int num_phases ));
void PhaseDensityFreePublicXtra P((void ));
int PhaseDensitySizeOfTempData P((void ));

/* problem_phase_internal_energy.c */
void InternalEnergyDensity P((int phase, Vector *pressure, Vector *temperature, Vector *energy, Vector *density, int fcn ));
PFModule *InternalEnergyDensityInitInstanceXtra P((void ));
void InternalEnergyDensityFreeInstanceXtra P((void ));
PFModule *InternalEnergyDensityNewPublicXtra P((int num_phases ));
void InternalEnergyDensityFreePublicXtra P((void ));
int InternalEnergyDensitySizeOfTempData P((void ));

/* problem_phase_mobility.c */
void PhaseMobility P((Vector *phase_mobility_x , Vector *phase_mobility_y , Vector *phase_mobility_z , Vector *perm_x , Vector *perm_y , Vector *perm_z , int phase , Vector *phase_saturation , double phase_viscosity ));
PFModule *PhaseMobilityInitInstanceXtra P((void ));
void PhaseMobilityFreeInstanceXtra P((void ));
PFModule *PhaseMobilityNewPublicXtra P((int num_phases ));
void PhaseMobilityFreePublicXtra P((void ));
int PhaseMobilitySizeOfTempData P((void ));

/* problem_phase_rel_perm.c */
void PhaseRelPerm P((Vector *phase_rel_perm , Vector *phase_pressure , Vector *phase_density , double gravity , ProblemData *problem_data , int fcn ));
PFModule *PhaseRelPermInitInstanceXtra P((Grid *grid , double *temp_data ));
void PhaseRelPermFreeInstanceXtra P((void ));
PFModule *PhaseRelPermNewPublicXtra P((void ));
void PhaseRelPermFreePublicXtra P((void ));
int PhaseRelPermSizeOfTempData P((void ));

/* problem_phase_source.c */
void PhaseSource P((Vector *phase_source , Vector *phase_temperature, Problem *problem , ProblemData *problem_data , double time ));
PFModule *PhaseSourceInitInstanceXtra P((Grid *grid));
void PhaseSourceFreeInstanceXtra P((void ));
PFModule *PhaseSourceNewPublicXtra P((void));
void PhaseSourceFreePublicXtra P((void ));
int PhaseSourceSizeOfTempData P((void ));

/* problem_temp_source.c */
void TempSource P((Vector *temp_source , Problem *problem , ProblemData *problem_data , double time ));
PFModule *TempSourceInitInstanceXtra P((Grid *grid));
void TempSourceFreeInstanceXtra P((void ));
PFModule *TempSourceNewPublicXtra P((void));
void TempSourceFreePublicXtra P((void ));
int TempSourceSizeOfTempData P((void ));

/* problem_phase_viscosity.c */
void PhaseViscosity P((int phase , Vector *pressure , Vector *temperature , Vector *viscosity, int fcn ));
PFModule *PhaseViscosityInitInstanceXtra P((void ));
void PhaseViscosityFreeInstanceXtra P((void ));
PFModule *PhaseViscosityNewPublicXtra P((int num_phases ));
void PhaseViscosityFreePublicXtra P((void ));
int PhaseViscositySizeOfTempData P((void ));

/* problem_porosity.c */
void Porosity P((ProblemData *problem_data , Vector *porosity , int num_geounits , GeomSolid **geounits , GrGeomSolid **gr_geounits ));
PFModule *PorosityInitInstanceXtra P((Grid *grid , double *temp_data ));
void PorosityFreeInstanceXtra P((void ));
PFModule *PorosityNewPublicXtra P((void ));
void PorosityFreePublicXtra P((void ));
int PorositySizeOfTempData P((void ));

/* problem_retardation.c */
void Retardation P((Vector *solidmassfactor , int contaminant , ProblemData *problem_data ));
PFModule *RetardationInitInstanceXtra P((double *temp_data ));
void RetardationFreeInstanceXtra P((void ));
PFModule *RetardationNewPublicXtra P((int num_contaminants ));
void RetardationFreePublicXtra P((void ));
int RetardationSizeOfTempData P((void ));

/* problem_richards_bc_internal.c */
void RichardsBCInternal P((Problem *problem , ProblemData *problem_data , Vector *f , Matrix *A , double time , Vector *pressure , int fcn ));
PFModule *RichardsBCInternalInitInstanceXtra P((void ));
void RichardsBCInternalFreeInstanceXtra P((void ));
PFModule *RichardsBCInternalNewPublicXtra P((void ));
void RichardsBCInternalFreePublicXtra P((void ));
int RichardsBCInternalSizeOfTempData P((void ));

/* problem_saturation.c */
void Saturation P((Vector *phase_saturation , Vector *phase_pressure , Vector *phase_density , double gravity , ProblemData *problem_data , int fcn ));
PFModule *SaturationInitInstanceXtra P((Grid *grid , double *temp_data ));
void SaturationFreeInstanceXtra P((void ));
PFModule *SaturationNewPublicXtra P((void ));
void SaturationFreePublicXtra P((void ));
int SaturationSizeOfTempData P((void ));

/* problem_saturation_constitutive.c */
void SaturationConstitutive P((Vector **phase_saturations ));
PFModule *SaturationConstitutiveInitInstanceXtra P((Grid *grid ));
void SaturationConstitutiveFreeInstanceXtra P((void ));
PFModule *SaturationConstitutiveNewPublicXtra P((int num_phases ));
void SaturationConstitutiveFreePublicXtra P((void ));
int SaturationConstitutiveSizeOfTempData P((void ));

/* problem_thermal_conductivity.c */
void ThermalConductivity P((Vector *phase_thermalconductivity, Vector *phase_pressure, Vector *phase_saturation, double gravity, ProblemData *problem_data, int fcn ));
PFModule *ThermalConductivityInitInstanceXtra P((Grid *grid ));
void ThermalConductivityFreeInstanceXtra P((void ));
PFModule *ThermalConductivityNewPublicXtra P((void ));
void ThermalConductivityFreePublicXtra P((void ));
int ThermalConductivitySizeOfTempData P((void ));

/* problem_toposlope_x.c */
void XSlope P((ProblemData *problem_data, Vector *x_sl ));
PFModule *XSlopeInitInstanceXtra P((Grid *grid));
void XSlopeFreeInstanceXtra P((void ));
PFModule *XSlopeNewPublicXtra P((void));
void XSlopeFreePublicXtra P((void ));
int XSlopeSizeOfTempData P((void ));

/* problem_toposlope_y.c */
void YSlope P((ProblemData *problem_data, Vector *y_slope));
PFModule *YSlopeInitInstanceXtra P((Grid *grid));
void YSlopeFreeInstanceXtra P((void ));
PFModule *YSlopeNewPublicXtra P((void));
void YSlopeFreePublicXtra P((void ));
int YSlopeSizeOfTempData P((void ));

/* random.c */
void SeedRand P((int seed ));
double Rand P((void ));

/* ratqr.c */
int ratqr_ P((int *n , double *eps1 , double *d , double *e , double *e2 , int *m , double *w , int *ind , double *bd , int *type , int *idef , int *ierr ));
double epslon_ P((double *x ));

/* rb_GS_point.c */
void RedBlackGSPoint P((Vector *x , Vector *b , double tol , int zero ));
PFModule *RedBlackGSPointInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void RedBlackGSPointFreeInstanceXtra P((void ));
PFModule *RedBlackGSPointNewPublicXtra P((char *name ));
void RedBlackGSPointFreePublicXtra P((void ));
int RedBlackGSPointSizeOfTempData P((void ));

/* read_parflow_binary.c */
void ReadPFBinary_Subvector P((amps_File file , Subvector *subvector , Subgrid *subgrid ));
void ReadPFBinary P((char *filename , Vector *v ));

/* reg_from_stenc.c */
void ComputeRegFromStencil P((Region **dep_reg_ptr , Region **ind_reg_ptr , SubregionArray *cr_array , Region *send_reg , Region *recv_reg , Stencil *stencil ));
SubgridArray *GetGridNeighbors P((SubgridArray *subgrids , SubgridArray *all_subgrids , Stencil *stencil ));
void CommRegFromStencil P((Region **send_region_ptr , Region **recv_region_ptr , Grid *grid , Stencil *stencil ));

/* region.c */
Subregion *NewSubregion P((int ix , int iy , int iz , int nx , int ny , int nz , int sx , int sy , int sz , int rx , int ry , int rz , int process ));
SubregionArray *NewSubregionArray P((void ));
Region *NewRegion P((int size ));
void FreeSubregion P((Subregion *subregion ));
void FreeSubregionArray P((SubregionArray *subregion_array ));
void FreeRegion P((Region *region ));
Subregion *DuplicateSubregion P((Subregion *subregion ));
SubregionArray *DuplicateSubregionArray P((SubregionArray *subregion_array ));
Region *DuplicateRegion P((Region *region ));
void AppendSubregion P((Subregion *subregion , SubregionArray *sr_array ));
void DeleteSubregion P((SubregionArray *sr_array , int index ));
void AppendSubregionArray P((SubregionArray *sr_array_0 , SubregionArray *sr_array_1 ));

/* richards_jacobian_eval.c */
void RichardsJacobianEval P((Vector *pressure, Matrix **ptr_to_J , Vector *temperature, Vector *saturation , Vector *density , Vector *viscosity, ProblemData *problem_data , double dt , double time , int symm_part));
PFModule *RichardsJacobianEvalInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data , int symmetric_jac ));
void RichardsJacobianEvalFreeInstanceXtra P((void ));
PFModule *RichardsJacobianEvalNewPublicXtra P((void ));
void RichardsJacobianEvalFreePublicXtra P((void ));
int RichardsJacobianEvalSizeOfTempData P((void ));

/* temperature_jacobian_eval.c */
void TemperatureJacobianEval P((Vector *temperature, Matrix **ptr_to_J , Vector *pressure, Vector *saturation , Vector *density , Vector *heat_capacity_water, Vector *heat_capacity_rock, Vector *x_velocity, Vector *y_velocity, Vector *z_velocity, ProblemData *problem_data , double dt , double time , int symm_part));
PFModule *TemperatureJacobianEvalInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data , int symmetric_jac ));
void TemperatureJacobianEvalFreeInstanceXtra P((void ));
PFModule *TemperatureJacobianEvalNewPublicXtra P((void ));
void TemperatureJacobianEvalFreePublicXtra P((void ));
int TemperatureJacobianEvalSizeOfTempData P((void ));

/* sadvection_godunov.c */
void SatGodunov P((ProblemData *problem_data , int phase , Vector *old_saturation , Vector *new_saturation , Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *z_permeability , Vector *solid_mass_factor , double *viscosity , double *density , double gravity , double time , double deltat , int order ));
PFModule *SatGodunovInitInstanceXtra P((Problem *problem , Grid *grid , double *temp_data ));
void SatGodunovFreeInstanceXtra P((void ));
PFModule *SatGodunovNewPublicXtra P((void ));
void SatGodunovFreePublicXtra P((void ));
int SatGodunovSizeOfTempData P((void ));

/* scale.c */
void Scale P((double alpha , Vector *y ));

/* select_time_step.c */
void SelectTimeStep P((double *dt , char *dt_info , double time , Problem *problem , ProblemData *problem_data ));
PFModule *SelectTimeStepInitInstanceXtra P((void ));
void SelectTimeStepFreeInstanceXtra P((void ));
PFModule *SelectTimeStepNewPublicXtra P((void ));
void SelectTimeStepFreePublicXtra P((void ));
int SelectTimeStepSizeOfTempData P((void ));

/* set_problem_data.c */
void SetProblemData P((ProblemData *problem_data ));
PFModule *SetProblemDataInitInstanceXtra P((Problem *problem , Grid *grid , Grid *grid2d, double *temp_data ));
void SetProblemDataFreeInstanceXtra P((void ));
PFModule *SetProblemDataNewPublicXtra P((void ));
void SetProblemDataFreePublicXtra P((void ));
int SetProblemDataSizeOfTempData P((void ));

/* sim_shear.c */
double **SimShear P((double **shear_min_ptr , double **shear_max_ptr , GeomSolid *geom_solid , SubgridArray *subgrids , int type ));

/* solver.c */
void Solve P((void ));
void NewSolver P((void ));
void FreeSolver P((void ));

/* solver_impes.c */
void SolverImpes P((void ));
PFModule *SolverImpesInitInstanceXtra P((void ));
void SolverImpesFreeInstanceXtra P((void ));
PFModule *SolverImpesNewPublicXtra P((char *name ));
void SolverImpesFreePublicXtra P((void ));
int SolverImpesSizeOfTempData P((void ));

/* solver_lb.c */
void SolverDiffusion P((void ));
PFModule *SolverDiffusionInitInstanceXtra P((void ));
void SolverDiffusionFreeInstanceXtra P((void ));
PFModule *SolverDiffusionNewPublicXtra P((char *name ));
void SolverDiffusionFreePublicXtra P((void ));
int SolverDiffusionSizeOfTempData P((void ));

/* solver_richards.c */
void SolverRichards P((void ));
PFModule *SolverRichardsInitInstanceXtra P((void ));
void SolverRichardsFreeInstanceXtra P((void ));
PFModule *SolverRichardsNewPublicXtra P((char *name ));
void SolverRichardsFreePublicXtra P((void ));
int SolverRichardsSizeOfTempData P((void ));


/* subsrf_sim.c */
void SubsrfSim P((ProblemData *problem_data , Vector *perm_x , Vector *perm_y , Vector *perm_z , int num_geounits , GeomSolid **geounits , GrGeomSolid **gr_geounits ));
PFModule *SubsrfSimInitInstanceXtra P((Grid *grid , double *temp_data ));
void SubsrfSimFreeInstanceXtra P((void ));
PFModule *SubsrfSimNewPublicXtra P((void ));
void SubsrfSimFreePublicXtra P((void ));
int SubsrfSimSizeOfTempData P((void ));
void AdvanceRichards P((PFModule *this_module, 
		       double start_time,      
		       double stop_time,       
		       double dt,              
		       int compute_time_step,  
		       Vector *evap_trans,     
		       Vector **pressure_out,  
		       Vector **porosity_out,
			Vector **saturation_out));
void SetupRichards P((PFModule *this_module));

/* time_cycle_data.c */
TimeCycleData *NewTimeCycleData P((int number_of_cycles , int *number_of_intervals ));
void FreeTimeCycleData P((TimeCycleData *time_cycle_data ));
void PrintTimeCycleData P((TimeCycleData *time_cycle_data ));
int TimeCycleDataComputeIntervalNumber P((Problem *problem , double time , TimeCycleData *time_cycle_data , int cycle_number ));
double TimeCycleDataComputeNextTransition P((Problem *problem , double time , TimeCycleData *time_cycle_data ));
void ReadGlobalTimeCycleData P((void ));
void FreeGlobalTimeCycleData P((void ));

/* timing.c */
void NewTiming P((void ));
int RegisterTiming P((char *name ));
void PrintTiming P((void ));
void FreeTiming P((void ));

/* total_velocity_face.c */
void TotalVelocityFace P((Vector *xvel , Vector *yvel , Vector *zvel , ProblemData *problem_data , Vector *total_mobility_x , Vector *total_mobility_y , Vector *total_mobility_z , Vector *pressure , Vector **saturations ));
PFModule *TotalVelocityFaceInitInstanceXtra P((Problem *problem , Grid *grid , Grid *x_grid , Grid *y_grid , Grid *z_grid , double *temp_data ));
void TotalVelocityFaceFreeInstanceXtra P((void ));
PFModule *TotalVelocityFaceNewPublicXtra P((void ));
void TotalVelocityFaceFreePublicXtra P((void ));
int TotalVelocityFaceSizeOfTempData P((void ));

/* turning_bands.c */
void Turn P((Vector *field , void *vxtra ));
int InitTurn P((void ));
void *NewTurn P((char *geom_name ));
void FreeTurn P((void *xtra ));

/* turning_bandsRF.c */
void TurningBandsRF P((GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata ));
PFModule *TurningBandsRFInitInstanceXtra P((Grid *grid , double *temp_data ));
void TurningBandsRFFreeInstanceXtra P((void ));
PFModule *TurningBandsRFNewPublicXtra P((char *geom_name ));
void TurningBandsRFFreePublicXtra P((void ));
int TurningBandsRFSizeOfTempData P((void ));

/* usergrid_input.c */
Subgrid *ReadUserSubgrid P((void ));
Grid *ReadUserGrid P((void ));
void FreeUserGrid P((Grid *user_grid ));

/* vector.c */
CommPkg *NewVectorCommPkg P((Vector *vector , ComputePkg *compute_pkg ));
CommHandle *InitVectorUpdate P((Vector *vector , int update_mode ));
void FinalizeVectorUpdate P((CommHandle *handle ));
Vector *NewTempVector P((Grid *grid , int nc , int num_ghost ));
void SetTempVectorData P((Vector *vector , double *data ));
Vector *NewVector P((Grid *grid , int nc , int num_ghost ));
void FreeTempVector P((Vector *vector ));
void FreeVector P((Vector *vector ));
void InitVector P((Vector *v , double value ));
void InitVectorAll P((Vector *v , double value ));
void InitVectorInc P((Vector *v , double value , double inc ));
void InitVectorRandom P((Vector *v , long seed ));

/* vector_utilities.c */
void PFVLinearSum P((double a , Vector *x , double b , Vector *y , Vector *z ));
void PFVConstInit P((double c , Vector *z ));
void PFVProd P((Vector *x , Vector *y , Vector *z ));
void PFVDiv P((Vector *x , Vector *y , Vector *z ));
void PFVScale P((double c , Vector *x , Vector *z ));
void PFVAbs P((Vector *x , Vector *z ));
void PFVInv P((Vector *x , Vector *z ));
void PFVAddConst P((Vector *x , double b , Vector *z ));
double PFVDotProd P((Vector *x , Vector *y ));
double PFVMaxNorm P((Vector *x ));
double PFVWrmsNorm P((Vector *x , Vector *w ));
double PFVWL2Norm P((Vector *x , Vector *w ));
double PFVL1Norm P((Vector *x ));
double PFVMin P((Vector *x ));
double PFVMax P((Vector *x ));
int PFVConstrProdPos P((Vector *c , Vector *x ));
void PFVCompare P((double c , Vector *x , Vector *z ));
int PFVInvTest P((Vector *x , Vector *z ));
void PFVCopy P((Vector *x , Vector *y ));
void PFVSum P((Vector *x , Vector *y , Vector *z ));
void PFVDiff P((Vector *x , Vector *y , Vector *z ));
void PFVNeg P((Vector *x , Vector *z ));
void PFVScaleSum P((double c , Vector *x , Vector *y , Vector *z ));
void PFVScaleDiff P((double c , Vector *x , Vector *y , Vector *z ));
void PFVLin1 P((double a , Vector *x , Vector *y , Vector *z ));
void PFVLin2 P((double a , Vector *x , Vector *y , Vector *z ));
void PFVAxpy P((double a , Vector *x , Vector *y ));
void PFVScaleBy P((double a , Vector *x ));
int PFVConstrMask P((Vector *c, Vector *x, Vector *m));
double PFVMinQuotient P((Vector *num, Vector *denom));
void PFVVector P((Vector *specie, Vector *v));

/* w_jacobi.c */
void WJacobi P((Vector *x , Vector *b , double tol , int zero ));
PFModule *WJacobiInitInstanceXtra P((Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data ));
void WJacobiFreeInstanceXtra P((void ));
PFModule *WJacobiNewPublicXtra P((char *name ));
void WJacobiFreePublicXtra P((void ));
int WJacobiSizeOfTempData P((void ));

/* well.c */
WellData *NewWellData P((void ));
void FreeWellData P((WellData *well_data ));
void PrintWellData P((WellData *well_data , unsigned int print_mask ));
void WriteWells P((char *file_prefix , Problem *problem , WellData *well_data , double time , int write_header ));

/* well_package.c */
void WellPackage P((ProblemData *problem_data ));
PFModule *WellPackageInitInstanceXtra P((void ));
void WellPackageFreeInstanceXtra P((void ));
PFModule *WellPackageNewPublicXtra P((int num_phases , int num_contaminants ));
void WellPackageFreePublicXtra P((void ));
int WellPackageSizeOfTempData P((void ));

/* wells_lb.c */
void LBWells P((Lattice *lattice , Problem *problem , ProblemData *problem_data ));

/* write_parflow_binary.c */
long SizeofPFBinarySubvector P((Subvector *subvector , Subgrid *subgrid ));
void WritePFBinary_Subvector P((amps_File file , Subvector *subvector , Subgrid *subgrid ));
void WritePFBinary P((char *file_prefix , char *file_suffix , Vector *v ));
long SizeofPFSBinarySubvector P((Subvector *subvector , Subgrid *subgrid , double drop_tolerance ));
void WritePFSBinary_Subvector P((amps_File file , Subvector *subvector , Subgrid *subgrid , double drop_tolerance ));
void WritePFSBinary P((char *file_prefix , char *file_suffix , Vector *v , double drop_tolerance ));

#undef P
