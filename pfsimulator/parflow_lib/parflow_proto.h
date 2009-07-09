/* Header.c */

/* advection_godunov.c */
void Godunov (ProblemData *problem_data , int phase , int concentration , Vector *old_concentration , Vector *new_concentration , Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *solid_mass_factor , double time , double deltat , int order );
PFModule *GodunovInitInstanceXtra (Problem *problem , Grid *grid , double *temp_data );
void GodunovFreeInstanceXtra (void );
PFModule *GodunovNewPublicXtra (void );
void GodunovFreePublicXtra (void );
int GodunovSizeOfTempData (void );

/* axpy.c */
void Axpy (double alpha , Vector *x , Vector *y );

/* background.c */
Background *ReadBackground (void );
void FreeBackground (Background *background );
void SetBackgroundBounds (Background *background , Grid *grid );

/* bc_lb.c */
void LBInitializeBC (Lattice *lattice , Problem *problem , ProblemData *problem_data );

/* bc_pressure.c */
BCPressureData *NewBCPressureData (void );
void FreeBCPressureData (BCPressureData *bc_pressure_data );
void PrintBCPressureData (BCPressureData *bc_pressure_data );

/* bc_pressure_package.c */
void BCPressurePackage (ProblemData *problem_data );
PFModule *BCPressurePackageInitInstanceXtra (Problem *problem );
void BCPressurePackageFreeInstanceXtra (void );
PFModule *BCPressurePackageNewPublicXtra (int num_phases );
void BCPressurePackageFreePublicXtra (void );
int BCPressurePackageSizeOfTempData (void );

/* calc_elevations.c */
double **CalcElevations (GeomSolid *geom_solid , int ref_patch , SubgridArray *subgrids );

/* cghs.c */
void CGHS (Vector *x , Vector *b , double tol , int zero );
PFModule *CGHSInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void CGHSFreeInstanceXtra (void );
PFModule *CGHSNewPublicXtra (char *name );
void CGHSFreePublicXtra (void );
int CGHSSizeOfTempData (void );

/* char_vector.c */
CommPkg *NewCharVectorUpdatePkg (CharVector *charvector , int update_mode );
CommHandle *InitCharVectorUpdate (CharVector *charvector , int update_mode );
void FinalizeCharVectorUpdate (CommHandle *handle );
CharVector *NewTempCharVector (Grid *grid , int nc , int num_ghost );
void SetTempCharVectorData (CharVector *charvector , char *data );
CharVector *NewCharVector (Grid *grid , int nc , int num_ghost );
void FreeTempCharVector (CharVector *charvector );
void FreeCharVector (CharVector *charvector );
void InitCharVector (CharVector *v , int value );
void InitCharVectorAll (CharVector *v , int value );
void InitCharVectorInc (CharVector *v , int value , int inc );

/* chebyshev.c */
void Chebyshev (Vector *x , Vector *b , double tol , int zero , double ia , double ib , int num_iter );
PFModule *ChebyshevInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void ChebyshevFreeInstanceXtra (void );
PFModule *ChebyshevNewPublicXtra (char *name );
void ChebyshevFreePublicXtra (void );
int ChebyshevSizeOfTempData (void );

/* comm_pkg.c */
void ProjectRegion (Region *region , int sx , int sy , int sz , int ix , int iy , int iz );
Region *ProjectRBPoint (Region *region , int rb [4 ][3 ]);
void CreateComputePkgs (Grid *grid );
void FreeComputePkgs (Grid *grid );

/* communication.c */
int NewCommPkgInfo (Subregion *data_sr , Subregion *comm_sr , int index , int num_vars , int *loop_array );
CommPkg *NewCommPkg (Region *send_region , Region *recv_region , SubregionArray *data_space , int num_vars , double *data );
void FreeCommPkg (CommPkg *pkg );
// SGS what's up with this?
CommHandle *InitCommunication (CommPkg *comm_pkg );
void FinalizeCommunication (CommHandle *handle );

/* computation.c */
ComputePkg *NewComputePkg (Region *send_reg , Region *recv_reg , Region *dep_reg , Region *ind_reg );
void FreeComputePkg (ComputePkg *compute_pkg );

/* compute_maximums.c */
double ComputePhaseMaximum (double phase_u_max , double dx , double phase_v_max , double dy , double phase_w_max , double dz );
double ComputeTotalMaximum (Problem *problem , EvalStruct *eval_struct , double s_lower , double s_upper , double total_u_max , double dx , double total_v_max , double dy , double total_w_max , double beta_max , double dz );

/* compute_total_concentration.c */
double ComputeTotalConcen (GrGeomSolid *gr_domain , Grid *grid , Vector *substance );

/* constantRF.c */
void ConstantRF (GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata );
PFModule *ConstantRFInitInstanceXtra (Grid *grid , double *temp_data );
void ConstantRFFreeInstanceXtra (void );
PFModule *ConstantRFNewPublicXtra (char *geom_name );
void ConstantRFFreePublicXtra (void );
int ConstantRFSizeOfTempData (void );

/* constant_porosity.c */
void ConstantPorosity (GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field );
PFModule *ConstantPorosityInitInstanceXtra (Grid *grid , double *temp_data );
void ConstantPorosityFreeInstanceXtra (void );
PFModule *ConstantPorosityNewPublicXtra (char *geom_name );
void ConstantPorosityFreePublicXtra (void );
int ConstantPorositySizeOfTempData (void );

/* copy.c */
void Copy (Vector *x , Vector *y );

/* create_grid.c */
SubgridArray *GetGridSubgrids (SubgridArray *all_subgrids );
Grid *CreateGrid (Grid *user_grid );

/* diag_scale.c */
void DiagScale (Vector *x , Matrix *A , Vector *b , Vector *d );

/* diffuse_lb.c */
void DiffuseLB (Lattice *lattice , Problem *problem , int max_iterations , char *file_prefix );
void LatticeFlowInit (Lattice *lattice , Problem *problem );
double MaxVectorValue (Vector *field );
double MaxVectorDividend (Vector *field1 , Vector *field2 );

/* discretize_pressure.c */
void DiscretizePressure (Matrix **ptr_to_A , Vector **ptr_to_f , ProblemData *problem_data , double time , Vector *total_mobility_x , Vector *total_mobility_y , Vector *total_mobility_z , Vector **phase_saturations );
PFModule *DiscretizePressureInitInstanceXtra (Problem *problem , Grid *grid , double *temp_data );
void DiscretizePressureFreeInstanceXtra (void );
PFModule *DiscretizePressureNewPublicXtra (void );
void DiscretizePressureFreePublicXtra (void );
int DiscretizePressureSizeOfTempData (void );

/* distribute_usergrid.c */
SubgridArray *DistributeUserGrid (Grid *user_grid );

/* dpofa.c */
int dpofa_ (double *a , int *lda , int *n , int *info );
double ddot_ (int *n , double *dx , int *incx , double *dy , int *incy );

/* dposl.c */
int dposl_ (double *a , int *lda , int *n , double *b );
int daxpy_ (int *n , double *da , double *dx , int *incx , double *dy , int *incy );

/* gauinv.c */
int gauinv_ (double *p , double *xp , int *ierr );

/* general.c */
char *malloc_chk (int size , char *file , int line );
char *calloc_chk (int count , int elt_size , char *file , int line );
int Exp2 (int p );
void printMemoryInfo(FILE *log_file);
void recordMemoryInfo();
void printMaxMemory(FILE *log_file);

/* geom_t_solid.c */
GeomTSolid *GeomNewTSolid (GeomTIN *surface , int **patches , int num_patches , int *num_patch_triangles );
void GeomFreeTSolid (GeomTSolid *solid );
int GeomReadTSolids (GeomTSolid ***solids_data_ptr , char *geom_input_name );
GeomTSolid *GeomTSolidFromBox (double xl , double yl , double zl , double xu , double yu , double zu );

/* geometry.c */
GeomVertexArray *GeomNewVertexArray (GeomVertex **vertices , int nV );
void GeomFreeVertexArray (GeomVertexArray *vertex_array );
GeomTIN *GeomNewTIN (GeomVertexArray *vertex_array , GeomTriangle **triangles , int nT );
void GeomFreeTIN (GeomTIN *surface );
GeomSolid *GeomNewSolid (void *data , int type );
void GeomFreeSolid (GeomSolid *solid );
int GeomReadSolids (GeomSolid ***solids_ptr , char *geom_input_name , int type );
GeomSolid *GeomSolidFromBox (double xl , double yl , double zl , double xu , double yu , double zu , int type );
void IntersectLineWithTriangle (unsigned int line_direction , double coord_0 , double coord_1 , double v0_x , double v0_y , double v0_z , double v1_x , double v1_y , double v1_z , double v2_x , double v2_y , double v2_z , int *intersects , double *point , int *normal_component );

/* globals.c */
void NewGlobals (char *run_name );
void FreeGlobals (void );
void LogGlobals (void );

/* grgeom_list.c */
ListMember *NewListMember (double value , int normal_component , int triangle_id );
void FreeListMember (ListMember *member );
void ListInsert (ListMember **head , ListMember *member );
int ListDelete (ListMember **head , ListMember *member );
ListMember *ListSearch (ListMember *head , double value , int normal_component , int triangle_id );
ListMember *ListValueSearch (ListMember *head , double value );
ListMember *ListValueNormalComponentSearch (ListMember *head , double value , int normal_component );
ListMember *ListTriangleIDSearch (ListMember *head , int triangle_id );
void ListFree (ListMember **head );
int ListLength (ListMember *head );
void ListPrint (ListMember *head );

/* grgeom_octree.c */
int GrGeomCheckOctree (GrGeomOctree *grgeom_octree );
void GrGeomFixOctree (GrGeomOctree *grgeom_octree , GrGeomOctree **patch_octrees , int num_patches , int level , int num_indices );
GrGeomOctree *GrGeomNewOctree (void );
void GrGeomNewOctreeChildren (GrGeomOctree *grgeom_octree );
void GrGeomFreeOctree (GrGeomOctree *grgeom_octree );
GrGeomOctree *GrGeomOctreeFind (int *new_level , GrGeomOctree *grgeom_octree_root , int ix , int iy , int iz , int level );
GrGeomOctree *GrGeomOctreeAddCell (GrGeomOctree *grgeom_octree_root , unsigned int cell , int ix , int iy , int iz , int level );
GrGeomOctree *GrGeomOctreeAddFace (GrGeomOctree *grgeom_octree_root , int line_direction , int cell_index0 , int cell_index1 , int face_index , int extent_lower , int extent_upper , int level , int normal_in_direction );
void GrGeomOctreeFromTIN (GrGeomOctree **solid_octree_ptr , GrGeomOctree ***patch_octrees_ptr , GeomTIN *solid , int **patches , int num_patches , int *num_patch_triangles , GrGeomExtentArray *extent_array , double xlower , double ylower , double zlower , double xupper , double yupper , double zupper , int min_level , int max_level );
void GrGeomOctreeFromInd (GrGeomOctree **solid_octree_ptr , Vector *indicator_field , int indicator , double xlower , double ylower , double zlower , double xupper , double yupper , double zupper , int octree_bg_level , int octree_ix , int octree_iy , int octree_iz );
void GrGeomPrintOctreeStruc (amps_File file , GrGeomOctree *grgeom_octree );
int GrGeomPrintOctreeLevel (amps_File file , GrGeomOctree *grgeom_octree , int level , int current_level );
void GrGeomPrintOctree (char *filename , GrGeomOctree *grgeom_octree_root );
void GrGeomPrintOctreeCells (char *filename , GrGeomOctree *octree , int last_level );
void GrGeomOctreeFree (GrGeomOctree *grgeom_octree_root );

/* grgeometry.c */
int GrGeomGetOctreeInfo (double *xlp , double *ylp , double *zlp , double *xup , double *yup , double *zup , int *ixp , int *iyp , int *izp );
GrGeomExtentArray *GrGeomNewExtentArray (GrGeomExtents *extents , int size );
void GrGeomFreeExtentArray (GrGeomExtentArray *extent_array );
GrGeomExtentArray *GrGeomCreateExtentArray (SubgridArray *subgrids , int xl_ghost , int xu_ghost , int yl_ghost , int yu_ghost , int zl_ghost , int zu_ghost );
GrGeomSolid *GrGeomNewSolid (GrGeomOctree *data , GrGeomOctree **patches , int num_patches , int octree_bg_level , int octree_ix , int octree_iy , int octree_iz );
void GrGeomFreeSolid (GrGeomSolid *solid );
void GrGeomSolidFromInd (GrGeomSolid **solid_ptr , Vector *indicator_field , int indicator );
void GrGeomSolidFromGeom (GrGeomSolid **solid_ptr , GeomSolid *geom_solid , GrGeomExtentArray *extent_array );

/* grid.c */
Grid *NewGrid (SubgridArray *subgrids , SubgridArray *all_subgrids );
void FreeGrid (Grid *grid );
int ProjectSubgrid (Subgrid *subgrid , int sx , int sy , int sz , int ix , int iy , int iz );
Subgrid *ConvertToSubgrid (Subregion *subregion );
Subgrid *ExtractSubgrid (int rx , int ry , int rz , Subgrid *subgrid );
Subgrid *IntersectSubgrids (Subgrid *subgrid1 , Subgrid *subgrid2 );
SubgridArray *SubtractSubgrids (Subgrid *subgrid1 , Subgrid *subgrid2 );
SubgridArray *UnionSubgridArray (SubgridArray *subgrids );

/* hbt.c */
HBT *HBT_new (int (*compare_method )(), void (*free_method )(), void (*printf_method )(), int (*scanf_method )(), int malloc_flag );
HBT_element *_new_HBT_element (HBT *tree , void *object , int sizeof_obj );
void _free_HBT_element (HBT *tree , HBT_element *el );
void _HBT_free (HBT *tree , HBT_element *subtree );
void HBT_free (HBT *tree );
void *HBT_lookup (HBT *tree , void *obj );
void *HBT_replace (HBT *tree , void *obj , int sizeof_obj );
int HBT_insert (HBT *tree , void *obj , int sizeof_obj );
void *HBT_delete (HBT *tree , void *obj );
void *HBT_successor (HBT *tree , void *obj );
void _HBT_printf (FILE *file , void (*printf_method )(), HBT_element *tree );
void HBT_printf (FILE *file , HBT *tree );
void HBT_scanf (FILE *file , HBT *tree );

/* infinity_norm.c */
double InfinityNorm (Vector *x );

/* innerprod.c */
double InnerProd (Vector *x , Vector *y );

/* inputRF.c */
void InputRF (GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata );
PFModule *InputRFInitInstanceXtra (Grid *grid , double *temp_data );
void InputRFFreeInstanceXtra (void );
PFModule *InputRFNewPublicXtra (char *geom_name );
void InputRFFreePublicXtra (void );
int InputRFSizeOfTempData (void );

/* input_database.c */
void IDB_Print (FILE *file , IDB_Entry *entry );
int IDB_Compare (IDB_Entry *a , IDB_Entry *b );
void IDB_Free (IDB_Entry *a );
IDB_Entry *IDB_NewEntry (char *key , char *value );
IDB *IDB_NewDB (char *filename );
void IDB_FreeDB (IDB *database );
void IDB_PrintUsage (FILE *file , IDB *database );
char *IDB_GetString (IDB *database , char *key );
char *IDB_GetStringDefault (IDB *database , char *key , char *default_value );
double IDB_GetDoubleDefault (IDB *database , char *key , double default_value );
double IDB_GetDouble (IDB *database , char *key );
int IDB_GetIntDefault (IDB *database , char *key , int default_value );
int IDB_GetInt (IDB *database , char *key );
NameArray NA_NewNameArray (char *string );
int NA_AppendToArray (NameArray name_array , char *string );
void NA_FreeNameArray (NameArray name_array );
int NA_NameToIndex (NameArray name_array , char *name );
char *NA_IndexToName (NameArray name_array , int index );
int NA_Sizeof (NameArray name_array );
void InputError (char *format , char *s1 , char *s2 );

/* kinsol_nonlin_solver.c */
int KINSolInitPC (int neq , N_Vector pressure , N_Vector uscale , N_Vector fval , N_Vector fscale , N_Vector vtemp1 , N_Vector vtemp2 , void *nl_function , double uround , long int *nfePtr , void *current_state );
int KINSolCallPC (int neq , N_Vector pressure , N_Vector uscale , N_Vector fval , N_Vector fscale , N_Vector vtem , N_Vector ftem , void *nl_function , double uround , long int *nfePtr , void *current_state );
void PrintFinalStats (FILE *out_file , long int *integer_outputs_now , long int *integer_outputs_total );
int KinsolNonlinSolver (Vector *pressure , Vector *density , Vector *old_density , Vector *saturation , Vector *old_saturation , double t , double dt , ProblemData *problem_data, Vector *old_pressure, double *outflow, Vector *evap_trans, Vector *ovrl_bc_flx );
PFModule *KinsolNonlinSolverInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , double *temp_data );
void KinsolNonlinSolverFreeInstanceXtra (void );
PFModule *KinsolNonlinSolverNewPublicXtra (void );
void KinsolNonlinSolverFreePublicXtra (void );
int KinsolNonlinSolverSizeOfTempData (void );

/* kinsol_pc.c */
void KinsolPC (Vector *rhs );
PFModule *KinsolPCInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , double *temp_data , Vector *pressure , Vector *saturation , Vector *density , double dt , double time );
void KinsolPCFreeInstanceXtra (void );
PFModule *KinsolPCNewPublicXtra (char *name , char *pc_name );
void KinsolPCFreePublicXtra (void );
int KinsolPCSizeOfTempData (void );

/* l2_error_norm.c */
void L2ErrorNorm (double time , Vector *pressure , ProblemData *problem_data , double *l2_error_norm );
PFModule *L2ErrorNormInitInstanceXtra (void );
void L2ErrorNormFreeInstanceXtra (void );
PFModule *L2ErrorNormNewPublicXtra (void );
void L2ErrorNormFreePublicXtra (void );
int L2ErrorNormSizeOfTempData (void );

/* line_process.c */
void LineProc (double *Z , double phi , double theta , double dzeta , int izeta , int nzeta , double Kmax , double dK );

/* logging.c */
void NewLogging (void );
void FreeLogging (void );
FILE *OpenLogFile (char *module_name );
int CloseLogFile (FILE *log_file );

/* matdiag_scale.c */
void MatDiagScale (Vector *x , Matrix *A , Vector *b , int flag );
PFModule *MatDiagScaleInitInstanceXtra (Grid *grid );
void MatDiagScaleFreeInstanceXtra (void );
PFModule *MatDiagScaleNewPublicXtra (char *name );
void MatDiagScaleFreePublicXtra (void );
int MatDiagScaleSizeOfTempData (void );

/* matrix.c */
Stencil *NewStencil (int shape [][3 ], int sz );
CommPkg *NewMatrixUpdatePkg (Matrix *matrix , Stencil *ghost );
CommHandle *InitMatrixUpdate (Matrix *matrix );
void FinalizeMatrixUpdate (CommHandle *handle );
Matrix *NewMatrix (Grid *grid , SubregionArray *range , Stencil *stencil , int symmetry , Stencil *ghost );
void FreeStencil (Stencil *stencil );
void FreeMatrix (Matrix *matrix );
void InitMatrix (Matrix *A , double value );

/* matvec.c */
void Matvec (double alpha , Matrix *A , Vector *x , double beta , Vector *y );

/* max_field_value.c */
double MaxFieldValue (Vector *field , Vector *phi , int dir );
double MaxPhaseFieldValue (Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *phi );
double MaxTotalFieldValue (Problem *problem , EvalStruct *eval_struct , Vector *saturation , Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *beta , Vector *phi );

/* mg_semi.c */
void MGSemi (Vector *x , Vector *b , double tol , int zero );
void SetupCoarseOps (Matrix **A_l , Matrix **P_l , int num_levels , SubregionArray **f_sra_l , SubregionArray **c_sra_l );
PFModule *MGSemiInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void MGSemiFreeInstanceXtra (void );
PFModule *MGSemiNewPublicXtra (char *name );
void MGSemiFreePublicXtra (void );
int MGSemiSizeOfTempData (void );

/* mg_semi_prolong.c */
void MGSemiProlong (Matrix *A_f , Vector *e_f , Vector *e_c , Matrix *P , SubregionArray *f_sr_array , SubregionArray *c_sr_array , ComputePkg *compute_pkg , CommPkg *e_f_comm_pkg );
ComputePkg *NewMGSemiProlongComputePkg (Grid *grid , Stencil *stencil , int sx , int sy , int sz , int c_index , int f_index );

/* mg_semi_restrict.c */
void MGSemiRestrict (Matrix *A_f , Vector *r_f , Vector *r_c , Matrix *P , SubregionArray *f_sr_array , SubregionArray *c_sr_array , ComputePkg *compute_pkg , CommPkg *r_f_comm_pkg );
ComputePkg *NewMGSemiRestrictComputePkg (Grid *grid , Stencil *stencil , int sx , int sy , int sz , int c_index , int f_index );

/* n_vector.c */
void SetPf2KinsolData (Grid *grid , int num_ghost );
N_Vector N_VNew (int N , void *machEnv );
void N_VPrint (N_Vector x );
void FreeTempVector(Vector *vector);

/* new_endpts.c */
void NewEndpts (double *alpha , double *beta , double *pp , int *size_ptr , int n , double *a_ptr , double *b_ptr , double *cond_ptr , double ereps );

/* nl_function_eval.c */
void KINSolFunctionEval (int size , N_Vector pressure , N_Vector fval , void *current_state );
void NlFunctionEval (Vector *pressure , Vector *fval , ProblemData *problem_data , Vector *saturation , Vector *old_saturation , Vector *density , Vector *old_density , double dt , double time, Vector *old_pressure, double *outflow , Vector *evap_trans, Vector *ovrl_bc_flx);
PFModule *NlFunctionEvalInitInstanceXtra (Problem *problem , Grid *grid , double *temp_data );
void NlFunctionEvalFreeInstanceXtra (void );
PFModule *NlFunctionEvalNewPublicXtra (void );
void NlFunctionEvalFreePublicXtra (void );
int NlFunctionEvalSizeOfTempData (void );

/* nodiag_scale.c */
void NoDiagScale (Vector *x , Matrix *A , Vector *b , int flag );
PFModule *NoDiagScaleInitInstanceXtra (Grid *grid );
void NoDiagScaleFreeInstanceXtra (void );
PFModule *NoDiagScaleNewPublicXtra (char *name );
void NoDiagScaleFreePublicXtra (void );
int NoDiagScaleSizeOfTempData (void );

/* parflow.c */
int main (int argc , char *argv []);

/* pcg.c */
void PCG (Vector *x , Vector *b , double tol , int zero );
PFModule *PCGInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void PCGFreeInstanceXtra (void );
PFModule *PCGNewPublicXtra (char *name );
void PCGFreePublicXtra (void );
int PCGSizeOfTempData (void );

/* permeability_face.c */
void PermeabilityFace (Vector *zperm , Vector *permeability );
PFModule *PermeabilityFaceInitInstanceXtra (Grid *z_grid );
void PermeabilityFaceFreeInstanceXtra (void );
PFModule *PermeabilityFaceNewPublicXtra (void );
void PermeabilityFaceFreePublicXtra (void );
int PermeabilityFaceSizeOfTempData (void );

/* perturb_lb.c */
void PerturbSystem (Lattice *lattice , Problem *problem );

/* pf_module.c */
PFModule *NewPFModule (void *call , void *init_instance_xtra , void *free_instance_xtra , void *new_public_xtra , void *free_public_xtra , void *sizeof_temp_data , void *instance_xtra , void *public_xtra );
PFModule *DupPFModule (PFModule *pf_module );
void FreePFModule (PFModule *pf_module );

/* pf_pfmg.c */
void PFMG (Vector *soln , Vector *rhs , double tol , int zero );
PFModule *PFMGInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *pf_matrix , double *temp_data );
void PFMGFreeInstanceXtra (void );
PFModule *PFMGNewPublicXtra (char *name );
void PFMGFreePublicXtra (void );
int PFMGSizeOfTempData (void );

/* pf_pfmg_octree.c */
void PFMGOctree (Vector *soln , Vector *rhs , double tol , int zero );
PFModule *PFMGOctreeInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *pf_matrix , double *temp_data );
void PFMGOctreeFreeInstanceXtra (void );
PFModule *PFMGOctreeNewPublicXtra (char *name );
void PFMGOctreeFreePublicXtra (void );
int PFMGOctreeSizeOfTempData (void );

/* pf_smg.c */
void SMG (Vector *soln , Vector *rhs , double tol , int zero );
PFModule *SMGInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *pf_matrix , double *temp_data );
void SMGFreeInstanceXtra (void );
PFModule *SMGNewPublicXtra (char *name );
void SMGFreePublicXtra (void );
int SMGSizeOfTempData (void );

/* pfield.c */
void PField (Grid *grid , GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata , Statistics *stats );

/* pgsRF.c */
void PGSRF (GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata );
PFModule *PGSRFInitInstanceXtra (Grid *grid , double *temp_data );
void PGSRFFreeInstanceXtra (void );
PFModule *PGSRFNewPublicXtra (char *geom_name );
void PGSRFFreePublicXtra (void );
int PGSRFSizeOfTempData (void );

/* phase_velocity_face.c */
void PhaseVelocityFace (Vector *xvel , Vector *yvel , Vector *zvel , ProblemData *problem_data , Vector *pressure , Vector **saturations , int phase );
PFModule *PhaseVelocityFaceInitInstanceXtra (Problem *problem , Grid *grid , Grid *x_grid , Grid *y_grid , Grid *z_grid , double *temp_data );
void PhaseVelocityFaceFreeInstanceXtra (void );
PFModule *PhaseVelocityFaceNewPublicXtra (void );
void PhaseVelocityFaceFreePublicXtra (void );
int PhaseVelocityFaceSizeOfTempData (void );

/* ppcg.c */
void PPCG (Vector *x , Vector *b , double tol , int zero );
PFModule *PPCGInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void PPCGFreeInstanceXtra (void );
PFModule *PPCGNewPublicXtra (char *name );
void PPCGFreePublicXtra (void );
int PPCGSizeOfTempData (void );

/* printgrid.c */
void PrintGrid (char *filename , Grid *grid );

/* printmatrix.c */
void PrintSubmatrixAll (amps_File file , Submatrix *submatrix , Stencil *stencil );
void PrintMatrixAll (char *filename , Matrix *A );
void PrintSubmatrix (amps_File file , Submatrix *submatrix , Subregion *subregion , Stencil *stencil );
void PrintMatrix (char *filename , Matrix *A );
void PrintSortMatrix (char *filename , Matrix *A , int all );

/* printvector.c */
void PrintSubvectorAll (amps_File file , Subvector *subvector );
void PrintVectorAll (char *filename , Vector *v );
void PrintSubvector (amps_File file , Subvector *subvector , Subgrid *subgrid );
void PrintVector (char *filename , Vector *v );

/* problem.c */
Problem *NewProblem (int solver );
void FreeProblem (Problem *problem , int solver );
ProblemData *NewProblemData (Grid *grid, Grid *grid2d );
void FreeProblemData (ProblemData *problem_data );

/* problem_bc.c */
BCStruct *NewBCStruct (SubgridArray *subgrids , GrGeomSolid *gr_domain , int num_patches , int *patch_indexes , int *bc_types , double ***values );
void FreeBCStruct (BCStruct *bc_struct );

/* problem_bc_internal.c */
void BCInternal (Problem *problem , ProblemData *problem_data , Matrix *A , Vector *f , double time );
PFModule *BCInternalInitInstanceXtra (void );
void BCInternalFreeInstanceXtra (void );
PFModule *BCInternalNewPublicXtra (void );
void BCInternalFreePublicXtra (void );
int BCInternalSizeOfTempData (void );

/* problem_bc_phase_saturation.c */
void BCPhaseSaturation (Vector *saturation , int phase , GrGeomSolid *gr_domain );
PFModule *BCPhaseSaturationInitInstanceXtra (void );
void BCPhaseSaturationFreeInstanceXtra (void );
PFModule *BCPhaseSaturationNewPublicXtra (int num_phases );
void BCPhaseSaturationFreePublicXtra (void );
int BCPhaseSaturationSizeOfTempData (void );

/* problem_bc_pressure.c */
BCStruct *BCPressure (ProblemData *problem_data , Grid *grid , GrGeomSolid *gr_domain , double time );
PFModule *BCPressureInitInstanceXtra (Problem *problem);
void BCPressureFreeInstanceXtra (void );
PFModule *BCPressureNewPublicXtra (int num_phases );
void BCPressureFreePublicXtra (void );
int BCPressureSizeOfTempData (void );

/* problem_capillary_pressure.c */
void CapillaryPressure (Vector *capillary_pressure , int phase_i , int phase_j , ProblemData *problem_data , Vector *phase_saturation );
PFModule *CapillaryPressureInitInstanceXtra (void );
void CapillaryPressureFreeInstanceXtra (void );
PFModule *CapillaryPressureNewPublicXtra (int num_phases );
void CapillaryPressureFreePublicXtra (void );
int CapillaryPressureSizeOfTempData (void );

/* problem_domain.c */
void Domain (ProblemData *problem_data );
PFModule *DomainInitInstanceXtra (Grid *grid );
void DomainFreeInstanceXtra (void );
PFModule *DomainNewPublicXtra (void );
void DomainFreePublicXtra (void );
int DomainSizeOfTempData (void );

/* problem_eval.c */
EvalStruct *NewEvalStruct (Problem *problem );
void FreeEvalStruct (EvalStruct *eval_struct );

/* problem_geometries.c */
void Geometries (ProblemData *problem_data );
PFModule *GeometriesInitInstanceXtra (Grid *grid);
void GeometriesFreeInstanceXtra (void );
PFModule *GeometriesNewPublicXtra (void );
void GeometriesFreePublicXtra (void );
int GeometriesSizeOfTempData (void );

/* problem_ic_phase_concen.c */
void ICPhaseConcen (Vector *ic_phase_concen , int phase , int contaminant , ProblemData *problem_data );
PFModule *ICPhaseConcenInitInstanceXtra (void );
void ICPhaseConcenFreeInstanceXtra (void );
PFModule *ICPhaseConcenNewPublicXtra (int num_phases , int num_contaminants );
void ICPhaseConcenFreePublicXtra (void );
int ICPhaseConcenSizeOfTempData (void );

/* problem_ic_phase_pressure.c */
void ICPhasePressure (Vector *ic_pressure , Vector *mask, ProblemData *problem_data , Problem *problem );
PFModule *ICPhasePressureInitInstanceXtra (Problem *problem , Grid *grid , double *temp_data );
void ICPhasePressureFreeInstanceXtra (void );
PFModule *ICPhasePressureNewPublicXtra (void );
void ICPhasePressureFreePublicXtra (void );
int ICPhasePressureSizeOfTempData (void );

/* problem_mannings.c */
void Mannings (ProblemData *problem_data, Vector *mann, Vector *dummy);
PFModule *ManningsInitInstanceXtra (Grid *grid);
void ManningsFreeInstanceXtra (void );
PFModule *ManningsNewPublicXtra (void);
void ManningsFreePublicXtra (void );
int ManningsSizeOfTempData (void );

/* problem_spec_storage.c */
void SpecStorage (ProblemData *problem_data, Vector *specific_storage );
PFModule *SpecStorageInitInstanceXtra (void );
void SpecStorageFreeInstanceXtra (void );
PFModule *SpecStorageNewPublicXtra (void );
void SpecStorageFreePublicXtra (void );
int SpecStorageSizeOfTempData (void );

/* problem_ic_phase_satur.c */
void ICPhaseSatur (Vector *ic_phase_satur , int phase , ProblemData *problem_data );
PFModule *ICPhaseSaturInitInstanceXtra (void );
void ICPhaseSaturFreeInstanceXtra (void );
PFModule *ICPhaseSaturNewPublicXtra (int num_phases );
void ICPhaseSaturFreePublicXtra (void );
int ICPhaseSaturSizeOfTempData (void );

/* problem_phase_density.c */
void PhaseDensity (int phase , Vector *phase_pressure , Vector *density_v , double *pressure_d , double *density_d , int fcn );
PFModule *PhaseDensityInitInstanceXtra (void );
void PhaseDensityFreeInstanceXtra (void );
PFModule *PhaseDensityNewPublicXtra (int num_phases );
void PhaseDensityFreePublicXtra (void );
int PhaseDensitySizeOfTempData (void );

/* problem_phase_mobility.c */
void PhaseMobility (Vector *phase_mobility_x , Vector *phase_mobility_y , Vector *phase_mobility_z , Vector *perm_x , Vector *perm_y , Vector *perm_z , int phase , Vector *phase_saturation , double phase_viscosity );
PFModule *PhaseMobilityInitInstanceXtra (void );
void PhaseMobilityFreeInstanceXtra (void );
PFModule *PhaseMobilityNewPublicXtra (int num_phases );
void PhaseMobilityFreePublicXtra (void );
int PhaseMobilitySizeOfTempData (void );

/* problem_phase_rel_perm.c */
void PhaseRelPerm (Vector *phase_rel_perm , Vector *phase_pressure , Vector *phase_density , double gravity , ProblemData *problem_data , int fcn );
PFModule *PhaseRelPermInitInstanceXtra (Grid *grid , double *temp_data );
void PhaseRelPermFreeInstanceXtra (void );
PFModule *PhaseRelPermNewPublicXtra (void );
void PhaseRelPermFreePublicXtra (void );
int PhaseRelPermSizeOfTempData (void );

/* problem_phase_source.c */
void PhaseSource (Vector *phase_source , Problem *problem , ProblemData *problem_data , double time );
PFModule *PhaseSourceInitInstanceXtra (Grid *grid);
void PhaseSourceFreeInstanceXtra (void );
PFModule *PhaseSourceNewPublicXtra (void);
void PhaseSourceFreePublicXtra (void );
int PhaseSourceSizeOfTempData (void );

/* problem_porosity.c */
void Porosity (ProblemData *problem_data , Vector *porosity , int num_geounits , GeomSolid **geounits , GrGeomSolid **gr_geounits );
PFModule *PorosityInitInstanceXtra (Grid *grid , double *temp_data );
void PorosityFreeInstanceXtra (void );
PFModule *PorosityNewPublicXtra (void );
void PorosityFreePublicXtra (void );
int PorositySizeOfTempData (void );

/* problem_retardation.c */
void Retardation (Vector *solidmassfactor , int contaminant , ProblemData *problem_data );
PFModule *RetardationInitInstanceXtra (double *temp_data );
void RetardationFreeInstanceXtra (void );
PFModule *RetardationNewPublicXtra (int num_contaminants );
void RetardationFreePublicXtra (void );
int RetardationSizeOfTempData (void );

/* problem_richards_bc_internal.c */
void RichardsBCInternal (Problem *problem , ProblemData *problem_data , Vector *f , Matrix *A , double time , Vector *pressure , int fcn );
PFModule *RichardsBCInternalInitInstanceXtra (void );
void RichardsBCInternalFreeInstanceXtra (void );
PFModule *RichardsBCInternalNewPublicXtra (void );
void RichardsBCInternalFreePublicXtra (void );
int RichardsBCInternalSizeOfTempData (void );

/* problem_saturation.c */
void Saturation (Vector *phase_saturation , Vector *phase_pressure , Vector *phase_density , double gravity , ProblemData *problem_data , int fcn );
PFModule *SaturationInitInstanceXtra (Grid *grid , double *temp_data );
void SaturationFreeInstanceXtra (void );
PFModule *SaturationNewPublicXtra (void );
void SaturationFreePublicXtra (void );
int SaturationSizeOfTempData (void );

/* problem_saturation_constitutive.c */
void SaturationConstitutive (Vector **phase_saturations );
PFModule *SaturationConstitutiveInitInstanceXtra (Grid *grid );
void SaturationConstitutiveFreeInstanceXtra (void );
PFModule *SaturationConstitutiveNewPublicXtra (int num_phases );
void SaturationConstitutiveFreePublicXtra (void );
int SaturationConstitutiveSizeOfTempData (void );

/* problem_toposlope_x.c */
void XSlope (ProblemData *problem_data, Vector *x_sl, Vector *dummy );
PFModule *XSlopeInitInstanceXtra (Grid *grid);
void XSlopeFreeInstanceXtra (void );
PFModule *XSlopeNewPublicXtra (void);
void XSlopeFreePublicXtra (void );
int XSlopeSizeOfTempData (void );

/* problem_toposlope_y.c */
void YSlope (ProblemData *problem_data, Vector *y_slope, Vector *dummy );
PFModule *YSlopeInitInstanceXtra (Grid *grid);
void YSlopeFreeInstanceXtra (void );
PFModule *YSlopeNewPublicXtra (void);
void YSlopeFreePublicXtra (void );
int YSlopeSizeOfTempData (void );

/* random.c */
void SeedRand (int seed );
double Rand (void );

/* ratqr.c */
int ratqr_ (int *n , double *eps1 , double *d , double *e , double *e2 , int *m , double *w , int *ind , double *bd , int *type , int *idef , int *ierr );
double epslon_ (double *x );

/* rb_GS_point.c */
void RedBlackGSPoint (Vector *x , Vector *b , double tol , int zero );
PFModule *RedBlackGSPointInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void RedBlackGSPointFreeInstanceXtra (void );
PFModule *RedBlackGSPointNewPublicXtra (char *name );
void RedBlackGSPointFreePublicXtra (void );
int RedBlackGSPointSizeOfTempData (void );

/* read_parflow_binary.c */
void ReadPFBinary_Subvector (amps_File file , Subvector *subvector , Subgrid *subgrid );
void ReadPFBinary (char *filename , Vector *v );

/* reg_from_stenc.c */
void ComputeRegFromStencil (Region **dep_reg_ptr , Region **ind_reg_ptr , SubregionArray *cr_array , Region *send_reg , Region *recv_reg , Stencil *stencil );
SubgridArray *GetGridNeighbors (SubgridArray *subgrids , SubgridArray *all_subgrids , Stencil *stencil );
void CommRegFromStencil (Region **send_region_ptr , Region **recv_region_ptr , Grid *grid , Stencil *stencil );

/* region.c */
Subregion *NewSubregion (int ix , int iy , int iz , int nx , int ny , int nz , int sx , int sy , int sz , int rx , int ry , int rz , int process );
SubregionArray *NewSubregionArray (void );
Region *NewRegion (int size );
void FreeSubregion (Subregion *subregion );
void FreeSubregionArray (SubregionArray *subregion_array );
void FreeRegion (Region *region );
Subregion *DuplicateSubregion (Subregion *subregion );
SubregionArray *DuplicateSubregionArray (SubregionArray *subregion_array );
Region *DuplicateRegion (Region *region );
void AppendSubregion (Subregion *subregion , SubregionArray *sr_array );
void DeleteSubregion (SubregionArray *sr_array , int index );
void AppendSubregionArray (SubregionArray *sr_array_0 , SubregionArray *sr_array_1 );

/* richards_jacobian_eval.c */
int KINSolMatVec (void *current_state , N_Vector x , N_Vector y , int *recompute , N_Vector pressure );
void RichardsJacobianEval (Vector *pressure , Matrix **ptr_to_J , Vector *saturation , Vector *density , ProblemData *problem_data , double dt , double time , int symm_part );
PFModule *RichardsJacobianEvalInitInstanceXtra (Problem *problem , Grid *grid , double *temp_data , int symmetric_jac );
void RichardsJacobianEvalFreeInstanceXtra (void );
PFModule *RichardsJacobianEvalNewPublicXtra (void );
void RichardsJacobianEvalFreePublicXtra (void );
int RichardsJacobianEvalSizeOfTempData (void );

/* sadvection_godunov.c */
void SatGodunov (ProblemData *problem_data , int phase , Vector *old_saturation , Vector *new_saturation , Vector *x_velocity , Vector *y_velocity , Vector *z_velocity , Vector *z_permeability , Vector *solid_mass_factor , double *viscosity , double *density , double gravity , double time , double deltat , int order );
PFModule *SatGodunovInitInstanceXtra (Problem *problem , Grid *grid , double *temp_data );
void SatGodunovFreeInstanceXtra (void );
PFModule *SatGodunovNewPublicXtra (void );
void SatGodunovFreePublicXtra (void );
int SatGodunovSizeOfTempData (void );

/* scale.c */
void Scale (double alpha , Vector *y );

/* select_time_step.c */
void SelectTimeStep (double *dt , char *dt_info , double time , Problem *problem , ProblemData *problem_data );
PFModule *SelectTimeStepInitInstanceXtra (void );
void SelectTimeStepFreeInstanceXtra (void );
PFModule *SelectTimeStepNewPublicXtra (void );
void SelectTimeStepFreePublicXtra (void );
int SelectTimeStepSizeOfTempData (void );

/* set_problem_data.c */
void SetProblemData (ProblemData *problem_data );
PFModule *SetProblemDataInitInstanceXtra (Problem *problem , Grid *grid , Grid *grid2d, double *temp_data );
void SetProblemDataFreeInstanceXtra (void );
PFModule *SetProblemDataNewPublicXtra (void );
void SetProblemDataFreePublicXtra (void );
int SetProblemDataSizeOfTempData (void );

/* sim_shear.c */
double **SimShear (double **shear_min_ptr , double **shear_max_ptr , GeomSolid *geom_solid , SubgridArray *subgrids , int type );

/* solver.c */
void Solve (void );
void NewSolver (void );
void FreeSolver (void );

/* solver_impes.c */
void SolverImpes (void );
PFModule *SolverImpesInitInstanceXtra (void );
void SolverImpesFreeInstanceXtra (void );
PFModule *SolverImpesNewPublicXtra (char *name );
void SolverImpesFreePublicXtra (void );
int SolverImpesSizeOfTempData (void );

/* solver_lb.c */
void SolverDiffusion (void );
PFModule *SolverDiffusionInitInstanceXtra (void );
void SolverDiffusionFreeInstanceXtra (void );
PFModule *SolverDiffusionNewPublicXtra (char *name );
void SolverDiffusionFreePublicXtra (void );
int SolverDiffusionSizeOfTempData (void );

/* solver_richards.c */
void SolverRichards (void );
PFModule *SolverRichardsInitInstanceXtra (void );
void SolverRichardsFreeInstanceXtra (void );
PFModule *SolverRichardsNewPublicXtra (char *name );
void SolverRichardsFreePublicXtra (void );
int SolverRichardsSizeOfTempData (void );
ProblemData *GetProblemDataRichards (PFModule *this_module);
Problem  *GetProblemRichards (PFModule *this_module);
PFModule *GetICPhasePressureRichards (PFModule *this_module);
void AdvanceRichards (PFModule *this_module, 
		       double start_time,      
		       double stop_time,       
		       double dt,              
		       int compute_time_step,  
		       Vector *evap_trans,     
		       Vector **pressure_out,  
		       Vector **porosity_out,
			Vector **saturation_out);
void SetupRichards (PFModule *this_module);


/* subsrf_sim.c */
void SubsrfSim (ProblemData *problem_data , Vector *perm_x , Vector *perm_y , Vector *perm_z , int num_geounits , GeomSolid **geounits , GrGeomSolid **gr_geounits );
PFModule *SubsrfSimInitInstanceXtra (Grid *grid , double *temp_data );
void SubsrfSimFreeInstanceXtra (void );
PFModule *SubsrfSimNewPublicXtra (void );
void SubsrfSimFreePublicXtra (void );
int SubsrfSimSizeOfTempData (void );

/* time_cycle_data.c */
TimeCycleData *NewTimeCycleData (int number_of_cycles , int *number_of_intervals );
void FreeTimeCycleData (TimeCycleData *time_cycle_data );
void PrintTimeCycleData (TimeCycleData *time_cycle_data );
int TimeCycleDataComputeIntervalNumber (Problem *problem , double time , TimeCycleData *time_cycle_data , int cycle_number );
double TimeCycleDataComputeNextTransition (Problem *problem , double time , TimeCycleData *time_cycle_data );
void ReadGlobalTimeCycleData (void );
void FreeGlobalTimeCycleData (void );

/* timing.c */
#if defined(PF_TIMING)
void NewTiming (void );
int RegisterTiming (char *name );
void PrintTiming (void );
void FreeTiming (void );
#endif

/* total_velocity_face.c */
void TotalVelocityFace (Vector *xvel , Vector *yvel , Vector *zvel , ProblemData *problem_data , Vector *total_mobility_x , Vector *total_mobility_y , Vector *total_mobility_z , Vector *pressure , Vector **saturations );
PFModule *TotalVelocityFaceInitInstanceXtra (Problem *problem , Grid *grid , Grid *x_grid , Grid *y_grid , Grid *z_grid , double *temp_data );
void TotalVelocityFaceFreeInstanceXtra (void );
PFModule *TotalVelocityFaceNewPublicXtra (void );
void TotalVelocityFaceFreePublicXtra (void );
int TotalVelocityFaceSizeOfTempData (void );

/* turning_bands.c */
void Turn (Vector *field , void *vxtra );
int InitTurn (void );
void *NewTurn (char *geom_name );
void FreeTurn (void *xtra );

/* turning_bandsRF.c */
void TurningBandsRF (GeomSolid *geounit , GrGeomSolid *gr_geounit , Vector *field , RFCondData *cdata );
PFModule *TurningBandsRFInitInstanceXtra (Grid *grid , double *temp_data );
void TurningBandsRFFreeInstanceXtra (void );
PFModule *TurningBandsRFNewPublicXtra (char *geom_name );
void TurningBandsRFFreePublicXtra (void );
int TurningBandsRFSizeOfTempData (void );

/* usergrid_input.c */
Subgrid *ReadUserSubgrid (void );
Grid *ReadUserGrid (void );
void FreeUserGrid (Grid *user_grid );

/* vector.c */
CommPkg *NewVectorCommPkg (Vector *vector , ComputePkg *compute_pkg );
CommHandle *InitVectorUpdate (Vector *vector , int update_mode );
void FinalizeVectorUpdate (CommHandle *handle );
Vector *NewVector (Grid *grid , int nc , int num_ghost );
void FreeVector (Vector *vector );
void InitVector (Vector *v , double value );
void InitVectorAll (Vector *v , double value );
void InitVectorInc (Vector *v , double value , double inc );
void InitVectorRandom (Vector *v , long seed );

/* vector_utilities.c */
void PFVLinearSum (double a , Vector *x , double b , Vector *y , Vector *z );
void PFVConstInit (double c , Vector *z );
void PFVProd (Vector *x , Vector *y , Vector *z );
void PFVDiv (Vector *x , Vector *y , Vector *z );
void PFVScale (double c , Vector *x , Vector *z );
void PFVAbs (Vector *x , Vector *z );
void PFVInv (Vector *x , Vector *z );
void PFVAddConst (Vector *x , double b , Vector *z );
double PFVDotProd (Vector *x , Vector *y );
double PFVMaxNorm (Vector *x );
double PFVWrmsNorm (Vector *x , Vector *w );
double PFVWL2Norm (Vector *x , Vector *w );
double PFVL1Norm (Vector *x );
double PFVMin (Vector *x );
double PFVMax (Vector *x );
int PFVConstrProdPos (Vector *c , Vector *x );
void PFVCompare (double c , Vector *x , Vector *z );
int PFVInvTest (Vector *x , Vector *z );
void PFVCopy (Vector *x , Vector *y );
void PFVSum (Vector *x , Vector *y , Vector *z );
void PFVDiff (Vector *x , Vector *y , Vector *z );
void PFVNeg (Vector *x , Vector *z );
void PFVScaleSum (double c , Vector *x , Vector *y , Vector *z );
void PFVScaleDiff (double c , Vector *x , Vector *y , Vector *z );
void PFVLin1 (double a , Vector *x , Vector *y , Vector *z );
void PFVLin2 (double a , Vector *x , Vector *y , Vector *z );
void PFVAxpy (double a , Vector *x , Vector *y );
void PFVScaleBy (double a , Vector *x );

/* w_jacobi.c */
void WJacobi (Vector *x , Vector *b , double tol , int zero );
PFModule *WJacobiInitInstanceXtra (Problem *problem , Grid *grid , ProblemData *problem_data , Matrix *A , double *temp_data );
void WJacobiFreeInstanceXtra (void );
PFModule *WJacobiNewPublicXtra (char *name );
void WJacobiFreePublicXtra (void );
int WJacobiSizeOfTempData (void );

/* well.c */
WellData *NewWellData (void );
void FreeWellData (WellData *well_data );
void PrintWellData (WellData *well_data , unsigned int print_mask );
void WriteWells (char *file_prefix , Problem *problem , WellData *well_data , double time , int write_header );

/* well_package.c */
void WellPackage (ProblemData *problem_data );
PFModule *WellPackageInitInstanceXtra (void );
void WellPackageFreeInstanceXtra (void );
PFModule *WellPackageNewPublicXtra (int num_phases , int num_contaminants );
void WellPackageFreePublicXtra (void );
int WellPackageSizeOfTempData (void );

/* wells_lb.c */
void LBWells (Lattice *lattice , Problem *problem , ProblemData *problem_data );

/* write_parflow_binary.c */
long SizeofPFBinarySubvector (Subvector *subvector , Subgrid *subgrid );
void WritePFBinary_Subvector (amps_File file , Subvector *subvector , Subgrid *subgrid );
void WritePFBinary (char *file_prefix , char *file_suffix , Vector *v );
long SizeofPFSBinarySubvector (Subvector *subvector , Subgrid *subgrid , double drop_tolerance );
void WritePFSBinary_Subvector (amps_File file , Subvector *subvector , Subgrid *subgrid , double drop_tolerance );
void WritePFSBinary (char *file_prefix , char *file_suffix , Vector *v , double drop_tolerance );

/* write_parflow_silo.c */
void     WriteSilo(char    *file_prefix, char    *file_suffix, Vector  *v, 
                   double time, int step, char *variable_name);
void     WriteSiloInit(char    *file_prefix);

/* wrf_parflow.c */
void wrfparflowinit_ ();
void wrfparflowadvance_(double *current_time, 
			double *dt,
                        float *wrf_flux,
                        float *wrf_pressure,
                        float *wrf_porosity,
                        float *wrf_saturation,
			int    *num_soil_layers,
			int    *ghost_size_i_lower,  /* Number of ghost cells */
			int    *ghost_size_j_lower,
			int    *ghost_size_i_upper,
			int    *ghost_size_j_upper);

void WRF2PF(float  *wrf_array, 
	    int     wrf_depth, 
	    int     ghost_size_i_lower,  /* Number of ghost cells */
	    int     ghost_size_j_lower,
	    int     ghost_size_i_upper,
	    int     ghost_size_j_upper,
	    Vector *pf_vector,
	    Vector *top);

void PF2WRF ( Vector *pf_vector,
	      float  *wrf_array,
	      int     wrf_depth,
	      int     ghost_size_i_lower,  /* Number of ghost cells */
	      int     ghost_size_j_lower,
	      int     ghost_size_i_upper,
	      int     ghost_size_j_upper,
	      Vector *top);

void ComputeTop (  Problem     *problem,     
		   ProblemData *problem_data
		   );

int CheckTime(Problem *problem, char *key, double time);

/* evaptranssum.c */
void EvapTransSum(ProblemData *problem_data, double dt, Vector *evap_trans_sum, Vector *evap_trans);

void OverlandSum(ProblemData *problem_data, 
		 Vector      *pressure,       /* Current pressure values */
		 double dt, 
		 Vector *overland_sum);
