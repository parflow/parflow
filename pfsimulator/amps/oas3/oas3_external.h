#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
extern int oas_pfl_vardef_mp_comp_id_;
extern int oas_pfl_vardef_mp_ierror_;
extern int oas_pfl_vardef_mp_localcomm_;
#else
extern int __oas_pfl_vardef_MOD_comp_id;
extern int __oas_pfl_vardef_MOD_ierror;
extern int __oas_pfl_vardef_MOD_localcomm;
#endif

#define oas_pfl_init oas_pfl_init_
#define CALL_oas_pfl_init(arg1) oas_pfl_init(arg1);
void oas_pfl_init(int *arg1);
#define oas_pfl_finalize oas_pfl_finalize_
#define CALL_oas_pfl_finalize(arg2) oas_pfl_finalize(arg2);
void oas_pfl_finalize(int *arg2);
