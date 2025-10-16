!#include <misc.h>

module drv_gridmodule 

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! DESCRIPTION:
!  Module for grid space variable specification.
!
! REVISION HISTORY:
!  15 Jan 2000: Paul Houser; Initial code
!=========================================================================

  use MOD_Precision
  !use clm_varpar, only : max_nlevsoi
  !use MOD_Vars_Global, only : nl_soil
  implicit none
  public griddec

  type griddec

!=== GRID SPACE User-Defined Parameters ==================================

     integer :: patchclass
     real(r8):: patchlatr
     real(r8):: patchlonr
     !real(r8):: vf_quartz(nl_soil)  !soil, nc readin ##not used at all
     real(r8):: int_soil_grav_l(8) !soil, nc readin ##used in soil_hcap_cond, maybe setup in 2nd file
     real(r8):: int_soil_sand_l(8)      !soil, nc readin ##used in soil_hcap_cond, maybe setup in 2nd file 
     real(r8):: int_soil_clay_l(8)    !soil, nc readin
     real(r8):: int_soil_oc_l(8) !soil, nc readin
     real(r8):: int_soil_bd_l(8)    !soil, nc readin
     !real(r8):: porsl(1:,i)      !soil, nc readin, from pf 
     !real(r8):: psi0(nl_soil)       !soil, nc readin              
     !real(r8):: bsw(nl_soil)        !soil, nc readin 
     !real(r8):: theta_r(nl_soil)    !soil, nc readin                                                
     !real(r8):: alpha_vgm(nl_soil)  !soil, nc readin 
     !real(r8):: n_vgm(nl_soil)      !soil, nc readin 
     !real(r8):: L_vgm(nl_soil)      !soil, nc readin  
     !real(r8):: hksati(nl_soil)     !soil, nc readin 
     !real(r8):: csol(nl_soil)       !soil, nc readin
     !real(r8):: k_solids(nl_soil)   !soil, nc readin
     !real(r8):: dksatu(nl_soil)     !soil, nc readin
     !real(r8):: dksatf(nl_soil)     !soil, nc readin
     !real(r8):: dkdry(nl_soil)      !soil, nc readin
     !real(r8):: BA_alpha(nl_soil)   !soil, nc readin
     !real(r8):: BA_beta(nl_soil)    !soil, nc readin
     !real(r8):: htoplc              !Forest_Height.nc
     !real(r8):: OM_density(nl_soil)   !soil, nc readin 
     !real(r8):: BD_all(nl_soil)       !soil, nc readin   

  end type griddec

end module drv_gridmodule




