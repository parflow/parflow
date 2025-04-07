!#include <misc.h>

subroutine clm_dynvegpar (clm,clm_forc_veg)

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
!  Vegetation dynamic parameters and snow cover fraction as subgrid vectors 
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!  15 November 2000: Mariana Vertenstein
!=========================================================================
! $Id: clm_dynvegpar.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! CLM tile variables
  use clm_varcon, only :istice
  implicit none

!=== Arguments ===========================================================

  type (clm1d)   :: clm 

!=== Local Variables =====================================================

  real(r8) seasb   !temperature dependence of vegetation cover [-]
  real(r8) fb      !fraction of canopy layer covered by snow
  integer,intent(in)  :: clm_forc_veg

!=== End Variable List ===================================================

! Note: temporarily set, they can be given by measurement, or dynamic ecosystem model
! only nonzero if NOT glacier/ice, water or bare soil

  if ((.not. clm%lakpoi) .AND. (.not. clm%baresoil) .AND. (clm%itypwat/=istice)) then  
     !seasb     = max(dble(0.), dble(1.) - dble(0.0016)*max(298.-clm%t_soisno(7),dble(0.0))**2)
     seasb     = max(dble(0.), dble(1.) - dble(0.0016)*max(298.-clm%t_soisno(clm%soi_z),dble(0.0))**2) ! NBE: Added variable to set layer #
     clm%tlai  = clm%maxlai + (clm%minlai-clm%maxlai)*(1.-seasb)
     clm%tsai  = clm%tsai
  else
     clm%tlai  = 0.
     clm%tsai  = 0.
  endif

! Adjust lai and sai for burying by snow. if exposed lai and sai are less than 0.05,
! set equal to zero to prevent numerical problems associated with very small lai,sai

! LB revised 5/17/16 - use fraction of veg height covered in snow to approximate unburied LAI+SAI
! Assumes veg height is 10xRoughness Length. Not lateral snow fraction 
  fb = 0.1*clm%snowdp/clm%z0m
  fb=min(dble(1.) ,fb) 
  !fb = fb/(1.+fb) !- never covers grass

  if  (clm_forc_veg == 0) then
      clm%elai = clm%tlai*(1.-fb)
      clm%esai = clm%tsai*(1.-fb)
  endif

  if (clm%elai < 0.05) clm%elai = 0._r8
  if (clm%esai < 0.05) clm%esai = 0._r8

! Fraction of vegetation free of snow

  if ((clm%elai + clm%esai) >= 0.05) then
     clm%frac_veg_nosno = 1
  else
     clm%frac_veg_nosno = 0
  endif
  
! Fraction of soil covered by snow

  clm%frac_sno = clm%snowdp/(10.*clm%zlnd + clm%snowdp)  

end subroutine clm_dynvegpar
