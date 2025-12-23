!#include <misc.h>

subroutine clm_balchk (clm, nstep) 

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
  !  Water and energy balance check  
  !
  !  This subroutine accumulates the numerical truncation errors of the water
  !  and energy balance calculation. It is helpful to see the performance of 
  !  the process of integration.
  !
  !  The error for energy balance: 
  !  error = abs(Net radiation - the change of internal energy - Sensible heat
  !              - Latent heat) 
  !  The error should be less than 0.02 W/m2 in each time integration interval;
  !
  !  The error for water balance:
  !  error = abs(precipitation - change of water storage - evaporation - runoff)
  !  The error should be less than 0.001 mm in  each time integration interval.
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
  !  10 November 2000: Mariana Vertenstein
  !=========================================================================
  ! $Id: clm_balchk.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use clmtype
  use clm_varpar, only : nlevsoi
  use clm_varcon, only : istsoil, tfrz
  implicit none

  !=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm     !CLM 1-D Module
  integer     , intent(in)    :: nstep   !time step

  !=== Local Variables =====================================================

  !=== End Variable List ===================================================

  !------------------------------------------------------------------------
  ! Water balance 
  !------------------------------------------------------------------------

  !  clm%errh2o = clm%endwb - clm%begwb - &
  !           ( clm%forc_rain  + clm%forc_snow - clm%qflx_evap_tot - clm%qflx_surf &
  !           - clm%qflx_qrgwl + clm%qflx_qirr - clm%qflx_drain ) * clm%dtime

  !@ Stefan: new balance check (must be verified again 03/11/2004)
  clm%errh2o = clm%endwb - clm%begwb - &
       (clm%qflx_infl - clm%qflx_tran_veg) * clm%dtime

  if (abs(clm%errh2o) > .10) then
     !@     write(6,200)'water balance error',nstep,clm%kpatch,clm%errh2o
     !@     call endrun
  endif

  !------------------------------------------------------------------------
  ! Solar radiation energy balance
  !------------------------------------------------------------------------

  clm%errsol = clm%fsa + clm%fsr - &
       (clm%forc_solad(1) + clm%forc_solad(2) + clm%forc_solai(1) + clm%forc_solai(2))

  !if (abs(clm%errsol) > .10 ) then
  !write(166,100)'solar radiation balance error',nstep,clm%kpatch,clm%errsol
  call endrun
  !endif

  !------------------------------------------------------------------------
  ! Longwave radiation energy balance
  !------------------------------------------------------------------------

  clm%errlon = clm%eflx_lwrad_out - clm%eflx_lwrad_net - clm%forc_lwrad

  !if (abs(clm%errlon) > .10 ) then
  !write(166,100)' longwave energy balance error',nstep,clm%kpatch,clm%errlon
  call endrun
  !endif

  !------------------------------------------------------------------------
  ! Surface energy balance
  !------------------------------------------------------------------------

  clm%errseb = clm%sabv + clm%sabg  &
       + clm%forc_lwrad - clm%eflx_lwrad_out &
       - clm%eflx_sh_tot &
       - clm%eflx_lh_tot &
       - clm%eflx_soil_grnd 

  !if (abs(clm%errseb) > .10 ) then
  !write(166,100)' surface flux energy balance error',nstep,clm%kpatch,clm%errseb
  !write(166,300)clm%sabv + clm%sabg,clm%forc_lwrad - clm%eflx_lwrad_out, &
  !     clm%eflx_sh_tot, clm%eflx_lh_tot, clm%eflx_soil_grnd, &
  !     clm%eflx_impsoil
  if (abs(clm%errseb) > 10.) call endrun
  !endif

  !------------------------------------------------------------------------
  ! Accumulation of water and surface energy balance error
  !------------------------------------------------------------------------

  clm%acc_errh2o = clm%acc_errh2o + clm%errh2o
  clm%acc_errseb = clm%acc_errseb + clm%errseb

100 format (1x,a14,' nstep =',i10,' point =',i6,' imbalance =',f8.2,' W/m2')
!200 format (1x,a14,' nstep =',i10,' point =',i6,' imbalance =',f8.2,' mm')
300 format(' ','absorbed solar= ',d13.5,&
         ' longwave net= ',d13.5,&
         ' sensible flux= ',d13.5,&
         ' latent flux= ',d13.5,&
         ' ground heat flux = ',d13.5,&
         ' implicit soil evaporation= ',d13.5,' (all in W/m2)')

end subroutine clm_balchk











