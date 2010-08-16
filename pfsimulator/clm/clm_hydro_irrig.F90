!#include <misc.h>

subroutine clm_hydro_irrig (clm,gmt)

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
! Irrigate crops to depth of 1 m and modify runoff accordingly
!
! REVISION HISTORY:
!  7 November 2000: Mariana Vertenstein; Initial code
!
!=========================================================================
! $Id: clm_hydro_irrig.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype
  use clm_varpar, only : nlevsoi
  use clm_varcon, only : denh2o
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm !CLM 1-D Module
  real(r8) gmt     !needed for irrigation scheduling

!=== Local Variables =====================================================

  integer i        !index
  real(r8) wtold   !temporary var
  real(r8) wtnew   !temporary var
  real(r8) temp    !temporary var 
  real(r8) testsat !saturation to compare to threshold

!=========================================================================

! To saturate the soil, make the liquid water volume equal the minimum of 
! the effective porosity and watopt. Irrigate soil to a depth of 30 cm.

!  IMF: Original irrigation scheme 
!       This was never implemented -- clm%irrig hardwired as .false. in drv_readvegpf 
!  clm%qflx_qirr = 0.
!  if (clm%irrig .and. (clm%elai+clm%esai)>0) then
!     wtold = 0.
!     wtnew = 0.
!     do i = 1, nlevsoi
!        if (clm%zi(i) <= 0.3) then
!           wtold = wtold + clm%h2osoi_liq(i)
!           temp = min(clm%watopt(i),clm%eff_porosity(i))
!           clm%h2osoi_liq(i) = max(temp*clm%dz(i)*denh2o,clm%h2osoi_liq(i))
!           wtnew = wtnew + clm%h2osoi_liq(i)
!        end if
!     end do
!     clm%qflx_qirr = (wtnew - wtold) / clm%dtime
!  end if

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! IMF: New irrigation schemes

  ! Initially set irrigation to zero 
  clm%qflx_qirr = 0.
  clm%qflx_qirr_inst = 0.

  ! Test if veg type gets irrigated
  if ( (clm%irrig==1) .and. ((clm%elai+clm%esai)>0) ) then

    ! Test irrigation cycle
    ! If Constant, irrigate if irr_start <= gmt <= irr_stop
    ! If Deficit,  irrigate if soil moisture <= irr_threshold

    ! Constant (irr_cycle == 0) .AND. (irr_start <= gmt <= irr_stop)
    if ( (clm%irr_cycle==0) .and. (gmt >= clm%irr_start) .and. (gmt <= clm%irr_stop) ) then

     ! Set irrigation flag to 1.0:
     clm%irr_flag = 1.0d0

     ! Select irrigation type: 
     select case (clm%irr_type)
     case ( 0 )  ! None 
        clm%qflx_qirr = 0.
     case ( 1 )  ! Spray
        clm%qflx_qirr = clm%irr_rate
     case ( 2 )  ! Drip
        clm%qflx_qirr = clm%irr_rate
     case ( 3 )  ! Instant (original CLM formulation)
        wtold = 0.
        wtnew = 0.
        do i = 1, nlevsoi
           if (clm%zi(i) <= 0.3) then
              wtold = wtold + clm%h2osoi_liq(i)
              temp = min(clm%watopt(i),clm%eff_porosity(i))
              clm%h2osoi_liq(i) = max(temp*clm%dz(i)*denh2o,clm%h2osoi_liq(i))
              wtnew = wtnew + clm%h2osoi_liq(i)
           end if
        end do
        clm%qflx_qirr_inst = (wtnew - wtold) / clm%dtime 
     end select
    endif ! Apply irrigation 

    ! Deficit-based irrigation schedule (irr_cycle == 1)
    if ( clm%irr_cycle==1 ) then
 
     ! If irrigation start time, test if need to irrigate (based on saturation)
     if ( gmt==clm%irr_start ) then 

      ! Calculate test saturation
      select case (clm%threshold_type)
      case ( 0 )  ! Top soil layer only -- layer 1
         testsat = clm%h2osoi_liq(1) / ( clm%watsat(1)*clm%dz(1)*denh2o )
      case ( 1 )  ! Bottom soil layer only -- layer nlevsoi
         testsat = clm%h2osoi_liq(nlevsoi) / ( clm%watsat(nlevsoi)*clm%dz(nlevsoi)*denh2o )
      case ( 2 )  ! Average over soil column
         testsat = 0.
         do i = 1, nlevsoi
            testsat = testsat + ( clm%h2osoi_liq(i) / ( clm%watsat(i)*clm%dz(i)*denh2o ) )
         enddo
         testsat = testsat / float( nlevsoi )
      end select   

      ! Determine whether to irrigate
      if ( testsat < clm%irr_threshold ) then 
       clm%irr_flag = 1.0d0  ! irrigate if testsat < threshold
      else
       clm%irr_flag = 0.0d0  ! otherwise don't irrigate
      endif

     endif ! daily irrigation flag
 
     ! If (irr_start <= gmt <= irr_stop) *AND* (sat < threshold), apply irrigation
     if ( (gmt >= clm%irr_start) .and. (gmt <= clm%irr_stop) .and. (clm%irr_flag > 0.) ) then
      select case (clm%irr_type)
      case ( 0 )  ! None 
         clm%qflx_qirr = 0.
      case ( 1 )  ! Spray
         clm%qflx_qirr = clm%irr_rate
      case ( 2 )  ! Drip
         clm%qflx_qirr = clm%irr_rate
      case ( 3 )  ! Instant (original CLM formulation)
         wtold = 0.
         wtnew = 0.
         do i = 1, nlevsoi
            if (clm%zi(i) <= 0.3) then
             wtold = wtold + clm%h2osoi_liq(i)
             temp = min(clm%watopt(i),clm%eff_porosity(i))
             clm%h2osoi_liq(i) = max(temp*clm%dz(i)*denh2o,clm%h2osoi_liq(i))
             wtnew = wtnew + clm%h2osoi_liq(i)
            end if
         end do
         clm%qflx_qirr_inst = (wtnew - wtold) / clm%dtime
      end select
     endif ! Applying irrigation

    endif ! irr_cycle == 1

  endif ! irrig == 1

end subroutine clm_hydro_irrig





