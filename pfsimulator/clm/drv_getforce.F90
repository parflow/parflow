!#include <misc.h>

subroutine drv_getforce (drv,tile,clm,nx,ny,sw_pf,lw_pf,prcp_pf,tas_pf,u_pf,v_pf, &
  patm_pf,qatm_pf,lai_pf,sai_pf,z0m_pf,displa_pf,istep_pf,clm_forc_veg)

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
!  Access meteorological data - this current version reads 1D forcing
!  and distributes it to the clm domain (spatially constant).  This routine
!  must be modified to allow for spatially variable forcing, or coupling to
!  a GCM.
!
!  The user may likely want to modify this subroutine significantly,
!  to include such things as space/time intrpolation of forcing to the
!  CLM grid, reading of spatially variable binary data, etc.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: drv_getforce.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use clm_varcon, only : tfrz, tcrit
  implicit none

!=== Arguments ===========================================================

  type (drvdec) ,intent(inout) :: drv              
  type (tiledec),intent(inout) :: tile(drv%nch)
  type (clm1d)  ,intent(inout) :: clm (drv%nch)
  integer,intent(in)  :: istep_pf,nx,ny
  integer,intent(in)  :: clm_forc_veg                       ! BH: whether vegetation (LAI, SAI, z0m, displa) is being forced 0=no, 1=yes
  real(r8),intent(in) :: sw_pf((nx+2)*(ny+2)*3)             ! SW rad, passed from PF
  real(r8),intent(in) :: lw_pf((nx+2)*(ny+2)*3)             ! LW rad, passed from PF
  real(r8),intent(in) :: prcp_pf((nx+2)*(ny+2)*3)           ! Precip, passed from PF
  real(r8),intent(in) :: tas_pf((nx+2)*(ny+2)*3)            ! Air temp, passed from PF
  real(r8),intent(in) :: u_pf((nx+2)*(ny+2)*3)              ! u-wind, passed from PF
  real(r8),intent(in) :: v_pf((nx+2)*(ny+2)*3)              ! v-wind, passed from PF
  real(r8),intent(in) :: patm_pf((nx+2)*(ny+2)*3)           ! air pressure, passed from PF
  real(r8),intent(in) :: qatm_pf((nx+2)*(ny+2)*3)           ! air specific humidity, passed from PF
  real(r8),intent(in) :: lai_pf((nx+2)*(ny+2)*3)            ! lai, passed from PF !BH
  real(r8),intent(in) :: sai_pf((nx+2)*(ny+2)*3)            ! sai, passed from PF !BH
  real(r8),intent(in) :: z0m_pf((nx+2)*(ny+2)*3)            ! z0m, passed from PF !BH
  real(r8),intent(in) :: displa_pf((nx+2)*(ny+2)*3)         ! displacement height, passed from PF !BH
  !real(r8),intent(in) :: slope_x_pf((nx+2)*(ny+2)*3)        ! slope in x direction, passed from PF !IJB
  !real(r8),intent(in) :: slope_y_pf((nx+2)*(ny+2)*3)        ! slope in y direction, passed from PF !IJB


!=== Local Variables =====================================================

  real(r8) solar     ! incident solar radiation [w/m2]
  real(r8) prcp      ! precipitation [mm/s]
  integer t,i,j,k,l  ! Looping indices
! integer nx,ny      ! Array sizes
  
!=== End Variable List ===================================================

!=== Increment Time Step Counter
! clm%istep=clm%istep+1 
  clm%istep=istep_pf

! Valdai - 1D Met data

  ! IMF: modified for 2D
  ! Loop over tile space (convert from pf-to-clm)
  do t = 1,drv%nch

     i = tile(t)%col
     j = tile(t)%row
     l = (1+i) + (nx+2)*(j) + (nx+2)*(ny+2)
     solar                  = sw_pf(l)
     clm(t)%forc_lwrad      = lw_pf(l)
     prcp                   = prcp_pf(l)
     clm(t)%forc_t          = tas_pf(l)
     clm(t)%forc_u          = u_pf(l)
     clm(t)%forc_v          = v_pf(l)
     clm(t)%forc_pbot       = patm_pf(l)
     clm(t)%forc_q          = qatm_pf(l)
     !clm(t)%slope_x         = slope_x_pf(l)
     !clm(t)%slope_y         = slope_y_pf(l)
	 ! BH: added the option for forcing or not the vegetation
	if  (clm_forc_veg== 1) then 
		clm(t)%elai	        = lai_pf(l)
		clm(t)%esai	        = sai_pf(l)	
		clm(t)%z0m	        = z0m_pf(l) 
		clm(t)%displa	    = displa_pf(l)     
	endif
	 
     !Treat air density
     clm(t)%forc_rho        = clm(t)%forc_pbot/(clm(t)%forc_t*2.8704e2)

     !Treat solar (SW)
     clm(t)%forc_solad(1)   = solar*35./100.   !forc_sols
     clm(t)%forc_solad(2)   = solar*35./100.   !forc_soll
     clm(t)%forc_solai(1)   = solar*15./100.   !forc_solsd
     clm(t)%forc_solai(2)   = solar*15./100.   !forc_solad
     
     !Treat precip
     !(Set upper limit of air temperature for snowfall at 275.65K.
     ! This cut-off was selected based on Fig. 1, Plate 3-1, of Snow
     ! Hydrology (1956)).
     if (prcp > 0.) then
        if(clm(t)%forc_t > (tfrz + tcrit))then
           clm(t)%itypprc   = 1
           clm(t)%forc_rain = prcp
           clm(t)%forc_snow = 0.
        else
           clm(t)%itypprc   = 2
           clm(t)%forc_rain = 0.
           clm(t)%forc_snow = prcp
        endif
     else
        clm(t)%itypprc      = 0
        clm(t)%forc_rain    = 0.
        clm(t)%forc_snow    = 0
     endif
  enddo

end subroutine drv_getforce
