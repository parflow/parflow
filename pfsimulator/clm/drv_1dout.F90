!#include <misc.h>

subroutine drv_1dout (drv, tile, clm,clm_write_logs)

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
!  Average CLM domain output and write 1-D CLM results. 
!
!  NOTE:Due to the complexity of various 2-D file formats, we have
!       excluded its treatment here, leaving it to the user's design.  
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: drv_1dout.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_tilemodule      ! Tile-space variables
  use drv_module          ! 1-D Land Model Driver variables
  use clmtype             ! 1-D CLM variables
  use clm_varpar, only : nlevsoi, nlevsno
  use clm_varcon, only : hvap
  implicit none

!=== Arguments =============================================================

  type (drvdec)  :: drv              
  type (tiledec) :: tile(drv%nch)
  type (clm1d)   :: clm (drv%nch)
  integer :: clm_write_logs

!=== Local Variables =======================================================

  integer  :: n,i,t                        ! Temporary counters
  integer  :: mask(drv%nch)                ! Spatial averaging water mask
  real(r8) :: real_snl(drv%nch)            ! Real value of snow layers
  real(r8) :: real_frac_veg_nosno(drv%nch) ! Rreal value of snow free frac veg
  real(r8) :: drv_gridave                  ! Spatial Averaging Function
  real(r8) :: dz        (drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: t_soisno  (drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: h2osoi_liq(drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: h2osoi_ice(drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: totsurf,topsoil,surface,evapor,infiltr,fraction,evap_tot, &  !@ for diagnostics
              evap_soi,evap_veg,tran_veg,ice_layer1
  real     :: clm_error

!=== End Variable List ===================================================

  n=drv%nch
!  c=drv%nc
!  r=drv%nr

  do t=1,drv%nch 
!#if (defined GRID_AVERAGE_NONSOIL)     
     !all points will be grid averaged, inluding lakes, wetlands and land ice
     mask(t) = 1.
!#else
     ! lakes, wetlands and land-ice will not be grid averaged
     if (clm(t)%lakpoi) then
        mask(t) = 0
     else
        mask(t) = 1
     endif
!#endif
  end do

  do t = 1,drv%nch 
     do i = -nlevsno+1,nlevsoi
        dz(t,i)         = clm(t)%dz(i)
        t_soisno(t,i)   = clm(t)%t_soisno(i)
        h2osoi_liq(t,i) = clm(t)%h2osoi_liq(i)
        h2osoi_ice(t,i) = clm(t)%h2osoi_ice(i)
     enddo
  enddo

  real_snl = real(clm%snl)
  real_frac_veg_nosno = real(clm%frac_veg_nosno)

  if (clm_write_logs==1) then  ! NBE: Allow interrupt of writing the log files
!  write(20)  drv%time, drv%ss, drv%mn, drv%hr, drv%da, drv%mo, drv%yr                          ! [1] Time
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%forc_solad(1)+clm%forc_solad(2)+clm%forc_solai(1)+clm%forc_solai(2))  ! [2]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%sabv)                                           ! [3]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%sabg)                                           ! [4]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%forc_lwrad)                                     ! [5]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%eflx_lwrad_out)                                 ! [6]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_tot)                                    ! [7]  W/m2    
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_veg)                                    ! [8]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_grnd)                                   ! [9]  W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_tot*hvap)                             ! [10] W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_veg*hvap)                             ! [11] W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_tran_veg*hvap)                             ! [12] W/m2
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_soi*hvap)                             ! [13] W/m2
!  write 
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%t_rad)                                          ! [15] K
!  write(20)  drv_gridave (n,mask,tile%fgrd,(clm%forc_rain+clm%forc_snow)*clm%dtime)            ! [16] mm / time-step
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_tot*clm%dtime)                        ! [17] mm / time-step
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_surf*clm%dtime)                            ! [18] mm / time-step
!  write(20)  drv_gridave (n,mask,tile%fgrd,(clm%qflx_drain + clm%qflx_surf)*clm%dtime)         ! [19] mm / time-step
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%h2osno)                                         ! [20] millimeter
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%snowdp)                                         ! [21] meter
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%snowage)                                        ! [22] [added] 
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%h2ocan)                                         ! [23] millimeter
!  write(20)  drv_gridave (n,mask,tile%fgrd,sqrt(clm%forc_u*clm%forc_u+clm%forc_v*clm%forc_v))  ! [24] m/s
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%forc_t)                                         ! [25] K
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%forc_q)                                         ! [26] kg/kg
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%forc_pbot)                                      ! [27] pa
!  write(20)  drv_gridave (n,mask,tile%fgrd,real_frac_veg_nosno)                                ! [28] -
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%frac_sno)                                       ! [29] -
!  write(20)  drv_gridave (n,mask,tile%fgrd,real_snl)                                           ! [30] -
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%t_veg)                                          ! [31] K
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%t_ref2m)                                        ! [32] [added] K         
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%taux)                                           ! [33] [added] kg/m/s**2 
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%tauy)                                           ! [34] [added] kg/m/s**2 
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%qflx_snomelt)                                   ! [35] [added] mm/s      
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%t_grnd)                                         ! [36] [added] K         
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%errh2o)                                         ! [37] mm
!  write(20)  drv_gridave (n,mask,tile%fgrd,clm%acc_errseb)/float(clm%istep)                    ! [38] W/m2
!  write(20) (drv_gridave (n,mask,tile%fgrd,dz        (:,i)), i=1,nlevsoi)   ! [39] meter      NOTE:This statement assumes spatially uniform layering
!  write(20) (drv_gridave (n,mask,tile%fgrd,t_soisno  (:,i)), i=1,nlevsoi)   ! [40] K          NOTE:This statement assumes spatially uniform layering
!  write(20) (drv_gridave (n,mask,tile%fgrd,h2osoi_liq(:,i)), i=1,nlevsoi)   ! [41] millimeter NOTE:This statement assumes spatially uniform layering
!  write(20) (drv_gridave (n,mask,tile%fgrd,h2osoi_ice(:,i)), i=1,nlevsoi)   ! [42] millimeter NOTE:This statement assumes spatially uniform layering

  write(20,*)  drv%time, drv%ss, drv%mn, drv%hr, drv%da, drv%mo, drv%yr, " [1]"                          ! [1] Time
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%forc_solad(1)+clm%forc_solad(2)+clm%forc_solai(1)+clm%forc_solai(2), drv), &
       " [2]"  ! [2]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%sabv, drv), " [3]"                                           ! [3]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%sabg, drv), " [4]"                                           ! [4]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%forc_lwrad, drv), " [5]"                                     ! [5]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%eflx_lwrad_out, drv), " [6]"                                 ! [6]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_tot, drv), " [7]"                                    ! [7]  W/m2    
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_veg, drv), " [8]"                                    ! [8]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_grnd, drv), " [9]"                                   ! [9]  W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_tot*hvap, drv), " [10]"                             ! [10] W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_veg*hvap, drv), " [11]"                             ! [11] W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_tran_veg*hvap, drv), " [12]"                             ! [12] W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_soi*hvap, drv), " [13]"                             ! [13] W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%eflx_soil_grnd, drv), " [14]"                                 ! [14] W/m2
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%t_rad, drv), " [15]"                                          ! [15] K
  write(20,*)  drv_gridave (n,mask,tile%fgrd,(clm%forc_rain+clm%forc_snow)*clm%dtime, drv), " [16]"            ! [16] mm / time-step
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_tot*clm%dtime, drv), " [17]"                        ! [17] mm / time-step
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_surf*clm%dtime, drv), " [18]"                            ! [18] mm / time-step
  write(20,*)  drv_gridave (n,mask,tile%fgrd,(clm%qflx_drain + clm%qflx_surf)*clm%dtime, drv), " [19]"         ! [19] mm / time-step
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%h2osno, drv), " [20]"                                         ! [20] millimeter
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%snowdp, drv), " [21]"                                         ! [21] meter
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%snowage, drv), " [22]"                                        ! [22] [added] 
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%h2ocan, drv), " [23]"                                         ! [23] millimeter
  write(20,*)  drv_gridave (n,mask,tile%fgrd,sqrt(clm%forc_u*clm%forc_u+clm%forc_v*clm%forc_v), drv), " [24]"  ! [24] m/s
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%forc_t, drv), " [25]"                                         ! [25] K
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%forc_q, drv), " [26]"                                         ! [26] kg/kg
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%forc_pbot, drv), " [27]"                                      ! [27] pa
  write(20,*)  drv_gridave (n,mask,tile%fgrd,real_frac_veg_nosno, drv), " [28]"                                ! [28] -
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%frac_sno, drv), " [29]"                                       ! [29] -
  write(20,*)  drv_gridave (n,mask,tile%fgrd,real_snl, drv), " [30]"                                           ! [30] -
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%t_veg, drv), " [31]"                                          ! [31] K
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%t_ref2m, drv), " [32]"                                        ! [32] [added] K         
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%taux, drv), " [33]"                                           ! [33] [added] kg/m/s**2 
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%tauy, drv), " [34]"                                           ! [34] [added] kg/m/s**2 
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%qflx_snomelt, drv), " [35]"                                   ! [35] [added] mm/s      
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%t_grnd, drv), " [36]"                                         ! [36] [added] K         
  write(20,*)  drv_gridave (n,mask,tile%fgrd,clm%errh2o, drv), " [37]"                                         ! [37] mm
  write(20,*) (drv_gridave (n,mask,tile%fgrd,clm%acc_errseb, drv)/float(clm(1)%istep)), " [38]"                    ! [38] W/m2
  write(20,*) (drv_gridave (n,mask,tile%fgrd,dz        (:,i), drv), i=1,nlevsoi), " [39]"   ! [39] meter      NOTE:This statement assumes spatially uniform layering
  write(20,*) (drv_gridave (n,mask,tile%fgrd,t_soisno  (:,i), drv), i=1,nlevsoi), " [40]"   ! [40] K          NOTE:This statement assumes spatially uniform layering
  write(20,*) (drv_gridave (n,mask,tile%fgrd,h2osoi_liq(:,i), drv), i=1,nlevsoi), " [41]"   ! [41] millimeter NOTE:This statement assumes spatially uniform layering
  write(20,*) (drv_gridave (n,mask,tile%fgrd,h2osoi_ice(:,i), drv), i=1,nlevsoi), " [42]"   ! [42] millimeter NOTE:This statement assumes spatially uniform layering
endif ! NBE: End of log file interupt
  
!@== Variables added for diagnostics
  topsoil  = drv_gridave (n,mask,tile%fgrd,clm%qflx_top_soil*clm%dtime, drv)  ! [44] Total water applied at the ground suface [mm]
  surface  = drv_gridave (n,mask,tile%fgrd,clm%qflx_surf*clm%dtime, drv)      ! [45] Surface flow [mm]
  evapor   = drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_grnd*clm%dtime, drv) ! [46] Evaporation from the ground / first soil layer [mm]
  infiltr  = drv_gridave (n,mask,tile%fgrd,clm%qflx_infl*clm%dtime, drv)      ! [47] Infiltration [mm]
  clm_error = drv_gridave (n,mask,tile%fgrd,clm%qflx_top_soil-clm%qflx_surf-clm%qflx_evap_grnd-clm%qflx_infl, drv) 
  evap_tot = drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_tot*clm%dtime, drv)
  evap_veg = drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_veg*clm%dtime, drv)
  evap_soi = drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_soi*clm%dtime, drv)
  tran_veg = drv_gridave (n,mask,tile%fgrd,clm%qflx_tran_veg*clm%dtime, drv)
  ice_layer1 = drv_gridave (n,mask,tile%fgrd,clm%h2osoi_ice(1), drv)
! SGS according to standard "f" must have fw.d format, changed f -> f20.8
  write(2008,'(i5,1x,f20.8,1x,12(e10.2,1x))') &
       clm(1)%istep,drv%time,totsurf,topsoil,surface,evapor,infiltr,fraction,clm_error,evap_tot, &
       evap_veg,evap_soi,tran_veg,ice_layer1
  
end subroutine drv_1dout







