!#include <misc.h>

subroutine drv_almaout (drv,tile,clm)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! drv_almaout.F90:
!
! DESCRIPTION:
!  This subroutine writes ALMA standard output.  Variables that are not
!  captured in the CLM are set to undefined values.
!
! REVISION HISTORY:
!  15 December 2000:  Jon Radakovich; Initial Code
!=========================================================================

! Declare Modules and data structures
  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use clm_varpar, ONLY : nlevsoi, nlevsno
  use clm_varcon, ONLY : hfus,hsub,denice,hvap,istwet,denh2o,tfrz

  implicit none
  type (drvdec)       :: drv
  type (tiledec)      :: tile(drv%nch)
  type (clm1d)        :: clm(drv%nch)

!=== Local Variables =====================================================
  integer :: n,t,i,count      ! Tile space counter
  integer :: mask(drv%nch)      ! Water mask
  real(r8) &
       qg(drv%nch),             & !Ground heat flux Qg [W/m2]
       qf(drv%nch),             & !Energy of fusion Qf [W/m2]
       qv(drv%nch),             & !Energy of sublimation Qv [W/m2]
       qtau(drv%nch),           & !Momentum flux Qtau [N/m2]
       qa(drv%nch),             & !Advective energy Qa [W/m2]
       delsoilheat(drv%nch),    & !Change in soil heat storage DelSoilHeat[W/m2]
       delcoldcont(drv%nch),    & !Change in snow cold content DelColdCont[W/m2]
       qs(drv%nch),             & !Surface runoff Qs [kg/m2s]
       qrec(drv%nch),           & !Recharge Qrec [kg/m2s]
       delsoilmoist(drv%nch),   & !Change in soil moisture DelSoilMoist [kg/m2]
       delswe(drv%nch),         & !Change in snow water equivalent DelSWE [kg/m2]
       delsurfstor(drv%nch),    & !Change in surface water storage DelSurfStor [kg/m2]
       delinterc(drv%nch),      & !Change in interception storage DelIntercept [kg/m2]
       totaldepth(drv%nch),     & !Total depth of soil layers [m]
       snowt(drv%nch),          & !Snow temperature SnowT [K]
       avgsurft(drv%nch),       & !Average surface temperature AvgSurfT [K]
       albedo(drv%nch),         & !Surface albedo Albedo [-]
       surfstor(drv%nch),       & !Surface water storage SurfStor [kg/m2]
       avgwatsat(drv%nch),      & !Depth averaged volumetric soil water at saturation (porosity) [kg/m2]
       swetwilt(drv%nch),       & !Depth averaged wilting point [kg/m2]
       swetint(drv%nch),        & !Depth averaged h2osoi_liq [kg/m2]
       soilwet(drv%nch),        & !Total soil wetness [-]
       ecanop(drv%nch),         & !Interception evaporation ECanop [kg/m2s] 
       ewater(drv%nch),         & !Open water evaporation EWater [kg/m2s]
       rootmoist(drv%nch),      & !Root zone soil moisture RootMoist [kg/m2s] 
       dis(drv%nch),            & !Simulated river discharge [m3/2]
       icefrac(drv%nch),        & !Ice-covered fraction IceFrac [-]
       icet(drv%nch),           & !Sea-ice thickness IceT [m]
       testdepth,               & !Test variable used for calculation of Fdepth and Tdepth [m]
       fdepth(drv%nch),         & !Frozen soil depth Fdepth [m]
       tdepth(drv%nch),         & !Depth to soil thaw Tdepth [m]
       salbedo(drv%nch)           !Snow albedo [-]

  real(r8) :: drv_gridave                              ! Spatial Averaging Function
  real(r8) :: dz        (drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: t_soisno  (drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: h2osoi_liq(drv%nch,-nlevsno+1:nlevsoi)
  real(r8) :: h2osoi_ice(drv%nch,-nlevsno+1:nlevsoi)

!=== End Variable List ===================================================

  n=drv%nch

  do t=1,drv%nch 
!if (defined GRID_AVERAGE_NONSOIL)     
     !all points will be grid averaged, inluding lakes, wetlands and land ice
!     mask(t) = 1.
!else
     ! lakes, wetlands and land-ice will not be grid averaged
     if (clm(t)%lakpoi) then
        mask(t) = 0
     else
        mask(t) = 1
     endif
!endif
  end do

  do t = 1,drv%nch 
     do i = -nlevsno+1,nlevsoi
        dz(t,i)         = clm(t)%dz(i)
        t_soisno(t,i)   = clm(t)%t_soisno(i)
        h2osoi_liq(t,i) = clm(t)%h2osoi_liq(i)
        h2osoi_ice(t,i) = clm(t)%h2osoi_ice(i)
     enddo
  enddo

! ALMA General Energy Balance Components
  qf=hfus*clm%qflx_snomelt
  qv=hsub*clm%qflx_sub_snow

! Calculation for Qg & DelColdCont  
! Even if snow if present Qg is equal to the heat flux between the soil/air interface
! DelColdCont is defined as the change in internal energy of the snow pack
! over a timestep, which is zero when there is no snow.
  do t=1,drv%nch
      if(clm(t)%snl < 0)then
         qg(t)=clm(t)%diffusion
         delcoldcont(t)=clm(t)%eflx_soil_grnd-clm(t)%diffusion
      else
         qg(t)=clm(t)%eflx_soil_grnd
         delcoldcont(t)=0.
      endif
  enddo

  qtau=sqrt((clm%taux*clm%taux)+(clm%tauy*clm%tauy))

! Qa is the heat transferred to snow cover by rain, not represented in CLM
  qa=0.

! DelSoilHeat is always zero because Qg is calculated at the soil/air interface
  delsoilheat=0.

  write(57) drv_gridave (n,mask,tile%fgrd,clm%fsa, drv)                 !Net shortwave radiation SWnet [W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,(-1)*clm%eflx_lwrad_net, drv) !Net longwave radiation LWnet [W/m2] 
  write(57) drv_gridave (n,mask,tile%fgrd,clm%eflx_lh_tot, drv)         !Latent heat flux Qle [W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%eflx_sh_tot, drv)         !Sensible heat flux Qh [W/m2]  
  write(57) drv_gridave (n,mask,tile%fgrd,qg, drv)                      !Ground heat flux Qg [W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,qf, drv)                      !Energy of fusion Qf [W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,qv, drv)                      !Energy of sublimation Qv [W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,qtau, drv)                    !Momentum flux Qtau [N/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,qa, drv)                      !Advective energy Qa [W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,delsoilheat, drv)             !Change in soil heat storage DelSoilHeat[W/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,delcoldcont, drv)             !Change in snow cold content DelColdCont[W/m2]


! ALMA General Water Balance Components
  qs=clm%qflx_surf+clm%qflx_qrgwl-clm%qflx_qirr
  qrec=-9999.0   !Recharge from river to flood plain not a capability in CLM
  do t=1,drv%nch
     delsoilmoist(t)=0.
     totaldepth(t)=0.
     do i=1,nlevsoi
        delsoilmoist(t)=delsoilmoist(t)+ &
                        (h2osoi_ice(t,i)+h2osoi_liq(t,i))-clm(t)%h2osoi_liq_old(i)
        totaldepth(t)=totaldepth(t)+dz(t,i)
     enddo
     delsoilmoist(t)=delsoilmoist(t)
  enddo
  delswe=clm%h2osno-clm%h2osno_old
  delsurfstor=0.
  delinterc=clm%h2ocan-clm%h2ocan_old

  write(57) drv_gridave (n,mask,tile%fgrd,clm%forc_snow, drv)           !Snowfall rate Snowf [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%forc_rain, drv)           !Rainfall rate Rainf [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_tot, drv)       !Total evapotranspiration Evap [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,qs, drv)                      !Surface runoff Qs [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,qrec, drv)                    !Recharge Qrec [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%qflx_drain, drv)          !Subsurface runoff Qsb [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%qflx_snomelt, drv)        !Rate of snowmelt Qsm [kg/m2s]  
  write(57) drv_gridave (n,mask,tile%fgrd,delsoilmoist, drv)            !Change in soil moisture DelSoilMoist [kg/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,delswe, drv)                  !Change in snow water equivalent DelSWE [kg/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,delsurfstor, drv)             !Change in surface water storage DelSurfStor [kg/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,delinterc, drv)               !Change in interception storage DelIntercept [kg/m2]

! ALMA Surface State Variables

! SnowT is the snow surface temperature, i.e. top layer t_soisno
  do t=1,drv%nch
     snowt(t)=0.
     if (clm(t)%itypwat/=istwet)then 
        if(clm(t)%snl < 0)then
           snowt(t)=t_soisno(t,clm(t)%snl+1)
        endif
     endif
     if(snowt(t)==0.)snowt(t)=-9999.0  !SnowT is undefined when there is no snow
  enddo

! AvgSurfT is the average surface temperature which depends on
! the snow temperature, bare soil temperature and canopy temperature
  do t=1,drv%nch
     if(snowt(t).ne.-9999.0)then
        avgsurft(t)=clm(t)%frac_sno*snowt(t)+clm(t)%frac_veg_nosno*clm(t)%t_veg+  &
                    (1-(clm(t)%frac_sno+clm(t)%frac_veg_nosno))*clm(t)%t_grnd
     else
        avgsurft(t)=clm(t)%frac_veg_nosno*clm(t)%t_veg+ &
                    (1-clm(t)%frac_veg_nosno)*clm(t)%t_grnd
     endif
  enddo

  do t=1,drv%nch
     albedo(t)=clm(t)%surfalb
  enddo

!Surface water storage not captured in CLM
  surfstor=0.

  write(57) drv_gridave (n,mask,tile%fgrd,snowt, drv)                   !Snow temperature SnowT [K]  
  write(57) drv_gridave (n,mask,tile%fgrd,clm%t_veg, drv)               !Vegetation canopy temperature VegT [K]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%t_grnd, drv)              !Temperature of bare soil BaresoilT [K]
  write(57) drv_gridave (n,mask,tile%fgrd,avgsurft, drv)                !Average surface temperature AvgSurfT [K]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%t_rad, drv)               !Surface radiative temperature RadT [K]
  write(57) drv_gridave (n,mask,tile%fgrd,albedo, drv)                  !Surface albedo Albedo [-]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%h2osno, drv)              !Snow water equivalent SWE [kg/m2]
  write(57) drv_gridave (n,mask,tile%fgrd,surfstor, drv)                !Surface water storage SurfStor [kg/m2]

! ALMA Subsurface State Variables

!Average layer soil moisture (liquid+ice) SoilMoist [kg/m2]
  do i=1,nlevsoi
     write(57) drv_gridave (n,mask,tile%fgrd,h2osoi_liq(:,i)+h2osoi_ice(:,i), drv)
  enddo

!Average layer soil temperature SoilTemp [K]
  do i=1,nlevsoi
     write(57) drv_gridave (n,mask,tile%fgrd,t_soisno(:,i), drv)
  enddo

!Average layer liquid moisture LSoilMoist [kg/m2]
  do i=1,nlevsoi
     write(57) drv_gridave (n,mask,tile%fgrd,h2osoi_liq(:,i), drv)
  enddo

!Total soil wetness SoilWet [-]
!SoilWet = (vertically averaged SoilMoist - wilting point)/
!          (vertically averaged layer porosity - wilting point)
!where average SoilMoist is swetint, the wilting point is swetwilt,
!and avgwatsat is average porosity.
!totaldepth represents the totaldepth of all of the layers
  do t=1,drv%nch
     swetwilt(t)=0.
     swetint(t)=0.
     totaldepth(t)=0.
     avgwatsat(t)=0.
     do i=1,nlevsoi
        swetwilt(t)=swetwilt(t) + dz(t,i)*(clm(t)%watsat(i)*((-1)*clm(t)%smpmax/clm(t)%sucsat(i))**(-1/clm(t)%bsw(i)))
        avgwatsat(t)=avgwatsat(t)+dz(t,i)*clm(t)%watsat(i)
        totaldepth(t)=totaldepth(t)+clm(t)%dz(i)
        swetint(t)=swetint(t)+h2osoi_liq(t,i)
     enddo
     swetwilt(t)=swetwilt(t)/totaldepth(t)
     avgwatsat(t)=avgwatsat(t)/totaldepth(t)
     swetint(t)=(swetint(t)/denh2o)/totaldepth(t)     
     soilwet(t)=(swetint(t)-swetwilt(t))/(avgwatsat(t)-swetwilt(t))
  enddo
  write(57) drv_gridave (n,mask,tile%fgrd,soilwet, drv)
    
! ALMA Evaporation Components
!Ecanop is the total evaporation from vegetation - vegetation transpiration
  ecanop=clm%qflx_evap_veg-clm%qflx_tran_veg
!Ewater is not represented in the CLM
  ewater=0.

!Rootmoist is the total soil moisture available for evapotranspiration
  do t=1,drv%nch
     rootmoist(t)=0.
     do i=1,nlevsoi
        rootmoist(t)=rootmoist(t)+h2osoi_liq(t,i)*clm(t)%rootfr(i)
     enddo
  enddo

  write(57) drv_gridave (n,mask,tile%fgrd,ecanop, drv)               !Interception evaporation ECanop [kg/m2s]  
  write(57) drv_gridave (n,mask,tile%fgrd,clm%qflx_tran_veg, drv)    !Vegetation transpiration TVeg [kg/m2s]  
  write(57) drv_gridave (n,mask,tile%fgrd,clm%qflx_evap_grnd, drv)   !Bare soil evaporation ESoil [kg/m2s]  
  write(57) drv_gridave (n,mask,tile%fgrd,ewater, drv)               !Open water evaporation EWater [kg/m2s]
  write(57) drv_gridave (n,mask,tile%fgrd,rootmoist, drv)            !Root zone soil moisture RootMoist [kg/m2s]  
  write(57) drv_gridave (n,mask,tile%fgrd,clm%h2ocan, drv)           !Total canopy water storage CanopInt [kg/m2]  
  write(57) drv_gridave (n,mask,tile%fgrd,clm%qflx_sub_snow, drv)    !Snow sublimation SubSnow [kg/m2s]  
  write(57) drv_gridave (n,mask,tile%fgrd,clm%acond, drv)            !Aerodynamic conductance ACond [m/s]


! ALMA Streamflow 
! Dis is not captured in the CLM
  dis=-9999.0
  write(57) drv_gridave (n,mask,tile%fgrd,dis, drv)                  !Simulated river discharge [m3/s]


! ALMA Cold Season Processes
!Icefrac and IceT are not captured in CLM because there is no representation of sea-ice
  icefrac=-9999.0
  icet=-9999.0

!Fdepth is the frozen soil depth, which is undefined when no frozen soil is calculated
!Fdepth is calculated from the top down
  do t=1,drv%nch
     fdepth(t)=0.
     do i=1,nlevsoi
        testdepth=fdepth(t)
        if(t_soisno(t,i)<=tfrz)then
           fdepth(t)=fdepth(t)+dz(t,i)
        elseif(t_soisno(t,i)>tfrz)then
           EXIT  !If a layer is above freezing then the if statement is exited
        endif
!If the Fdepth does not change then a layer above freezing was encountered and the do loop
!must be exited
        if(testdepth.eq.fdepth(t))EXIT   
     enddo
     if(fdepth(t)==0.)fdepth(t)=-9999.0 
  enddo

!Tdepth is the thawed soil depth, which is undefined if the entire layer is thawed and zero
!when the top layer is frozen
  do t=1,drv%nch
     tdepth(t)=0.
     count=0
     do i=1,nlevsoi
        testdepth=tdepth(t)
        if(t_soisno(t,i)>tfrz)then
           count=count+1
           tdepth(t)=tdepth(t)+dz(t,i)
        elseif(t_soisno(t,i)<=tfrz)then
           EXIT  !If a layer is below freezing then the if statement is exited
        endif    
!If the Tdepth does not change then a layer below freezing was encountered and the do loop
!must be exited   
        if(testdepth.eq.tdepth(t))EXIT
     enddo
     if(count==nlevsoi)then   !Test to see if all layers are thawed
        tdepth(t)=-9999.0
     endif
  enddo

  do t=1,drv%nch
     salbedo(t)=clm(t)%snoalb
  enddo
        
  write(57) drv_gridave (n,mask,tile%fgrd,clm%frac_sno, drv)            !Snow covered fraction SnowFrac [-]
  write(57) drv_gridave (n,mask,tile%fgrd,icefrac, drv)                 !Ice-covered fraction IceFrac [-]
  write(57) drv_gridave (n,mask,tile%fgrd,icet, drv)                    !Sea-ice thickness IceT [m]
  write(57) drv_gridave (n,mask,tile%fgrd,fdepth, drv)                  !Frozen soil depth Fdepth [m]
  write(57) drv_gridave (n,mask,tile%fgrd,tdepth, drv)                  !Depth to soil thaw Tdepth [m]
  write(57) drv_gridave (n,mask,tile%fgrd,salbedo, drv)                 !Snow albedo [-]
  write(57) drv_gridave (n,mask,tile%fgrd,clm%snowdp, drv)              !Depth of snow layer SnowDepth [m]

end subroutine drv_almaout
