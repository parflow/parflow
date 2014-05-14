!#include <misc.h>

subroutine clm_typini (ntiles, clm, istep_pf)

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
!  initialize clm variables
!
! REVISION HISTORY:
!  15 Jan 2000: Paul Houser; Initial code
!=========================================================================
! $Id: clm_typini.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use infnan
  use clmtype
  use drv_module      ! 1-D Land Model Driver variables
  use drv_tilemodule  ! Tile-space variables
  implicit none

!=== Arguments ===========================================================  

  integer, intent(in)         :: ntiles     !number of tiles
  type (clm1d), intent(inout) :: clm(ntiles)
  integer, intent(in)         :: istep_pf 

!=== Local Variables =====================================================

  integer ::  k

!=== End of Variable Declaration =========================================


  !=== Initialize variables over full tile space
  do k = 1, ntiles

     ! miscellaneous vars...
     clm(k)%itypwat   = bigint   ! water type
     clm(k)%itypprc   = bigint   ! precipitation type (from met data) 1= rain 2 =snow
     clm(k)%isoicol   = bigint   ! color classes for soil albedos
     clm(k)%latdeg    = NaN      ! latitude (degrees)
     clm(k)%londeg    = NaN      ! longitude (degrees)
     clm(k)%dtime     = NaN      ! model time step [second]
     clm(k)%dtime_old = NaN      ! previous model time step
     clm(k)%istep     = istep_pf ! number of time step

     ! leaf constants (read into 2-D grid module variables)
     clm(k)%dewmx     = NaN      ! Maximum allowed dew [mm]

     ! roughness lengths (read into 2-D grid module variables)
     clm(k)%zlnd      = NaN      ! Roughness length for soil [m]
     clm(k)%zsno      = NaN      ! Roughness length for snow [m]
     clm(k)%csoilc    = NaN      ! Drag coefficient for soil under canopy [-]

     ! hydraulic constants of soil (read into 2-D grid module variables)
     clm(k)%wtfact    = NaN      ! Fraction of model area with high water table
     clm(k)%trsmx0    = NaN      ! Max transpiration for moist soil+100% veg. [mm/s]

     ! numerical finite-difference(read into 2-D grid module variables)
     clm(k)%capr      = NaN      ! Tuning factor to turn first layer T into surface T
     clm(k)%cnfac     = NaN      ! Crank Nicholson factor between 0 and 1
     clm(k)%smpmin    = NaN      ! Restriction for min of soil poten. (mm)
     clm(k)%ssi       = NaN      ! Irreducible water saturation of snow
     clm(k)%wimp      = NaN      ! Water impermeable if porosity < wimp
     clm(k)%pondmx    = NaN      ! Ponding depth (mm)

     ! vegetation static, dynamic, derived parameters
     clm(k)%fdry      = NaN      ! fraction of foliage that is green and dry [-]
     clm(k)%fwet      = NaN      ! fraction of foliage covered by water [-]
     clm(k)%tlai      = NaN      ! time interpolated leaf area index
     clm(k)%tsai      = NaN      ! time interpolated stem area index
     clm(k)%elai      = NaN      ! exposed leaf area index
     clm(k)%esai      = NaN      ! exposed stem area index
     clm(k)%minlai    = NaN      ! minimum leaf area index
     clm(k)%maxlai    = NaN      ! maximum leaf area index
     clm(k)%z0m       = NaN      ! aerodynamic roughness length [m]
     clm(k)%displa    = NaN      ! displacement height [m]
     clm(k)%dleaf     = NaN      ! inverse sqrt of leaf dimension [m**-0.5]
     clm(k)%bkmult    = 1.0d0    ! beetle kill multiplier, init to 1 in case not set @CAP 2014-02-24

     ! soil physical parameters
     clm(k)%bsw   (:) = NaN      ! Clapp and Hornberger "b"
     clm(k)%watsat(:) = NaN      ! volumetric soil water at saturation (porosity)
     clm(k)%hksat (:) = NaN      ! hydraulic conductivity at saturation (mm H2O /s)
     clm(k)%sucsat(:) = NaN      ! minimum soil suction (mm)
     clm(k)%watdry(:) = NaN      ! water content when evapotranspiration stops (new)
     clm(k)%watopt(:) = NaN      ! optimal water content for evapotranspiration (new)
     clm(k)%csol  (:) = NaN      ! heat capacity, soil solids (J/m**3/Kelvin)
     clm(k)%tkmg  (:) = NaN      ! thermal conductivity, soil minerals  [W/m-K]  
     clm(k)%tkdry (:) = NaN      ! thermal conductivity, dry soil       (W/m/Kelvin)
     clm(k)%tksatu(:) = NaN      ! thermal conductivity, saturated soil [W/m-K]  
     clm(k)%rootfr(:) = NaN      ! fraction of roots in each soil layer
     clm(k)%wilting_point      = NaN  
     clm(k)%field_capacity     = NaN  
     clm(k)%res_sat            = NaN  
     clm(k)%vegwaterstresstype = NaN  
     clm(k)%beta_type          = NaN  

     ! irrigation parameters
     clm(k)%irr_type           = NaN
     clm(k)%irr_cycle          = NaN
     clm(k)%irr_rate           = NaN
     clm(k)%irr_start          = NaN
     clm(k)%irr_stop           = NaN
     clm(k)%irr_threshold      = NaN
     clm(k)%threshold_type     = NaN
     clm(k)%irr_flag           = 0.d0  ! initialize irrigation flag to zero 

     ! forcing
     clm(k)%forc_u             = NaN   ! wind speed in eastward direction [m/s]
     clm(k)%forc_v             = NaN   ! wind speed in northward direction [m/s]
     clm(k)%forc_t             = NaN   ! temperature at agcm reference height [kelvin]
     clm(k)%forc_q             = NaN   ! specific humidity at agcm reference height [kg/kg]
     clm(k)%forc_rain          = NaN   ! rain rate [mm/s]
     clm(k)%forc_snow          = NaN   ! snow rate [mm/s]
     clm(k)%forc_pbot          = NaN   ! atmosphere pressure at the surface [pa]
     clm(k)%forc_rho           = NaN   ! density air [kg/m3]
     clm(k)%forc_hgt_u         = NaN   ! observational height of wind [m]
     clm(k)%forc_hgt_t         = NaN   ! observational height of temperature [m]
     clm(k)%forc_hgt_q         = NaN   ! observational height of humidity [m]
     clm(k)%forc_lwrad         = NaN   ! atmospheric infrared (longwave) radiation [W/m2]

     ! main variables needed for restart
     clm(k)%snl                = bigint! number of snow layers
     clm(k)%frac_veg_nosno     = bigint! fraction of veg cover, excluding snow-covered veg (now 0 OR 1) [-]
     clm(k)%zi(:)              = NaN   ! interface level below a "z" level (m)
     clm(k)%dz(:)              = NaN   ! layer depth (m)
     clm(k)%dz_mult(:)         = NaN   ! IMF: dz multiplier from ParFlow
     clm(k)%z(:)               = NaN   ! layer thickness (m)
     clm(k)%t_soisno(:)        = NaN   ! soil + snow layer temperature [K]
     clm(k)%h2osoi_liq(:)      = NaN   ! liquid water (kg/m2)
     clm(k)%h2osoi_ice(:)      = NaN   ! ice lens (kg/m2)
     clm(k)%frac_sno           = NaN   ! fractional snow cover
     clm(k)%t_veg              = NaN   ! leaf temperature [K]
     clm(k)%h2ocan             = NaN   ! depth of water on foliage [kg/m2/s]
     clm(k)%snowage            = NaN   ! non dimensional snow age [-]
     clm(k)%h2osno             = NaN   ! snow mass (kg/m2)
     clm(k)%h2osno_old         = NaN   ! snow mass for previous time step (kg/m2)
     clm(k)%snowdp             = NaN   ! snow depth (m)
     clm(k)%t_grnd             = NaN   ! ground surface temperature [k]

     ! fluxes
     clm(k)%taux               = NaN   ! wind stress: E-W [kg/m/s**2]
     clm(k)%tauy               = NaN   ! wind stress: N-S [kg/m/s**2]
     clm(k)%eflx_lh_tot        = NaN   ! latent heat flux from canopy height to atmosphere [W/2]
     clm(k)%eflx_sh_tot        = NaN   ! sensible heat from canopy height to atmosphere [W/m2]
     clm(k)%eflx_sh_grnd       = NaN   ! sensible heat flux from ground [W/m2]
     clm(k)%eflx_sh_veg        = NaN   ! sensible heat from leaves [W/m2]
     clm(k)%qflx_evap_tot      = NaN   ! evapotranspiration from canopy height to atmosphere [mm/s]
     clm(k)%qflx_evap_veg      = NaN   ! evaporation+transpiration from leaves [mm/s]
     clm(k)%qflx_evap_soi      = NaN   ! evaporation heat flux from ground [mm/s]
     clm(k)%qflx_tran_veg      = 0.0d0 ! transpiration rate [mm/s]
     clm(k)%qflx_tran_veg_old  = 0.0d0 ! transpiration rate [mm/s]
     clm(k)%eflx_lwrad_out     = NaN   ! outgoing long-wave radiation from ground+canopy
     clm(k)%eflx_soil_grnd     = NaN   ! ground heat flux [W/m2]
     clm(k)%qflx_surf          = NaN   ! surface runoff (mm h2o/s)
     clm(k)%t_ref2m            = NaN   ! 2 m height air temperature [K]
     clm(k)%t_rad              = NaN   ! radiative temperature [K]

     ! diagnostic Variables
     clm(k)%diagsurf(:)        = NaN   ! Surface diagnostics defined by user
     clm(k)%diagsoil(:,:)      = NaN   ! Soil layer diagnostics defined by user
     clm(k)%diagsnow(:,:)      = NaN   ! Snow layer diagnostics defined by user
     clm(k)%surfind            = bigint! Number of surface diagnostic variables
     clm(k)%soilind            = bigint! Number of soil layer diagnostic variables
     clm(k)%snowind            = bigint! Number of snow layer diagnostic variables

     ! hydrology 
     clm(k)%imelt(:)           = bigint! Flag for melting (=1), freezing (=2), Not=0         
     clm(k)%frac_iceold(:)     = NaN   ! Fraction of ice relative to the total water
     clm(k)%sfact              = NaN  ! term for implicit correction to evaporation
     clm(k)%sfactmax           = NaN  ! maximim of "sfact"
     clm(k)%qflx_snow_grnd     = NaN  ! ice onto ground [kg/(m2 s)]
     clm(k)%qflx_rain_grnd     = NaN  ! liquid water onto ground [kg/(m2 s)]
     clm(k)%qflx_evap_grnd     = NaN  ! ground surface evaporation rate (mm h2o/s)
     clm(k)%qflx_dew_grnd      = NaN  ! ground surface dew formation (mm h2o /s) [+]
     clm(k)%qflx_sub_snow      = NaN  ! sublimation rate from snow pack (mm h2o /s) [+]
     clm(k)%qflx_dew_snow      = NaN  ! surface dew added to snow pack (mm h2o /s) [+]
     clm(k)%qflx_snomelt       = NaN  ! rate of snowmelt [kg/(m2 s)]

     ! added to be consistent with LSM
     clm(k)%eflx_snomelt       = NaN 
     clm(k)%rhol(:)            = NaN  ! pft_varcon leaf reflectance  : 1=vis, 2=nir 
     clm(k)%rhos(:)            = NaN  ! pft_varcon stem reflectance  : 1=vis, 2=nir 
     clm(k)%taus(:)            = NaN  ! pft_varcon stem transmittance: 1=vis, 2=nir 
     clm(k)%taul(:)            = NaN  ! pft_varcon leaf transmittance: 1=vis, 2=nir 
     clm(k)%xl                 = NaN  ! pft_varcon leaf/stem orientation index
     clm(k)%vw                 = NaN  ! pft_varcon btran exponent:[(h2osoi_vol-watdry)/(watopt-watdry)]**vw

     ! surface solar radiation 
     clm(k)%rssun              = NaN  ! sunlit stomatal resistance (s/m)
     clm(k)%rssha              = NaN  ! shaded stomatal resistance (s/m)
     clm(k)%psnsun             = NaN  ! sunlit leaf photosynthesis (umol CO2 /m**2/ s) 
     clm(k)%psnsha             = NaN  ! shaded leaf photosynthesis (umol CO2 /m**2/ s)
     clm(k)%laisun             = NaN  ! sunlit leaf area
     clm(k)%laisha             = NaN  ! shaded leaf area
     clm(k)%sabg               = NaN  ! solar radiation absorbed by ground (W/m**2)
     clm(k)%sabv               = NaN  ! solar radiation absorbed by vegetation (W/m**2)
     clm(k)%fsa                = NaN  ! solar radiation absorbed (total) (W/m**2)
     clm(k)%fsr                = NaN  ! solar radiation reflected (W/m**2)
     clm(k)%ndvi               = NaN  ! Normalized Difference Vegetation Index (diagnostic)

     ! surface albedo 
     clm(k)%parsun             = NaN  ! average absorbed PAR for sunlit leaves (W/m**2)
     clm(k)%parsha             = NaN  ! average absorbed PAR for shaded leaves (W/m**2)
     clm(k)%albd(:)            = NaN  ! surface albedo (direct)                     
     clm(k)%albi(:)            = NaN  ! surface albedo (diffuse)                    
     clm(k)%albgrd(:)          = NaN  ! ground albedo (direct)                      
     clm(k)%albgri(:)          = NaN  ! ground albedo (diffuse)                     
     clm(k)%fabd(:)            = NaN  ! flux absorbed by veg per unit direct flux   
     clm(k)%fabi(:)            = NaN  ! flux absorbed by veg per unit diffuse flux  
     clm(k)%ftdd(:)            = NaN  ! down direct flux below veg per unit dir flx 
     clm(k)%ftid(:)            = NaN  ! down diffuse flux below veg per unit dir flx
     clm(k)%ftii(:)            = NaN  ! down diffuse flux below veg per unit dif flx
     clm(k)%fsun               = NaN  ! sunlit fraction of canopy                   
     clm(k)%surfalb            = NaN  ! instantaneous all-wave surface albedo
     clm(k)%snoalb             = NaN  ! instantaneous all-wave snow albedo

     ! hydrology
     clm(k)%h2osoi_vol(:)      = NaN  ! volumetric soil water (0<=h2osoi_vol<=watsat) [m3/m3]
     clm(k)%eff_porosity(:)    = NaN  ! effective porosity = porosity - vol_ice
     clm(k)%pf_flux(:)         = 0.0d0! sink/source flux Parflow initialized as zero
     clm(k)%pf_vol_liq(:)      = 0.0d0! partial volume of liquid water initialized as zero
     clm(k)%qflx_infl          = NaN  ! infiltration (mm H2O /s) 
     clm(k)%qflx_infl_old      = 0.0d0
     clm(k)%qflx_drain         = NaN  ! sub-surface runoff (mm H2O /s) 
     clm(k)%qflx_top_soil      = NaN  ! net water input into soil from top (mm/s)
     clm(k)%qflx_prec_intr     = NaN  ! interception of precipitation [mm/s]
     clm(k)%qflx_prec_grnd     = NaN  ! water onto ground including canopy runoff [kg/(m2 s)]
     clm(k)%qflx_qirr          = 0.0d0! irrigation applied at surface [mm/s] (added to rain or throughfall, depending) 
     clm(k)%qflx_qirr_inst(:)  = 0.0d0! irrigation applied by 'instant' method [mm/s] (added to pf_flux) 
     clm(k)%qflx_qrgwl         = NaN  ! qflx_surf at glaciers, wetlands, lakes
     clm(k)%btran              = NaN  ! transpiration wetness factor (0 to 1) 
     clm(k)%smpmax             = NaN  ! wilting point potential in mm (new)

     clm(k)%eflx_impsoil       = NaN  ! implicit evaporation for soil temperature equation (W/m**2)
     clm(k)%eflx_lh_vege       = NaN  ! veg evaporation heat flux (W/m**2) [+ to atm]
     clm(k)%eflx_lh_vegt       = NaN  ! veg transpiration heat flux (W/m**2) [+ to atm]
     clm(k)%eflx_lh_grnd       = NaN  ! ground evaporation heat flux (W/m**2) [+ to atm]   
     clm(k)%eflx_lwrad_net     = NaN  ! net infrared (longwave) rad (W/m**2) [+ = to atm]

     ! water and energy balance check
     clm(k)%begwb              = NaN  ! water mass begining of the time step
     clm(k)%endwb              = NaN  ! water mass end of the time step
     clm(k)%errh2o             = NaN  ! water conservation error (mm H2O)
     clm(k)%errsoi             = NaN  ! soil/lake energy conservation error (W/m**2)
     clm(k)%errseb             = NaN  ! surface energy conservation error (W/m**2)
     clm(k)%errsol             = NaN  ! solar radiation conservation error (W/m**2)
     clm(k)%errlon             = NaN  ! longwave radiation conservation error (W/m**2)
     clm(k)%acc_errseb         = NaN  ! accumulation of surface energy balance error
     clm(k)%acc_errh2o         = NaN  ! accumulation of water balance error

     !forcing
     clm(k)%forc_solad(:)      = NaN   ! direct beam radiation (vis=forc_sols , nir=forc_soll )
     clm(k)%forc_solai(:)      = NaN   ! diffuse radiation     (vis=forc_solsd, nir=forc_solld)

     ! temperatures
     clm(k)%dt_veg             = NaN   ! change in t_veg, last iteration (Kelvin)
     clm(k)%dt_grnd            = NaN   ! change in t_grnd, last iteration (Kelvin)

     ! new lsm terms from pft_varcon - to avoid indirect indexing
     clm(k)%z0m               = NaN    ! aerodynamic roughness length [m]
     clm(k)%displa            = NaN    ! displacement height [m]
     clm(k)%dleaf             = NaN    ! leaf dimension [m]
     clm(k)%xl                = NaN    ! pft_varcon leaf/stem orientation index
     clm(k)%vw                = NaN    ! pft_varcon btran exponent:[(h2osoi_vol-watdry)/(watopt-watdry)]**vw
     clm(k)%rhol(numrad)      = NaN    ! pft_varcon leaf reflectance  : 1=vis, 2=nir 
     clm(k)%rhos(numrad)      = NaN    ! pft_varcon stem reflectance  : 1=vis, 2=nir 
     clm(k)%taul(numrad)      = NaN    ! pft_varcon leaf transmittance: 1=vis, 2=nir 
     clm(k)%taus(numrad)      = NaN    ! pft_varcon stem transmittance: 1=vis, 2=nir 
     clm(k)%qe25              = NaN    ! quantum efficiency at 25c (umol co2 / umol photon)
     clm(k)%ko25              = NaN    ! o2 michaelis-menten constant at 25c (pa)
     clm(k)%kc25              = NaN    ! co2 michaelis-menten constant at 25c (pa)
     clm(k)%vcmx25            = NaN    ! maximum rate of carboxylation at 25c (umol co2/m**2/s)
     clm(k)%ako               = NaN    ! q10 for ko25
     clm(k)%akc               = NaN    ! q10 for kc25
     clm(k)%avcmx             = NaN    ! q10 for vcmx25
     clm(k)%bp                = NaN    ! minimum leaf conductance (umol/m**2/s)
     clm(k)%mp                = NaN    ! slope for conductance-to-photosynthesis relationship
     clm(k)%folnmx            = NaN    ! foliage nitrogen concentration when f(n)=1 (%)
     clm(k)%folnvt            = NaN    ! foliage nitrogen concentration (%)
     clm(k)%c3psn             = NaN    ! photosynthetic pathway: 0. = c4, 1. = c3

     ! alma output
     clm(k)%diffusion         = NaN  ! heat diffusion through layer zero interface 
     clm(k)%h2osoi_liq_old    = NaN  ! liquid water from previous timestep
     clm(k)%h2ocan_old        = NaN  ! depth of water on foliage from previous timestep
     clm(k)%acond             = NaN  ! aerodynamic conductance (m/s)

     ! overland flow
     clm(k)%frac              = NaN  ! fraction of water becoming surface runoff after some TOPMODEL approach
     
     ! topomasks - parflow-clm couple parameters
     clm(k)%topo_mask(:)      = NaN  
     clm(k)%planar_mask       = 0.0  ! was NaN...init to 0.0 

  end do

end subroutine clm_typini



















