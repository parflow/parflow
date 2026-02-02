!#include <misc.h>

subroutine clm_hydro_canopy (clm)

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
  !  Calculation of 
  !  (1) water storage of intercepted precipitation
  !  (2) direct throughfall and canopy drainage of precipitation
  !  (3) the fraction of foliage covered by water and the fraction
  !      of foliage that is dry and transpiring. 
  !  (4) snow layer initialization if the snow accumulation exceeds 10 mm.
  !
  ! Note:  The evaporation loss is taken off after the calculation of leaf 
  ! temperature in the subroutine clm_leaftem.f90 not in this subroutine.
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
  !=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : tfrz, istice, istwet, istsoil
  implicit none

  ! Parameters for saturation vapor pressure calculation
  real(r8), parameter :: es_a = 611.2d0      ! [Pa] reference saturation vapor pressure
  real(r8), parameter :: es_b = 17.67d0      ! coefficient for Clausius-Clapeyron
  real(r8), parameter :: es_c = 243.5d0      ! [C] coefficient for Clausius-Clapeyron

  !=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm        !CLM 1-D Module

  !=== Local Variables =====================================================

  real(r8)  &
       prcp,              & ! precipitation rate [mm/s]
       h2ocanmx,          & ! maximum allowed water on canopy [mm]
       fpi,               & ! coefficient of interception
       vegt,              & ! frac_veg_nosno*lsai
       xrun,              & ! excess water that exceeds the leaf capacity [mm/s]
       qflx_candrip,      & ! rate of canopy runoff and snow falling off canopy [mm/s]
       qflx_through,      & ! direct throughfall [mm/s]
       dz_snowf,          & ! layer thickness rate change due to precipitation [mm/s]
       flfall,            & ! fraction of liquid water within falling precip.
       bifall               ! bulk density of newly fallen dry snow [kg/m3]

  integer  &
       newnode              ! signification when new snow node is set, (1=yes, 0=non)

  real(r8)    :: &
       dewmxi               ! inverse of maximum allowed dew [1/mm]

  ! Variables for wetbulb rain-snow partitioning @RMM 2025
  real(r8) :: t_c             ! air temperature in Celsius
  real(r8) :: t_wb            ! wetbulb temperature in Celsius
  real(r8) :: t_wb_k          ! wetbulb temperature in Kelvin
  real(r8) :: rh_pct          ! relative humidity in percent
  real(r8) :: e_sat           ! saturation vapor pressure [Pa]
  real(r8) :: e_act           ! actual vapor pressure [Pa]
  real(r8) :: q_sat           ! saturation specific humidity [kg/kg]
  real(r8) :: psnow           ! probability/fraction of snow [-]
  real(r8) :: exponent        ! exponent for logistic function

  !=== End Variable List ===================================================
  
  qflx_candrip = 0.
  qflx_through = 0.

  ! ========================================================================
  ! [1] Canopy interception and precipitation onto ground surface
  ! ========================================================================

  ! IMF -- Add spray irrigation to rain rate
  !        (NOTE: Add directly to clm%forc_rain, not prcp)
  !        (NOTE: qflx_qirr is set to 0. UNLESS cycle,time,veg-type match up...see clm_hydro_irrig)
  if ( (clm%irrig==1) .and. (clm%irr_type==1) ) then
     clm%forc_rain = clm%forc_rain + clm%qflx_qirr              ! spray added to rain rate (above canopy)
  endif   

  ! 1.1 Add precipitation to leaf water 

  if (clm%itypwat==istsoil .OR. clm%itypwat==istwet) then       ! soil or wetland point

     qflx_candrip = 0.                 ! rate of canopy runoff
     qflx_through = 0.                 ! precipitation direct through canopy
     clm%qflx_prec_intr = 0.           ! intercepted precipitation  

     ! IMF: prcp is now total water applied to canopy -- rain + snow + spray irrig.
     prcp = clm%forc_rain + clm%forc_snow    ! total precipitation 

     if (clm%frac_veg_nosno == 1 .AND. prcp > 0.) then

        ! The leaf water capacities for solid and liquid are different, 
        ! generally double for snow, but these are of somewhat less significance
        ! for the water budget because of lower evap. rate at lower temperature.
        ! Hence, it is reasonable to assume that vegetation storage of solid water 
        ! is the same as liquid water.

        h2ocanmx = clm%dewmx * clm%frac_veg_nosno * (clm%elai + clm%esai)

        ! Direct throughfall

        fpi = 0.25d0*(1. - exp(-0.5*(clm%elai + clm%esai)))
        qflx_through  = prcp*(1.-fpi)*clm%frac_veg_nosno

        ! Water storage of intercepted precipitation and dew

        clm%qflx_prec_intr = prcp*fpi*clm%frac_veg_nosno
        clm%h2ocan = max(dble(0.), clm%h2ocan + clm%dtime*clm%qflx_prec_intr)

        ! Initialize rate of canopy runoff and snow falling off canopy

        qflx_candrip = 0.0

        ! Excess water that exceeds the leaf capacity

        xrun = (clm%h2ocan - h2ocanmx)/clm%dtime

        ! Test on maximum dew on leaf

        if (xrun > 0.) then
           qflx_candrip = xrun
           clm%h2ocan   = h2ocanmx
        endif

     endif

  else if (clm%itypwat == istice) then  !land ice

     clm%qflx_prec_intr = 0.
     clm%h2ocan   = 0.
     qflx_candrip = 0.
     qflx_through = 0.  

  endif

  ! IMF -- Add drip irrigation to throughfall
  !        (NOTE: Add directly to throughfall, bypassing canopy)
  !        (NOTE: qflx_qirr is set to 0. UNLESS cycle,time,veg-type match up...see clm_hydro_irrig)
  if ( (clm%irrig==1) .and. (clm%irr_type==2) ) then
     qflx_through = qflx_through + clm%qflx_qirr             ! Drip irrigation added to throughfall, below canopy 
  endif

  ! 1.2 Precipitation onto ground (kg/(m2 s))

  if (clm%frac_veg_nosno == 0) then
     clm%qflx_prec_grnd   = clm%forc_rain + clm%forc_snow
  else
     clm%qflx_prec_grnd   = qflx_through  + qflx_candrip  
  endif

  ! 1.3 The percentage of liquid water by mass, which is arbitrarily set to
  !     vary linearly with air temp, from 0% at 273.16 to 40% max at 275.16.
  !     @RMM 2025: Only use itypprc shortcut for CLM default method (type 0).
  !     Wetbulb methods (type 1,2) need to run regardless of itypprc to catch
  !     cases where air temp is warm but wetbulb is cold (humid conditions).

  if (clm%itypprc <= 1 .and. clm%snow_partition_type == 0) then
     flfall = 1.                              ! fraction of liquid water within falling precip.
     clm%qflx_snow_grnd = 0.                  ! ice onto ground (mm/s)
     clm%qflx_rain_grnd = clm%qflx_prec_grnd  ! liquid water onto ground (mm/s)
     dz_snowf = 0.                            ! rate of snowfall, snow depth/s (m/s)
  else
     ! @RMM 2025: Select rain-snow partitioning method
     ! snow_partition_type: 0=CLM linear, 1=wetbulb thresh, 2=wetbulb linear, 3=Dai, 4=Jennings
     select case (clm%snow_partition_type)

     case (1, 2)  ! Wetbulb-based methods
        ! Calculate wetbulb temperature using Stull (2011) psychrometric approximation
        ! First convert air temp to Celsius
        t_c = clm%forc_t - tfrz

        ! Calculate saturation vapor pressure using Clausius-Clapeyron
        e_sat = es_a * exp(es_b * t_c / (t_c + es_c))

        ! Calculate saturation specific humidity
        q_sat = 0.622d0 * e_sat / (clm%forc_pbot - 0.378d0 * e_sat)

        ! Calculate relative humidity from specific humidity
        ! Avoid division by zero and cap at 100%
        if (q_sat > 0.0d0) then
           rh_pct = min(100.0d0, max(0.0d0, 100.0d0 * clm%forc_q / q_sat))
        else
           rh_pct = 100.0d0
        endif

        ! Stull (2011) wet-bulb temperature approximation (result in Celsius)
        ! Valid for RH >= 5% and T between -20C and 50C
        t_wb = t_c * atan(0.151977d0 * sqrt(rh_pct + 8.313659d0)) &
             + atan(t_c + rh_pct) &
             - atan(rh_pct - 1.676331d0) &
             + 0.00391838d0 * (rh_pct**1.5d0) * atan(0.023101d0 * rh_pct) &
             - 4.686035d0

        ! Convert wetbulb to Kelvin for comparison
        t_wb_k = t_wb + tfrz

        if (clm%snow_partition_type == 1) then
           ! Wetbulb threshold method: all snow below threshold, all rain above
           if (t_wb_k <= clm%tw_threshold) then
              flfall = 0.0d0  ! all snow
           else
              flfall = 1.0d0  ! all rain
           endif
        else  ! case 2: wetbulb linear
           ! Linear transition over configurable range centered on threshold
           if (t_wb_k <= clm%tw_threshold - clm%snow_transition_width) then
              flfall = 0.0d0
           else if (t_wb_k >= clm%tw_threshold + clm%snow_transition_width) then
              flfall = 1.0d0
           else
              flfall = (t_wb_k - (clm%tw_threshold - clm%snow_transition_width)) / &
                       (2.0d0 * clm%snow_transition_width)
           endif
        endif

     case (3)  ! Dai (2008) sigmoidal method
        ! F(%) = a * [tanh(b*(T-c)) - d], converted to fraction by /100
        ! T in Celsius, coefficients from Table 1a (Land, ANN)
        ! Reference: Dai (2008) GRL doi:10.1029/2008GL033295
        t_c = clm%forc_t - tfrz
        psnow = (clm%dai_a / 100.0d0) * &
                (tanh(clm%dai_b * (t_c - clm%dai_c)) - clm%dai_d)
        psnow = max(0.0d0, min(1.0d0, psnow))
        flfall = 1.0d0 - psnow

     case (4)  ! Jennings et al. (2018) bivariate logistic method
        ! psnow = 1 / (1 + exp(a + b*T + g*RH))
        ! T in Celsius, RH in percent
        ! Reference: Jennings et al. (2018) Nat Commun doi:10.1038/s41467-018-03629-7
        t_c = clm%forc_t - tfrz

        ! Calculate saturation vapor pressure using Clausius-Clapeyron
        e_sat = es_a * exp(es_b * t_c / (t_c + es_c))

        ! Calculate saturation specific humidity
        q_sat = 0.622d0 * e_sat / (clm%forc_pbot - 0.378d0 * e_sat)

        ! Calculate relative humidity from specific humidity
        if (q_sat > 0.0d0) then
           rh_pct = min(100.0d0, max(0.0d0, 100.0d0 * clm%forc_q / q_sat))
        else
           rh_pct = 100.0d0
        endif

        ! Bivariate logistic regression
        exponent = clm%jennings_a + clm%jennings_b * t_c + clm%jennings_g * rh_pct
        psnow = 1.0d0 / (1.0d0 + exp(exponent))
        flfall = 1.0d0 - psnow

     case default  ! Case 0: CLM default linear (air temperature) with configurable thresholds
        ! Now uses configurable snow_t_low and snow_t_high instead of hardcoded values
        if (clm%forc_t <= clm%snow_t_low) then
           flfall = 0.0d0
        else if (clm%forc_t >= clm%snow_t_high) then
           flfall = 0.4d0
        else
           flfall = 0.4d0 * (clm%forc_t - clm%snow_t_low) / &
                    (clm%snow_t_high - clm%snow_t_low)
        endif

     end select

     ! Use Alta relationship, Anderson(1976); LaChapelle(1961), 
     ! U.S.Department of Agriculture Forest Service, Project F, 
     ! Progress Rep. 1, Alta Avalanche Study Center:Snow Layer Densification.

     if (clm%forc_t > tfrz + 2.) then
        bifall =189.
     else if (clm%forc_t > tfrz - 15.) then
        bifall=50. + 1.7*(clm%forc_t - tfrz + 15.)**1.5
     else
        bifall=50.
     endif

     clm%qflx_snow_grnd = clm%qflx_prec_grnd*(1.-flfall)             
     clm%qflx_rain_grnd = clm%qflx_prec_grnd*flfall 
     dz_snowf = clm%qflx_snow_grnd/bifall                
     clm%snowdp = clm%snowdp + dz_snowf*clm%dtime         
     clm%h2osno = clm%h2osno + clm%qflx_snow_grnd*clm%dtime      ! snow water equivalent (mm)

     if (clm%itypwat==istwet .AND. clm%t_grnd>=tfrz) then
        clm%h2osno=0. 
        clm%snowdp=0. 
        clm%snowage=0.
     endif

  endif

  ! ========================================================================
  ! [2] Determine the fraction of foliage covered by water and the 
  !     fraction of foliage that is dry and transpiring.
  ! ========================================================================

  ! fwet is the fraction of all vegetation surfaces which are wet 
  ! including stem area which contribute to evaporation
  ! fdry is the fraction of elai (***now in LSM***) which is dry because only leaves
  ! can transpire.  Adjusted for stem area which does not transpire.

  if (clm%h2ocan > 0. .and. clm%frac_veg_nosno == 1) then
     vegt     = clm%frac_veg_nosno*(clm%elai + clm%esai)
     dewmxi   = 1.0/clm%dewmx
     clm%fwet = ((dewmxi/vegt)*clm%h2ocan)**.666666666666
     clm%fwet = min(clm%fwet,dble(1.0))     ! Check for maximum limit of fwet
     clm%fdry = (1.-clm%fwet)*clm%elai/(clm%elai+clm%esai)
  else
     clm%fwet = 0.
     clm%fdry = 0.
  endif

  ! ========================================================================
  ! [3] When the snow accumulation exceeds 10 mm, initialize snow layer
  ! ========================================================================

  ! Currently, the water temperature for the precipitation is simply set 
  ! as the surface air temperature

  newnode = 0    ! signification when snow node will be initialized
  if (clm%snl == 0 .AND. clm%qflx_snow_grnd > 0.0 .AND. clm%snowdp >= 0.01) then  
     newnode = 1
     clm%snl = -1
     clm%dz(0) = clm%snowdp                       ! meter
     clm%z(0) = -0.5*clm%dz(0)
     clm%zi(-1) = -clm%dz(0)
     clm%snowage = 0.                             ! snow age
     clm%t_soisno (0) = min(tfrz, clm%forc_t)     ! K
     clm%h2osoi_ice(0) = clm%h2osno               ! kg/m2
     clm%h2osoi_liq(0) = 0.                       ! kg/m2
     clm%frac_iceold(0) = 1.
  endif

  ! The change of ice partial density of surface node due to precipitation
  ! only ice part of snowfall is added here, the liquid part will be added later

  if (clm%snl < 0 .AND. newnode == 0) then
     clm%h2osoi_ice(clm%snl+1) = clm%h2osoi_ice(clm%snl+1)+clm%dtime*clm%qflx_snow_grnd
     clm%dz(clm%snl+1) = clm%dz(clm%snl+1)+dz_snowf*clm%dtime
  endif

end subroutine clm_hydro_canopy
