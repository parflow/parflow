!#include <misc.h>

subroutine clm_leaftem (z0mv,       z0hv,       z0qv,           &
                        thm,        th,         thv,            & 
                        tg,         qg,         dqgdT,          &
                        htvp,       sfacx,      dqgmax,         &
                        emv,        emg,        dlrad,          &
                        ulrad,      cgrnds,     cgrndl,         &
                        cgrnd,      soil_beta,      clm)

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
!  This subroutine:
!  1. Calculates the leaf temperature: 
!     Use the Newton-Raphson iteration to solve for the foliage 
!     temperature that balances the surface energy budget:
!
!     f(t_veg) = Net radiation - Sensible - Latent = 0
!     f(t_veg) + d(f)/d(t_veg) * dt_veg = 0     (*)
!
!  Note:
!  (1) In solving for t_veg, t_grnd is given from the previous timestep.
!  (2) The partial derivatives of aerodynamical resistances, which cannot 
!      be determined analytically, are ignored for d(H)/dT and d(LE)/dT
!  (3) The weighted stomatal resistance of sunlit and shaded foliage is used 
!  (4) Canopy air temperature and humidity are derived from => Hc + Hg = Ha
!                                                           => Ec + Eg = Ea
!  (5) Energy loss is due to: numerical truncation of energy budget equation
!      (*); and "ecidif" (see the code) which is dropped into the sensible 
!      heat 
!  (6) The convergence criteria: the difference, del = t_veg(n+1)-t_veg(n) and 
!      del2 = t_veg(n)-t_veg(n-1) less than 0.01 K, and the difference of 
!      water flux from the leaf between the iteration step (n+1) and (n) 
!      less than 0.1 W/m2; or the iterative steps over 40.
!
!  2. Calculates the leaf fluxes, transpiration, photosynthesis and 
!     updates the dew accumulation due to evaporation.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_leaftem.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

! Declare Modules and data structures

  use precision
  use clmtype
  use clm_varcon, only : sb, cpair, hvap, vkc, grav
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm !CLM 1-D Module

  real(r8), intent(in) :: &
       htvp              ! latent heat of evaporation (/sublimation) [J/kg]

! vegetation parameters
  real(r8), intent(in) ::    &
       z0mv,              & ! roughness length, momentum [m]
       z0hv,              & ! roughness length, sensible heat [m]
       z0qv                 ! roughness length, latent heat [m]

! input variables
  real(r8), intent(in) ::    &
       thm,               & ! intermediate variable (forc_t+0.0098*forc_hgt_t)
       th,                & ! potential temperature (kelvin)
       thv                  ! virtual potential temperature (kelvin)

  real(r8), intent(in) ::    &
       tg,                & ! ground surface temperature [K]
       qg,                & ! specific humidity at ground surface [kg/kg]
       dqgdT,             & ! temperature derivative of "qg"
       sfacx,             & ! coefficient for "sfact"
       dqgmax,            & ! max of d(qg)/d(theta)
       emv,               & ! ground emissivity
       emg,               &  ! vegetation emissivity
       soil_beta            ! beta-type formulation for soil resistance / bare soil under canopy evap

  real(r8), intent(inout) :: &
       cgrnd,             & ! deriv. of soil energy flux wrt to soil temp [w/m2/k]
       cgrndl,            & ! deriv of soil latent heat flux wrt soil temp [w/m**2/k]  (RMM fixed)
       cgrnds               ! deriv, of soil sensible heat flux wrt soil temp [w/m2/k]

  real(r8), intent(out) ::   &
       dlrad,             & ! downward longwave radiation blow the canopy [W/m2]
       ulrad                ! upward longwave radiation above the canopy [W/m2]

!=== Local Variables =====================================================

  real(r8) zldis,         & ! reference height "minus" zero displacement heght [m]
       zii,               & ! convective boundary layer height [m]
       zeta,              & ! dimensionless height used in Monin-Obukhov theory
       beta,              & ! coefficient of conective velocity [-]
       wc,                & ! convective velocity [m/s]
       dth,               & ! diff of virtual temp. between ref. height and surface 
       dthv,              & ! diff of vir. poten. temp. between ref. height and surface
       dqh,               & ! diff of humidity between ref. height and surface
       obu,               & ! Monin-Obukhov length (m)
       um,                & ! wind speed including the stability effect [m/s]
       ur,                & ! wind speed at reference height [m/s]
       uaf,               & ! velocity of air within foliage [m/s]
       temp1,             & ! relation for potential temperature profile
       temp2,             & ! relation for specific humidity profile
       ustar,             & ! friction velocity [m/s]
       tstar,             & ! temperature scaling parameter
       qstar,             & ! moisture scaling parameter
       thvstar,           & ! virtual potential temperature scaling parameter
       taf,               & ! air temperature within canopy space [K]
       qaf                  ! humidity of canopy air [kg/kg]

  real(r8) rpp,           & ! fraction of potential evaporation from leaf [-]
       rppdry,            & ! fraction of potential evaporation through transp [-]
       cf,                & ! heat transfer coefficient from leaves [-]
       rb,                & ! leaf boundary layer resistance [s/m]
       ram(2),            & ! aerodynamical resistance [s/m]
       rah(2),            & ! thermal resistance [s/m]
       raw(2),            & ! moisture resistance [s/m]
       wta,               & ! heat conduactance for air [m/s]
       wtg,               & ! heat conduactance for ground [m/s]
       wtl,               & ! heat conduactance for leaf [m/s]
       wta0,              & ! normalized heat conduactance for air [-]
       wtl0,              & ! normalized heat conduactance for leaf [-]
       wtg0,              & ! normalized heat conduactance for ground [-]
       wtal,              & ! normalized heat conductance for air and leaf [-]
       wtgl,              & ! normalized heat conductance for leaf and ground [-]
       wtga,              & ! normalized heat cond. for air and ground  [-]
       wtaq,              & ! latent heat conduactance for air [m/s]
       wtlq,              & ! latent heat conduactance for leaf [m/s]
       wtgq,              & ! latent heat conduactance for ground [m/s]
       wtaq0,             & ! normalized latent heat conduactance for air [-]
       wtlq0,             & ! normalized latent heat conduactance for leaf [-]
       wtgq0,             & ! normalized heat conduactance for ground [-]
       wtalq,             & ! normalized latent heat cond. for air and leaf [-]
       wtglq,             & ! normalized latent heat cond. for leaf and ground [-]
       wtgaq,             & ! normalized latent heat cond. for air and ground [-]
       el,                & ! vapor pressure on leaf surface [pa]
       deldT,             & ! derivative of "el" on "t_veg" [pa/K]
       qsatl,             & ! leaf specific humidity [kg/kg]
       qsatldT,           & ! derivative of "qsatl" on "t_veg"
       air,bir,cir,       & ! atmos. radiation temporary set
       dc1,dc2,           & ! derivative of energy flux [W/m2/K]
       w, csoilcn           ! weight function and revised csoilc - declare  @RMM


  real(r8) delt,          & ! temporary
       delq                 ! temporary

  integer                 & !
       itlef,             & ! counter for leaf temperature iteration [-]
       itmax,             & ! maximum number of iteration [-]
       itmin                ! minimum number of iteration [-]

  real(r8) del,           & ! absolute change in leaf temp in current iteration [K]
       del2,              & ! change in leaf temperature in previous iteration [K]
       dele,              & ! change in latent heat flux from leaf [K]
       delmax,            & ! maximum change in  leaf temperature [K]
       dels,              & ! change in leaf temperature in current iteration [K]
       det,               & ! maximum leaf temp. change in two consecutive iter [K]
       dlemin,            & ! max limit for energy flux convergence [w/m2]
       dtmin,             & ! max limit for temperature convergence [K]
       efeb                 ! latent heat flux from leaf (previous iter) [mm/s]

  real(r8) efpot,         & ! potential latent energy flux [kg/m2/s]
       efe,               & ! water flux from leaf [mm/s]
       efsh,              & ! sensible heat from leaf [mm/s]
       epss                 ! minimum canopy weyness factor [-]

  integer nmozsgn           ! number of times moz changes sign

  real(r8) obuold,           & ! monin-obukhov length from previous iteration
       tlbef,                & ! leaf temperature from previous iteration [K]
       ecidif,               & ! excess energies [W/m2]
       err                     ! balance error

! Constant atmospheric co2 and o2
  real(r8) po2                 ! partial pressure  o2 (mol/mol)
  real(r8) pco2                ! partial pressure co2 (mol/mol)

  data po2,pco2 /0.209,355.e-06/

  real(r8) co2                 ! atmospheric co2 concentration (pa)
  real(r8) o2                  ! atmospheric o2 concentration (pa)

  real(r8) svpts               ! saturation vapor pressure at t_veg (pa)
  real(r8) eah                 ! canopy air vapor pressure (pa)
  real(r8) foln                ! foliage nitrogen (%)

  real(r8) :: mpe = 1.e-6      ! prevents overflow error if division by zero

!=== End Variable List ===================================================

! Initialization

  del   = 0.0  ! change in leaf temperature from previous iteration
  itlef = 0    ! counter for leaf temperature iteration
  efeb  = 0.0  ! latent head flux from leaf for previous iteration

  wtlq = 0.0
  wtlq0 = 0.0
  wtgq0 = 0.0
  wtalq = 0.0
  wtgaq = 0.0
  wtglq = 0.0
  wtaq = 0.0
  wtgq = 0.0
  wtaq0 = 0.0
  wtlq0 = 0.0
  wtgq0 = 0.0
  wtalq = 0.0
  wtgaq = 0.0
  wtglq = 0.0

! Assign iteration parameters

  delmax = 1.0  ! maximum change in  leaf temperature
  itmax  = 40   ! maximum number of iteration
  itmin  = 2    ! minimum number of iteration
  dtmin  = 0.01 ! max limit for temperature convergence
  dlemin = 0.1  ! max limit for energy flux convergence

! Net absorbed longwave radiation by canopy and ground
! =air+bir*t_veg**4+cir*t_grnd**4

  air =   clm%frac_veg_nosno * emv * (1.+(1.-emv)*(1.-emg)) * clm%forc_lwrad
  bir = - clm%frac_veg_nosno * (2.-emv*(1.-emg)) * emv * sb
  cir =   clm%frac_veg_nosno * emv*emg*sb

! Saturated vapor pressure and humidity and their derivation

  call clm_qsadv (clm%t_veg, clm%forc_pbot, el, deldT, qsatl, qsatldT)

! For use Bonan's stomatal resistance scheme
! atmospheric co2 and o2 are currently constants

  co2 = pco2*clm%forc_pbot
  o2  = po2*clm%forc_pbot

! Initialize flux profile

  nmozsgn = 0
  obuold = 0.
  zii=1000.         ! m  (pbl height)
  beta=1.           ! -  (in computing W_*)

  taf = (tg + thm)/2.
  qaf = (clm%forc_q+qg)/2.

  ur = max(dble(1.0),sqrt(clm%forc_u*clm%forc_u+clm%forc_v*clm%forc_v))    ! limit must set to 1.0, otherwise
  dth = thm-taf
  dqh = clm%forc_q-qaf
  dthv = dth*(1.+0.61*clm%forc_q)+0.61*th*dqh
  zldis = clm%forc_hgt_u - clm%displa
  if (zldis < 0.0d0) zldis = 5.0d0  !@ sk Must be consistent with value in clm_obult.F90

  call clm_obuini(ur,thv,dthv,zldis,z0mv,um,obu)

!                 +----------------------------+
!>----------------| BEGIN stability iteration: |-----------------------<
!                 +----------------------------+

  ITERATION : do while (itlef <= itmax) 

     tlbef = clm%t_veg
     del2 = del

! Evaluate stability-dependent variables using moz from prior iteration

     call clm_obult (clm%displa, z0mv, z0hv, z0qv, obu, um, ustar, temp1, temp2, clm)

! Aerodynamic resistance

     ram(1)=1./(ustar*ustar/um)
     rah(1)=1./(temp1*ustar) 
     raw(1)=1./(temp2*ustar) 

! Bulk boundary layer resistance of leaves

     uaf = um*sqrt( 1./(ram(1)*um) )
     cf = 0.01/(sqrt(uaf)*sqrt(clm%dleaf))
     rb = 1./(cf*uaf)

! Aerodynamic resistances raw and rah between heights zpd+z0h and z0hg.
! if no vegetation, rah(2)=0 because zpd+z0h = z0hg.
! (Dickinson et al., 1993, pp.54)
! Weighting the drag coefficient of soil under canopy for changes in canopy density
! per Zeng et al. 2005 JClimate and Lawrence et al. 2007 JHM
!  @BR "alpha" in weighting set to 2, csoilc changed in clm_input.dat to 0.0025
! @CLM Dry Bias

     w = exp(-2*(clm%elai+clm%esai))     !## added this line @RMM
     csoilcn = (vkc/(0.13*(clm%zlnd*uaf/1.5e-5)**0.45))*w + clm%csoilc*(1.-w)  !@RMM
     ram(2) = 0.               ! not used
     rah(2) = 1./(csoilcn*uaf)  !### Changed clm%csoilc to csoilcn
     raw(2) = rah(2)

! Stomatal resistances for sunlit and shaded fractions of canopy.
! should do each iteration to account for differences in eah, tv.

     svpts = el                        ! pa
     eah = clm%forc_pbot * qaf / 0.622 ! pa
     foln = clm%folnvt

     call clm_stomata(mpe      , clm%parsun, svpts     , eah       ,    &
                      thm      , o2        , co2       ,                &
                      clm%btran, rb        , clm%rssun , clm%psnsun,    &
                      clm%qe25 , clm%kc25  , clm%ko25  , clm%vcmx25,    &
                      clm%akc  , clm%ako   , clm%avcmx , clm%bp    ,    &
                      clm%mp   , foln      , clm%folnmx, clm%c3psn , clm)
                                                      
     call clm_stomata(mpe      , clm%parsha, svpts     , eah       ,    &
                      thm      , o2        , co2       ,                &
                      clm%btran, rb        , clm%rssha , clm%psnsha,    &
                      clm%qe25 , clm%kc25  , clm%ko25  , clm%vcmx25,    &
                      clm%akc  , clm%ako   , clm%avcmx , clm%bp    ,    &
                      clm%mp   , foln      , clm%folnmx, clm%c3psn , clm)

! Heat conductance for air, leaf and ground  

     call clm_condch(rah(1),rb,rah(2),wta,wtl,wtg,wta0,wtl0,wtg0, &
          wtal,wtga,wtgl,clm)

! Fraction of potential evaporation from leaf

     if (clm%fdry .gt. 0.0) then
        rppdry  = clm%fdry*rb*(clm%laisun/(rb+clm%rssun) + clm%laisha/(rb+clm%rssha))/clm%elai
     else
        rppdry = 0.0
     endif
     efpot = clm%forc_rho*wtl*(qsatl-qaf)

     if (efpot > 0. .AND. clm%btran > 0.) then

        clm%qflx_tran_veg = efpot*rppdry
        rpp = rppdry + clm%fwet
        epss = 1.e-10

! Check total evapotranspiration from leaves

        rpp = min(rpp, (clm%qflx_tran_veg+clm%h2ocan/clm%dtime)/efpot - epss)

! No transpiration, if potential evaporation from foliage is zero

     else

        rpp = 1.
        clm%qflx_tran_veg = 0.

     endif

! Update conductances for changes in rpp 
! Latent heat conductances for ground and leaf.
! Air has same conductance for both sensible and latent heat.

     call clm_condcq(raw(1),rb,raw(2),rpp,wtaq,wtlq,wtgq,wtaq0, &
          wtlq0,wtgq0,wtalq,wtgaq,wtglq,clm) 

! The partial derivatives of aerodynamical resistance are ignored 
! which cannot be determined analytically. 

     dc1 = clm%forc_rho*cpair*wtl
     dc2 = hvap*clm%forc_rho*wtlq

     efsh = dc1*(wtga*clm%t_veg-wtg0*tg-wta0*thm)
     efe = dc2*(wtgaq*qsatl-wtgq0*qg-wtaq0*clm%forc_q)

! Evaporation flux from foliage

     if (efe*efeb < 0.0) efe  =  0.1*efe
     clm%dt_veg = (clm%sabv + air + bir*clm%t_veg**4 + cir*tg**4 - efsh - efe) &
          / (- 4.*bir*clm%t_veg**3 +dc1*wtga +dc2*wtgaq*qsatldT)
     clm%t_veg = tlbef + clm%dt_veg
     dels = clm%t_veg-tlbef
     del  = abs(dels)
     err = 0.
     if (del > delmax) then
        clm%dt_veg = delmax*dels/del
        clm%t_veg = tlbef + clm%dt_veg
        err = clm%sabv + air + bir*tlbef**3*(tlbef + 4.*clm%dt_veg) &
             + cir*tg**4 - (efsh + dc1*wtga*clm%dt_veg)          &
             - (efe + dc2*wtgaq*qsatldT*clm%dt_veg)
     endif

! Fluxes from leaves to canopy space
! "efe" was limited as its sign changes frequently.  This limit may
! result in an imbalance in "hvap*qflx_evap_veg" and "efe + dc2*wtgaq*qsatldT*dt_veg" 

     efpot = clm%forc_rho*wtl*(wtgaq*(qsatl+qsatldT*clm%dt_veg) &
          -wtgq0*qg-wtaq0*clm%forc_q)
     clm%qflx_evap_veg = rpp*efpot

! Calculation of evaporative potentials (efpot) and
! interception losses; flux in kg m**-2 s-1.  ecidif 
! holds the excess energy if all intercepted water is evaporated
! during the timestep.  This energy is later added to the
! sensible heat flux.

     ecidif = 0.
     if (efpot > 0. .AND. clm%btran > 0.) then
        clm%qflx_tran_veg = efpot*rppdry
     else
        clm%qflx_tran_veg = 0.
     endif
     ecidif = max(dble(0.0), clm%qflx_evap_veg-clm%qflx_tran_veg-clm%h2ocan/clm%dtime)
     clm%qflx_evap_veg = min(clm%qflx_evap_veg,clm%qflx_tran_veg+clm%h2ocan/clm%dtime)

! The energy loss due to above two limits is added to 
! the sensible heat flux.

     clm%eflx_sh_veg = efsh + dc1*wtga*clm%dt_veg + err +hvap*ecidif

! Recalculate leaf saturated vapor pressure (eg) for updated leaf temperature
! and adjust specific humidity (qsatl) proportionately 

     call clm_qsadv(clm%t_veg,clm%forc_pbot,el,deldT,qsatl,qsatldT)

! Update vegetation/ground surface temperature, canopy air temperature, 
! canopy vapor pressure, aerodynamic temperature, and
! Monin-Obukhov stability parameter moz for next iteration. 

     taf = wtg0*tg + wta0*thm + wtl0*clm%t_veg
     qaf = wtlq0*qsatl+wtgq0*qg+clm%forc_q*wtaq0

! Update Monin-Obukhov length and wind speed including the stability effect

     dth = thm-taf       
     dqh = clm%forc_q-qaf

     tstar=temp1*dth
     qstar=temp2*dqh

     dthv=dth*(1.+0.61*clm%forc_q)+0.61*th*dqh

     thvstar=tstar*(1.+0.61*clm%forc_q) + 0.61*th*qstar
     zeta=zldis*vkc*grav*thvstar/(ustar**2*thv)

     if (zeta >= 0.) then     !stable
        zeta = min(dble(2.),max(zeta,dble(0.01)))
     else                     !unstable
        zeta = max(dble(-100.),min(zeta,dble(-0.01)))
     endif
     obu = zldis/zeta

     if (dthv >= 0.) then
        um=max(ur,dble(0.1))
     else
        wc=beta*(-grav*ustar*thvstar*zii/thv)**0.333
        um=sqrt(ur*ur+wc*wc)
     endif

     if (obuold*obu < 0.) nmozsgn = nmozsgn+1
     if (nmozsgn >= 4) then 
        obu = zldis/(-0.01)
     endif

     obuold = obu

! Test for convergence

     itlef = itlef+1
     if (itlef > itmin) then
        dele = abs(efe-efeb)
        efeb = efe
        det  = max(del,del2)
        if (det < dtmin .AND. dele < dlemin) exit 
     endif

! Repeat iteration

  enddo ITERATION

!                   +--------------------------+
!>------------------| END stability iteration: |-----------------------<
!                   +--------------------------+
! Balance check

  err = clm%sabv + air + bir*tlbef**3*(tlbef + 4.*clm%dt_veg) &
       + cir*tg**4 - clm%eflx_sh_veg - hvap*clm%qflx_evap_veg
  if (abs(err) > 0.1) then
     write(6,*) 'energy balance in canopy X',err
  endif

! Fluxes from ground to canopy space 

  delt  = wtal*tg-wtl0*clm%t_veg-wta0*thm
  delq  = wtalq*qg-wtlq0*qsatl-wtaq0*clm%forc_q
  clm%taux  = clm%taux - clm%frac_veg_nosno*clm%forc_rho*clm%forc_u/ram(1)
  clm%tauy  = clm%tauy - clm%frac_veg_nosno*clm%forc_rho*clm%forc_v/ram(1)
  clm%eflx_sh_grnd = clm%eflx_sh_grnd + cpair*clm%forc_rho*wtg*delt
  clm%qflx_evap_soi = clm%qflx_evap_soi +   soil_beta*clm%forc_rho*wtgq*delq
!!print*, 'soil_beta leaftem:',soil_beta
! 2 m height air temperature

  clm%t_ref2m   = clm%t_ref2m + clm%frac_veg_nosno*(taf + temp1*dth * &
       1./vkc *log((2.+z0hv)/z0hv))
  if (delq > 0.) then
     clm%sfact = clm%sfact + clm%forc_rho*(wtgq*wtalq)*sfacx
     clm%sfactmax = clm%sfactmax + clm%forc_rho*(wtgq*wtalq)*dqgmax
  endif

! Downward longwave radiation below the canopy    

  dlrad = clm%frac_veg_nosno*(1.-emv)*emg*clm%forc_lwrad &
       + clm%frac_veg_nosno * emv*emg * sb * &
       tlbef**3*(tlbef + 4.*clm%dt_veg)

! Upward longwave radiation above the canopy    

  ulrad = clm%frac_veg_nosno* ( (1.-emg)*(1.-emv)*(1.-emv)*clm%forc_lwrad &
       + emv*(1.+(1.-emg)*(1.-emv))*sb * tlbef**3 &
       *(tlbef + 4.*clm%dt_veg) + emg *(1.-emv) *sb * tg**4)

! Derivative of soil energy flux with respect to soil temperature (cgrnd) 
! apply soil beta function to ground latent heat flux 
!
  cgrnds = cgrnds + cpair*clm%forc_rho*wtg*wtal
  cgrndl = cgrndl + (clm%forc_rho*wtgq*wtalq*dqgdT)*soil_beta
  cgrnd  = cgrnds + cgrndl*htvp

! Update dew accumulation (kg/m2) 

  clm%h2ocan = max(dble(0.),clm%h2ocan + (clm%qflx_tran_veg-clm%qflx_evap_veg)*clm%dtime)

end subroutine clm_leaftem

