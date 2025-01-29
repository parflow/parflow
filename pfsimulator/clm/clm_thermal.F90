!#include <misc.h>

subroutine clm_thermal (clm)

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
  !  This is the main subroutine to execute the calculation of thermal 
  !  processes and surface fluxes.
  !  (1) Leaf temperature
  !      Foliage energy conservation is given by the foliage energy budget 
  !      equation:
  !                     Rnet - Hf - LEf = 0 
  !      The equation is solved by Newton-Raphson iteration, in which this 
  !      iteration includes the calculation of the photosynthesis and 
  !      stomatal resistance, and the integration of turbulent flux profiles. 
  !      The sensible and latent heat transfer between foliage and atmosphere 
  !      and ground is linked by the equations:  
  !                     Ha = Hf + Hg and Ea = Ef + Eg
  !
  !  (2) Snow and soil temperatures
  !      o The volumetric heat capacity is calculated as a linear combination 
  !        in terms of the volumetric fraction of the constituent phases. 
  !      o The thermal conductivity of soil is computed from 
  !        the algorithm of Johansen (as reported by Farouki 1981), and the 
  !        conductivity of snow is from the formulation used in
  !        SNTHERM (Jordan 1991).
  !      o Boundary conditions:  
  !        F = Rnet - Hg - LEg (top),  F= 0 (base of the soil column).
  !      o Soil / snow temperature is predicted from heat conduction 
  !        in 10 soil layers and up to 5 snow layers. 
  !        The thermal conductivities at the interfaces between two 
  !        neighboring layers (j, j+1) are derived from an assumption that 
  !        the flux across the interface is equal to that from the node j 
  !        to the interface and the flux from the interface to the node j+1. 
  !        The equation is solved using the Crank-Nicholson method and 
  !        results in a tridiagonal system equation.
  !
  !  (3) Phase change (see clm_meltfreeze.F90)
  !
  !  FLOW DIAGRAM FOR clm_thermal.F90
  !
  !  thermal ===> clm_qsadv
  !               clm_obuini
  !               clm_obult
  !               clm_leaftem  
  !                  ===> clm_qsadv    
  !                       clm_obuini  
  !                       clm_obult   
  !                       clm_stomata 
  !                       clm_condch  
  !                       clm_condcq  
  !               clm_thermalk          
  !               clm_tridia            
  !               clm_meltfreeze        
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
  !=========================================================================
  ! $Id: clm_thermal.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : denh2o, denice, roverg, hvap, hsub, &
       rair, cpair, grav, vkc, tfrz, sb, istice, istwet 
  use clm_varpar, only : nlevsoi
  implicit none

  !=== Arguments  =====================================================

  type (clm1d), intent(inout)  :: clm  !CLM 1-D Module

  !=== Local Variables =====================================================

  integer i,j

  real(r8)  &
       at(clm%snl+1 : nlevsoi),    & ! "a" vector for tridiagonal matrix
       bt(clm%snl+1 : nlevsoi),    & ! "b" vector for tridiagonal matrix
       ct(clm%snl+1 : nlevsoi),    & ! "c" vector for tridiagonal matrix
       rt(clm%snl+1 : nlevsoi),    & ! "r" vector for tridiagonal solution
       tg,                         & ! ground surface temperature [K]
       cv(clm%snl+1 : nlevsoi),    & ! heat capacity [J/(m2 K)]
       tk(clm%snl+1 : nlevsoi),    & ! thermal conductivity [W/(m K)]
       tssbef(clm%snl+1 : nlevsoi),& ! soil/snow temperature before update
       qred,                       & ! soil surface relative humidity
       z0mg,                       & ! roughness length over ground, momentum [m]
       z0hg,                       & ! roughness length over ground, sensible heat [m]
       z0qg,                       & ! roughness length over ground, latent heat [m]
       z0mv,                       & ! roughness length over vegetation, momentum [m]
       z0hv,                       & ! roughness length over vegetation, sensible heat [m]
       z0qv                          ! roughness length over vegetation, latent heat [m]

  real(r8)  htvp,                  & ! latent heat of vapor of water (or sublimation) [j/kg]
       fact(clm%snl+1 : nlevsoi),  & ! used in computing tridiagonal matrix
       fn  (clm%snl+1 : nlevsoi),  & ! heat diffusion through the layer interface [W/m2]
       fn1 (clm%snl+1 : nlevsoi),  & ! heat diffusion through the layer interface [W/m2]
       dzm,                        & ! used in computing tridiagonal matrix
       dzp                           ! used in computing tridiagonal matrix

  integer &
       niters,                     & ! maximum number of iterations for surface temperature
       iter,                       & ! iteration index
       nmozsgn                       ! number of times moz changes sign

  real(r8)  beta,                     & ! coefficient of conective velocity [-]
       zii,                        & ! convective boundary height [m]
       zldis,                      & ! reference height "minus" zero displacement heght [m]
       ur,                         & ! wind speed at reference height [m/s]
       th,                         & ! potential temperature (kelvin)
       thm,                        & ! intermediate variable (forc_t+0.0098*forc_hgt_t) 
       thv,                        & ! virtual potential temperature (kelvin)
       dth,                        & ! diff of virtual temp. between ref. height and surface
       dqh,                        & ! diff of humidity between ref. height and surface
       dthv,                       & ! diff of vir. poten. temp. between ref. height and surface
       thvstar,                    & ! virtual potential temperature scaling parameter
       obu,                        & ! monin-obukhov length (m)
       zeta,                       & ! dimensionless height used in Monin-Obukhov theory
       wc,                         & ! convective velocity [m/s]
       um,                         & ! wind speed including the stability effect [m/s]
       temp1,                      & ! relation for potential temperature profile
       temp2,                      & ! relation for specific humidity profile
       ustar,                      & ! friction velocity [m/s]
       tstar,                      & ! temperature scaling parameter
       qstar,                      & ! moisture scaling parameter
       ram,                        & ! aerodynamical resistance [s/m]
       rah,                        & ! thermal resistance [s/m]
       raw,                        & ! moisture resistance [s/m]
       raih,                       & ! temporary variable [kg/m2/s]
       raiw,                       & ! temporary variable [kg/m2/s]
       emg,                        & ! ground emissivity (0.97 for snow, glaciers and water surface; 0.96 for soil and wetland)
       emv,                        & ! vegetation emissivity
       avmuir                        ! ir inverse optical depth per unit leaf area

  real(r8)                         & 
       cgrnd,                      & ! deriv. of soil energy flux wrt to soil temp [w/m2/k]
       cgrndl,                     & ! deriv of soil latent heat flux wrt soil temp [w/m**2/k] (RMM, fixed)
       cgrnds,                     & ! deriv, of soil sensible heat flux wrt soil temp [w/m2/k]
       hs,                         & ! net energy flux into the surface (w/m2)
       dhsdt,                      & ! d(hs)/dT
       eg,                         & ! water vapor pressure at temperature T [pa]
       qsatg,                      & ! saturated humidity [kg/kg]
       degdT,                      & ! d(eg)/dT
       qsatgdT,                    & ! d(qsatg)/dT
       fac,                        & ! soil wetness of surface layer
       psit,                       & ! negative potential of soil
       hr,                         & ! relative humidity
       dqgmax,                     & ! maximum of d(qg)/d(theta)
       sfacx,                      & ! coefficient for "sfact"
       qg,                         & ! ground specific humidity [kg/kg]
       dqgdT,                      & ! d(qg)/dT
       wice0(clm%snl+1 : nlevsoi), & ! ice mass from previous time-step
       wliq0(clm%snl+1 : nlevsoi), & ! liquid mass from previous time-step
       wx,                         & ! patitial volume of ice and water of surface layer
       egsmax,                     & ! max. evaporation which soil can provide at one time step
       egidif,                     & ! the excess of evaporation over "egsmax"
       brr(clm%snl+1 : nlevsoi),   & ! temporary set 
       xmf,                        & ! total latent heat of phase change of ground water
       dlrad,                      & ! downward longwave radiation blow the canopy [W/m2]
       ulrad,                      & ! upward longwave radiation above the canopy [W/m2]
       tinc,                       & ! temperature difference of two time step
       obuold                        ! monin-obukhov length from previous iteration

  real(r8) temp, soil_beta, rz_beta, temp_rz                      !soil_beta, rz_beta, the soil beta function and root zone beta function [-]                                       
  real(r8) cf                        !s m**2/umol -> s/m

  !=== End Variable List ===================================================

  !=========================================================================
  ! [1] Initial set 
  !=========================================================================

  ! Fluxes 

  clm%taux     = 0.
  clm%tauy     = 0.
  clm%eflx_sh_tot    = 0.  
  clm%qflx_evap_tot    = 0.  
  clm%eflx_lh_tot   = 0.  
  clm%eflx_sh_veg    = 0.  
  clm%qflx_evap_veg    = 0.  
  clm%qflx_tran_veg      = 0.  
  clm%eflx_sh_grnd    = 0.
  clm%qflx_evap_soi    = 0.  
  dlrad    = 0.
  ulrad    = 0.
  cgrnds   = 0.
  cgrndl   = 0.
  cgrnd    = 0.
  clm%sfact    = 0.
  clm%sfactmax = 0.
  clm%t_ref2m     = 0.

  !Temperature and water mass from previous time step

  tg = clm%t_soisno(clm%snl+1)
  do i = clm%snl+1, nlevsoi
     tssbef(i) = clm%t_soisno(i)
     wice0(i) = clm%h2osoi_ice(i)
     wliq0(i) = clm%h2osoi_liq(i)
  enddo


  !=========================================================================
  ! [2] Specific humidity and its derivative at ground surface
  !=========================================================================

  qred = 1.
  if (clm%itypwat/=istwet .AND. clm%itypwat/=istice) then ! NOT wetland and ice land
     wx   = (clm%h2osoi_liq(1)/denh2o+clm%h2osoi_ice(1)/denice)/clm%dz(1)
     fac  = min(dble(1.), wx/clm%watsat(1))
     fac  = max( fac, dble(0.01) )
     psit = -clm%sucsat(1) * fac ** (- clm%bsw(1))
     psit = max(clm%smpmin, psit)
     !@ Stefan: replace original psit with values from Parflow
     !    do i=1,nlevsoi
     !@ RMM this need no-longer be a loop, since psit is just set to the top soil layer
     if (clm%pf_press(1)>= 0.0d0)  psit = 0.0d0
     if (clm%pf_press(1) < 0.0d0)  psit = clm%pf_press(1)
     !    enddo  
!RMM
! added beta-type formulation depending on soil moisture, the residual saturation for evap is passed in via parflow
! the type of soil beta is also a PF key
     select case (clm%beta_type)
     case (0)    ! none
     soil_beta = 1.0d0
     case (1)    ! linear
     soil_beta = (clm%pf_vol_liq(1) - clm%res_sat*clm%watsat(1)) /(clm%watsat(1) - clm%res_sat*clm%watsat(1))
     case (2)    ! cosine, like ISBA
     soil_beta = 0.5d0*(1.0d0 - cos(((clm%pf_vol_liq(1) - clm%res_sat*clm%watsat(1)) / & 
                  (clm%watsat(1) - clm%res_sat*clm%watsat(1)))*3.141d0))     
     end select
     
     !print*, 'clm%pf_vol_liq(1)/clm%watsat clm%res_sat clm%watsat(1) :',clm%pf_vol_liq(1)/clm%watsat(1),clm%res_sat,clm%watsat(1)

     if (soil_beta < 0.0) soil_beta = 0.00d0
     if (soil_beta > 1.) soil_beta = 1.d0
!print*,'soil_beta 1:',soil_beta

!LB - Reset beta to one if snow layers are present
     if (clm%snl < 0) soil_beta = 1.0d0
     hr   = dexp(psit/roverg/tg)
     qred = (1.-clm%frac_sno)*hr + clm%frac_sno
  else
     hr   = 0.
  endif

  call clm_qsadv(tg,clm%forc_pbot,eg,degdT,qsatg,qsatgdT)

  qg = qred*qsatg  
  dqgdT = qred*qsatgdT

  sfacx = 0.
  dqgmax = 0.
  if (clm%itypwat/=istwet .AND. clm%itypwat/=istice) then ! NOT wetland and ice land
     sfacx = (1.-clm%frac_sno)*hr*qsatg*clm%bsw(1)/(roverg*tg)
     dqgmax = (1.-qred)/clm%watsat(1) * qsatg
  endif

  if (qsatg > clm%forc_q .AND. clm%forc_q > qred*qsatg) then
     qg = clm%forc_q
     dqgdT = 0.
     sfacx = 0.
     dqgmax = 0.
  endif

  !=========================================================================
  ! [3] Leaf and ground surface temperature and fluxes
  !=========================================================================

  ! 3.1 Propositional variables

  ! Emissivity

  if (clm%h2osno>0. .OR.clm%itypwat==istice) then
     emg = 0.97
  else
     emg = 0.96
  endif
  avmuir=1.
  emv=1.-exp(-(clm%elai+clm%esai)/avmuir)

  ! Latent heat, we arbitrarily assume that the sublimation occurs 
  ! only as h2osoi_liq = 0

  htvp = hvap
  if (clm%h2osoi_liq(clm%snl+1) <= 0. .AND. clm%h2osoi_ice(clm%snl+1) > 0.) htvp = hsub

  ! Roughness length

  if (clm%frac_sno > 0.) then
     z0mg = clm%zsno
     z0hg = z0mg            ! initial set
     z0qg = z0mg            ! initial set
  else
     z0mg = clm%zlnd
     z0hg = z0mg
     z0qg = z0mg
  endif

  z0mv = clm%z0m
  z0hv = z0mv
  z0qv = z0mv

  ! Potential temperature at the reference height

  beta=1.        ! -  (in computing W_*)
  zii = 1000.    ! m  (pbl height)
  thm = clm%forc_t + 0.0098*clm%forc_hgt_t              
  th = clm%forc_t*(100000./clm%forc_pbot)**(rair/cpair)  ! potential T  (forc_t*(forc_psrf/forc_pbot)**(rair/cp))
  thv = th*(1.+0.61*clm%forc_q)                          ! virtual potential T
  ur = max(dble(1.0),sqrt(clm%forc_u*clm%forc_u+clm%forc_v*clm%forc_v))    ! limit must set to 1.0, otherwise,

  ! 3.2 BARE PART
  ! Ground fluxes and temperatures
  ! NOTE: in the current scheme clm%frac_veg_nosno is EITHER 1 or 0

  ! Compute sensible and latent fluxes and their derivatives with respect 
  ! to ground temperature using ground temperatures from previous time step.

  if (clm%frac_veg_nosno == 0) then  

     ! Initialization variables

     nmozsgn = 0
     obuold = 0.
     dth   = thm-tg
     dqh   = clm%forc_q-qg
     dthv  = dth*(1.+0.61*clm%forc_q)+0.61*th*dqh
     zldis = clm%forc_hgt_u-0.
     call clm_obuini(ur,thv,dthv,zldis,z0mg,um,obu)

     ! Evaluated stability-dependent variables using moz from prior iteration

     niters=3
     do iter = 1, niters         ! begin stability iteration
        call clm_obult(0.0d0,z0mg,z0hg,z0qg,obu,um,ustar,temp1,temp2,clm)
        tstar = temp1*dth
        qstar = temp2*dqh
        z0hg = z0mg/exp(0.13 * (ustar*z0mg/1.5e-5)**0.45)
        z0qg = z0hg

        thvstar=tstar*(1.+0.61*clm%forc_q) + 0.61*th*qstar
        zeta=zldis*vkc*grav*thvstar/(ustar**2*thv)
        if (zeta >= 0.) then     !stable
           zeta = min(dble(2.),max(zeta,dble(0.01)))
        else                     !unstable
           zeta = max(dble(-100.),min(zeta,dble(-0.01)))
        endif

        obu = zldis/zeta

        if (dthv >= 0.) then
           um = max(ur,dble(0.1))
        else
           wc = beta*(-grav*ustar*thvstar*zii/thv)**0.333
           um = sqrt(ur*ur+wc*wc)
        endif

        if (obuold*obu < 0.) nmozsgn = nmozsgn+1
        if (nmozsgn >= 4) EXIT

        obuold = obu
     enddo                       ! end stability iteration

     ! Get derivative of fluxes with respect to ground temperature

     clm%acond = ustar*ustar/um ! Add-in for ALMA output

     ram    = 1./(ustar*ustar/um)
     rah    = 1./(temp1*ustar) 
     raw    = 1./(temp2*ustar) 
     raih   = (1-clm%frac_veg_nosno)*clm%forc_rho*cpair/rah
     raiw   = (1-clm%frac_veg_nosno)*clm%forc_rho/raw          
     cgrnds = raih
! RMM
! apply soil beta to latent heat flux of ground
!
     cgrndl = soil_beta*raiw*dqgdT
     cgrnd  = cgrnds + htvp*cgrndl
     clm%sfact  = raiw*sfacx
     if (dqh >= 0.) clm%sfact = 0.
     clm%sfactmax = raiw*dqgmax

     ! Surface fluxes of momentum, sensible and latent heat
     ! using ground temperatures from previous time step

     clm%taux   = -(1-clm%frac_veg_nosno)*clm%forc_rho*clm%forc_u/ram        
     clm%tauy   = -(1-clm%frac_veg_nosno)*clm%forc_rho*clm%forc_v/ram
! RMM
! apply soil beta 
! to bare soil evaporation
     clm%eflx_sh_grnd  = -raih*dth
     clm%qflx_evap_soi  = -raiw*dqh
! check if this needs to be limited in the case of dew, i.e. don't limit dew, only evap
     !if(clm%qflx_evap_soi <= 0.0) 
     clm%qflx_evap_soi =soil_beta*clm%qflx_evap_soi
!print*, 'Soil Beta 2:', soil_beta
     clm%eflx_sh_tot  = clm%eflx_sh_grnd
     clm%qflx_evap_tot  = clm%qflx_evap_soi

     ! 2 m height air temperature

     clm%t_ref2m=(1-clm%frac_veg_nosno)*(tg+temp1*dth * 1./vkc *log((2.+z0hg)/z0hg))

     ! Equate canopy temperature to air over bareland.
     ! Needed as frac_veg_nosno=0 carried over to next time step

     clm%t_veg = clm%forc_t

     clm%btran = 0.     !needed for history file for bare soil
     cf = clm%forc_pbot/(8.314*thm)*1.e06 
     clm%rssun = 1./clm%bp * cf
     clm%rssha = 1./clm%bp * cf

     ! 3.3 VEGETATED PART
     ! Calculate canopy temperature, latent and sensible fluxes from the canopy,
     ! and leaf water change by evapotranspiration 

  else    

     clm%btran = 0
     temp_rz = 0.
     do i = 1, nlevsoi
        if(clm%h2osoi_liq(i) > 0.0) then
!           temp = ((-150000.0d0 - clm%pf_press(i))/(-150000.0d0) )
!@RMM
! added beta-type formulation depending on soil moisture, the lower value (i.e. the wilting point) is passed in via ParFlow
! and care should be taken to make sure this is set equal to or above residual saturation 
! the *type* of beta is important as well and is passed in as a key via ParFlow
! a root zone average is taken here
           select case (clm%vegwaterstresstype)
           case (0)     ! none
           temp = 1.0d0
           case (1)     ! pressure type
           temp = ((clm%wilting_point*1000.d0 - clm%pf_press(i))/(clm%wilting_point*1000.d0 - clm%field_capacity*1000.d0) )
           case (2)     ! SM type
           temp = (clm%pf_vol_liq(i) - clm%wilting_point*clm%watsat(i)) / &
	            (clm%field_capacity*clm%watsat(i) - clm%wilting_point*clm%watsat(i))
           end select
           if (temp < 0.) temp = 0.
           if (temp > 1.) temp = 1.
           temp_rz = temp ** clm%vw
           clm%soil_resistance(i) = temp_rz    !! @RMM, we store each soil resistnace factor over the soil layers
        else
           temp2 = 0.01d0
        endif
!       temp_rz = temp_rz / float(nlevsoi)
        !!@RMM, T is still based upon the total soil resistance but the option is provided to limit layer-by-layer T over the RZ
        clm%btran = clm%btran + clm%rootfr(i)*clm%soil_resistance(i)
     enddo

!@RMM
! transpiration cutoff depending on soil moisture, the default is only the top soil layer
! if this is distributed (rzwaterstress=1) then clm%soil_resistance(i) set above is used to limit
!  T in each soil layer individually
! option set from a user input via PF
     if (clm%rzwaterstress == 0) then
     if ( (clm%vegwaterstresstype == 1).and.(clm%pf_press(1)<=(clm%wilting_point*1000.d0)) ) clm%btran = 0.0d0
     if ( (clm%vegwaterstresstype == 2).and.(clm%pf_vol_liq(1)<=clm%wilting_point*clm%watsat(1)) ) clm%btran = 0.0d0
     end if

     call clm_leaftem(z0mv,z0hv,z0qv,thm,th,thv,tg,qg,dqgdT,htvp,sfacx,     &
          dqgmax,emv,emg,dlrad,ulrad,cgrnds,cgrndl,cgrnd,soil_beta,clm)

  endif

  !=========================================================================
  ! [4] Ground temperature
  !=======================================================================

  ! 4.1 Thermal conductivity and Heat capacity
  call clm_thermalk(tk,cv,clm)
  ! 4.2 Net ground heat flux into the surface and its temperature derivative

  hs    = clm%sabg + dlrad &
       + (1-clm%frac_veg_nosno)*emg*clm%forc_lwrad - emg*sb*tg**4 &
       - (clm%eflx_sh_grnd+clm%qflx_evap_soi*htvp) 

  dhsdT = - cgrnd - 4.*emg * sb * tg**3

  j       = clm%snl+1
  fact(j) = clm%dtime / cv(j) &
       * clm%dz(j) / (0.5*(clm%z(j)-clm%zi(j-1)+clm%capr*(clm%z(j+1)-clm%zi(j-1))))

  do j = clm%snl+1 + 1, nlevsoi
     fact(j) = clm%dtime/cv(j)
  enddo

  do j = clm%snl+1, nlevsoi - 1
     fn(j) = tk(j)*(clm%t_soisno(j+1)-clm%t_soisno(j))/(clm%z(j+1)-clm%z(j))
  enddo
  fn(nlevsoi) = 0.

  ! 4.3 Set up vector r and vectors a, b, c that define tridiagonal matrix

  j     = clm%snl+1
  dzp   = clm%z(j+1)-clm%z(j)
  at(j) = 0.
  bt(j) = 1+(1.-clm%cnfac)*fact(j)*tk(j)/dzp-fact(j)*dhsdT
  ct(j) =  -(1.-clm%cnfac)*fact(j)*tk(j)/dzp
  rt(j) = clm%t_soisno(j) +  fact(j)*( hs - dhsdT*clm%t_soisno(j) + clm%cnfac*fn(j) )

  do j    = clm%snl+1 + 1, nlevsoi - 1
     dzm   = (clm%z(j)-clm%z(j-1))
     dzp   = (clm%z(j+1)-clm%z(j))

     at(j) =   - (1.-clm%cnfac)*fact(j)* tk(j-1)/dzm
     bt(j) = 1.+ (1.-clm%cnfac)*fact(j)*(tk(j)/dzp + tk(j-1)/dzm)
     ct(j) =   - (1.-clm%cnfac)*fact(j)* tk(j)/dzp

     rt(j) = clm%t_soisno(j) + clm%cnfac*fact(j)*( fn(j) - fn(j-1) )
  enddo

  j     =  nlevsoi
  dzm   = (clm%z(j)-clm%z(j-1))
  at(j) =   - (1.-clm%cnfac)*fact(j)*tk(j-1)/dzm
  bt(j) = 1.+ (1.-clm%cnfac)*fact(j)*tk(j-1)/dzm
  ct(j) = 0.
  rt(j) = clm%t_soisno(j) - clm%cnfac*fact(j)*fn(j-1)


  ! 4.4 Solve for t_soisno

  i = size(at)
  call clm_tridia (i ,at ,bt ,ct ,rt ,clm%t_soisno(clm%snl+1:nlevsoi))

  !=========================================================================
  ! [5] Melting or Freezing 
  !=========================================================================
  do j = clm%snl+1, nlevsoi - 1
     fn1(j) = tk(j)*(clm%t_soisno(j+1)-clm%t_soisno(j))/(clm%z(j+1)-clm%z(j))
  enddo
  fn1(nlevsoi) = 0.

  j = clm%snl+1
  brr(j) = clm%cnfac*fn(j) + (1.-clm%cnfac)*fn1(j)

  do j = clm%snl+1 + 1, nlevsoi
     brr(j) = clm%cnfac*(fn(j)-fn(j-1)) + (1.-clm%cnfac)*(fn1(j)-fn1(j-1))
  enddo

  call clm_meltfreeze (fact(clm%snl+1), brr(clm%snl+1), hs, dhsdT, &
       tssbef(clm%snl+1),xmf, clm)

  tg = clm%t_soisno(clm%snl+1)


  !=========================================================================
  ! [6] Correct fluxes to present soil temperature
  !========================================================================= 

  tinc = clm%t_soisno(clm%snl+1) - tssbef(clm%snl+1)

  clm%eflx_sh_grnd =  clm%eflx_sh_grnd + tinc*cgrnds 
  clm%qflx_evap_soi =  clm%qflx_evap_soi + tinc*cgrndl

  ! Calculation of evaporative potential; flux in kg m**-2 s-1.  
  ! egidif holds the excess energy if all water is evaporated
  ! during the timestep.  This energy is later added to the
  ! sensible heat flux.

  egsmax = (clm%h2osoi_ice(clm%snl+1)+clm%h2osoi_liq(clm%snl+1)) / clm%dtime

  egidif = max(dble( 0.), clm%qflx_evap_soi - egsmax )
  clm%qflx_evap_soi = min ( clm%qflx_evap_soi, egsmax )
  clm%eflx_sh_grnd = clm%eflx_sh_grnd + htvp*egidif
  ! Ground heat flux

  clm%eflx_soil_grnd = clm%sabg + dlrad + (1-clm%frac_veg_nosno)*emg*clm%forc_lwrad &
       - emg*sb*tssbef(clm%snl+1)**3*(tssbef(clm%snl+1) + 4.*tinc) &
       - (clm%eflx_sh_grnd+clm%qflx_evap_soi*htvp)

  clm%eflx_sh_tot = clm%eflx_sh_veg + clm%eflx_sh_grnd
  clm%qflx_evap_tot = clm%qflx_evap_veg + clm%qflx_evap_soi
  clm%eflx_lh_tot= hvap*clm%qflx_evap_veg + htvp*clm%qflx_evap_soi   ! W/m2 (accounting for sublimation)

  clm%qflx_evap_grnd = 0.
  clm%qflx_sub_snow = 0.
  clm%qflx_dew_snow = 0.
  clm%qflx_dew_grnd = 0.

  if (clm%qflx_evap_soi >= 0.) then
     ! Do not allow for sublimation in melting (melting ==> evap. ==> sublimation)
     ! clm%qflx_evap_grnd = min(clm%h2osoi_liq(clm%snl+1)/clm%dtime, clm%qflx_evap_soi)
     !clm%qflx_evap_grnd = clm%qflx_evap_soi    
     !clm%qflx_sub_snow = clm%qflx_evap_soi - clm%qflx_evap_grnd
  if (clm%snl<0) then !LB - sublimation; includes evaporation of meltwater
        clm%qflx_sub_snow = clm%qflx_evap_soi  
	 else !LB - evaporation; passed to PF; includes sublimation if clm%snowdp < 0.01
        clm%qflx_evap_grnd = clm%qflx_evap_soi 
     endif
  else
     if (tg < tfrz) then
        clm%qflx_dew_snow = abs(clm%qflx_evap_soi)
     else
        clm%qflx_dew_grnd = abs(clm%qflx_evap_soi)
     endif
  endif

  ! Outgoing long-wave radiation from canopy + ground

  clm%eflx_lwrad_out = ulrad &
       + (1-clm%frac_veg_nosno)*(1.-emg)*clm%forc_lwrad &
       + (1-clm%frac_veg_nosno)*emg*sb * tssbef(clm%snl+1)**4 &
       ! For conservation we put the increase of ground longwave to outgoing
       + 4.*emg*sb*tssbef(clm%snl+1)**3*tinc

  ! Radiative temperature

  clm%t_rad = (clm%eflx_lwrad_out/sb)**0.25

  !=========================================================================
  ![7] Soil Energy balance check
  !=========================================================================

  clm%errsoi = 0. 
  do j = clm%snl+1, nlevsoi
     clm%errsoi = clm%errsoi - (clm%t_soisno(j)-tssbef(j))/fact(j) 
  enddo
  clm%errsoi = clm%errsoi + clm%eflx_soil_grnd - xmf

  !=========================================================================
  ![8] Variables needed by history tap
  !=========================================================================

  clm%dt_grnd        = tinc
  clm%eflx_lh_vege   = (clm%qflx_evap_veg - clm%qflx_tran_veg) * hvap
  clm%eflx_lh_vegt   = clm%qflx_tran_veg * hvap       
  clm%eflx_lh_grnd   = clm%qflx_evap_soi * htvp
  clm%eflx_lwrad_net = clm%eflx_lwrad_out -  clm%forc_lwrad  

end subroutine clm_thermal
