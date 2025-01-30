!#include <misc.h>

subroutine clm_lake (clm) 

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
  !  Calculates lake temperatures from a one-dimensional thermal
  !  stratification model based on eddy diffusion concepts to 
  !  represent vertical mixing of heat.
  !
  !  d ts    d            d ts     1 ds
  !  ---- = -- [(km + ke) ----] + -- --
  !   dt    dz             dz     cw dz   
  !
  !  where: ts = temperature (kelvin)
  !          t = time (s)
  !          z = depth (m)
  !         km = molecular diffusion coefficient (m**2/s)
  !         ke = eddy diffusion coefficient (m**2/s)
  !         cw = heat capacity (j/m**3/kelvin)
  !          s = heat source term (w/m**2)
  !
  ! There are two types of lakes: 
  !    Deep lakes are 50 m. 
  !    Shallow lakes are 10 m deep.
  !
  !    For unfrozen deep lakes:    ke > 0 and    convective mixing
  !    For unfrozen shallow lakes: ke = 0 and no convective mixing
  !
  ! Use the Crank-Nicholson method to set up tridiagonal system of equations to
  ! solve for ts at time n+1, where the temperature equation for layer i is
  ! r_i = a_i [ts_i-1] n+1 + b_i [ts_i] n+1 + c_i [ts_i+1] n+1
  !
  ! The solution conserves energy as:
  !
  ! cw*([ts(      1)] n+1 - [ts(      1)] n)*dz(      1)/dt + ... +
  ! cw*([ts(nlevlak)] n+1 - [ts(nlevlak)] n)*dz(nlevlak)/dt = fin
  !
  ! where:
  ! [ts] n   = old temperature (kelvin)
  ! [ts] n+1 = new temperature (kelvin)
  ! fin      = heat flux into lake (w/m**2)
  !          = beta*sabg + forc_lwrad - eflx_lwrad_out - eflx_sh_tot - eflx_lh_tot - hm + phi(1) + ... + phi(nlevlak) 
  !
  ! AUTHOR:
  !  Gordon Bonan
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
  !=========================================================================
  ! $Id: clm_lake.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  ! Declare Modules and data structures

  use precision
  use clmtype
  use clm_varpar, only : nlevlak, nlevsno, nlevsoi
  use clm_varcon, only : hvap, hfus, rair, cpair, cpliq, tkwat, tkice, &
       sb, vkc, grav, denh2o, tfrz
  implicit none

  !=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm      ! CLM 1-D Module

  !=== Local Variables =====================================================

  integer :: idlak = 1     ! index of lake, 1 = deep lake, 2 = shallow lake

  integer &
       niters,            & ! maximum number of iterations for surface temperature
       iter,              & ! iteration index
       nmozsgn              ! number of times moz changes sign

  real(r8)  ax,               & !
       bx,                & !
       beta1,             & ! coefficient of conective velocity [-]
       degdT,             & ! d(eg)/dT
       dqh,               & ! diff of humidity between ref. height and surface
       dth,               & ! diff of virtual temp. between ref. height and surface
       dthv,              & ! diff of vir. poten. temp. between ref. height and surface
       dzsur,             & !
       eg,                & ! water vapor pressure at temperature T [pa]
       emg,               & ! ground emissivity (0.97 for snow,
       hm,                & ! energy residual [W/m2]
       htvp,              & ! latent heat of vapor of water (or sublimation) [j/kg]
       obu,               & ! monin-obukhov length (m)
       obuold,            & ! monin-obukhov length of previous iteration
       qsatg,             & ! saturated humidity [kg/kg]
       qsatgdT,           & ! d(qsatg)/dT
       qstar,             & ! moisture scaling parameter
       qflx_evap_grnd,    & ! ground surface evaporation rate (mm h2o/s)
       qflx_dew_grnd,     & ! ground surface dew formation (mm h2o /s) [+]
       qflx_sub_snow,     & ! sublimation rate from snow pack (mm h2o /s) [+]
       qflx_dew_snow,     & ! surface dew added to snow pack (mm h2o /s) [+]
       qmelt,             & ! snow melt [mm/s]
       ram,               & ! aerodynamical resistance [s/m]
       rah,               & ! thermal resistance [s/m]
       raw,               & ! moisture resistance [s/m]
       stftg3,            & !
       temp1,             & ! relation for potential temperature profile
       temp2,             & ! relation for specific humidity profile
       tgbef,             & !
       th,                & ! potential temperature (kelvin)
       thm,               & ! intermediate variable (forc_t+0.0098*forc_hgt_t)
       thv,               & ! virtual potential temperature (kelvin)
       thvstar,           & ! virtual potential temperature scaling parameter
       tksur,             & ! thermal conductivity of snow/soil (w/m/kelvin)
       tstar,             & ! temperature scaling parameter
       um,                & ! wind speed including the stability effect [m/s]
       ur,                & ! wind speed at reference height [m/s]
       ustar,             & ! friction velocity [m/s]
       wc,                & ! convective velocity [m/s]
       zeta,              & ! dimensionless height used in Monin-Obukhov theory
       zii,               & ! convective boundary height [m]
       zldis,             & ! reference height "minus" zero displacement heght [m]
       z0mg,              & ! roughness length over ground, momentum [m]
       z0hg,              & ! roughness length over ground, sensible heat [m]
       z0qg                 ! roughness length over ground, latent heat [m]

  real(r8) beta(2),       & ! fraction solar rad absorbed at surface: depends on lake type
       za(2),             & ! base of surface absorption layer (m): depends on lake type
       eta(2),            & ! light extinction coefficient (/m): depends on lake type
       p0                   ! neutral value of turbulent prandtl number

  real(r8)  a(nlevlak),       & ! "a" vector for tridiagonal matrix
       b(nlevlak),            & ! "b" vector for tridiagonal matrix
       c(nlevlak),            & ! "c" vector for tridiagonal matrix
       r(nlevlak),            & ! "r" vector for tridiagonal solution
       rhow(nlevlak),         & ! density of water (kg/m**3)
       phi(nlevlak),          & ! solar radiation absorbed by layer (w/m**2)
       kme(nlevlak),          & ! molecular + eddy diffusion coefficient (m**2/s)
       
       cwat,              & ! specific heat capacity of water (j/m**3/kelvin)
       ws,                & ! surface friction velocity (m/s)
       ks,                & ! coefficient
       in,                & ! relative flux of solar radiation into layer
       out,               & ! relative flux of solar radiation out of layer
       ri,                & ! richardson number
       fin,               & ! heat flux into lake - flux out of lake (w/m**2)
       ocvts,             & ! (cwat*(t_soisno[n  ])*dz
       ncvts,             & ! (cwat*(t_soisno[n+1])*dz
       
       m1,                & ! intermediate variable for calculating r, a, b, c
       m2,                & ! intermediate variable for calculating r, a, b, c
       m3,                & ! intermediate variable for calculating r, a, b, c
       ke,                & ! eddy diffusion coefficient (m**2/s)
       km,                & ! molecular diffusion coefficient (m**2/s)
       zin,               & ! depth at top of layer (m)
       zout,              & ! depth at bottom of layer (m)
       drhodz,            & ! d [rhow] /dz (kg/m**4)
       n2,                & ! brunt-vaisala frequency (/s**2)
       num,               & ! used in calculating ri
       den,               & ! used in calculating ri
       tav,               & ! used in aver temp for convectively mixed layers
       nav,               & ! used in aver temp for convectively mixed layers
       phidum,            & ! temporary value of phi
       u2m                  ! 2 m wind speed (m/s)

  real(r8) cf               ! s m**2/umol -> s/m

  integer i,j              ! do loop or array index

  !=== End Variable List ===================================================

  ! ========================================================================
  ! determine beginning water balance for lake points
  ! ========================================================================

  clm%begwb = clm%h2osno  

  ! ========================================================================
  !*[1] constants and model parameters
  ! ========================================================================

  ! Constants for lake temperature model

  beta = (/0.4, 0.4/)                              ! (deep lake, shallow lake)
  za   = (/0.6, 0.5/)    
  eta  = (/0.1, 0.5/)  
  p0   = 1.  

  ! Aerodynamical roughness 

  if (clm%t_grnd >= tfrz)then                    ! for unfrozen lake
     z0mg = 0.01
  else                                           ! for frozen lake
     z0mg = 0.04
  endif
  z0hg = z0mg
  z0qg = z0mg

  ! Latent heat 

  if (clm%forc_t > tfrz) then
     htvp = hvap
  else
     htvp = hfus
  endif

  ! surface emissivity

  emg = 0.97

  ! ========================================================================
  !*[2] SURFACE TEMPERATURE and FLUXES
  ! ========================================================================

  dzsur = clm%dz(1) + clm%snowdp

  call clm_qsadv(clm%t_grnd,clm%forc_pbot,eg,degdT,qsatg,qsatgdT)

  ! Potential temperature at the reference height

  beta1=1.       ! -  (in computing W_*)
  zii = 1000.    ! m  (pbl height)

  thm = clm%forc_t + 0.0098*clm%forc_hgt_t  ! intermed. variable equiv to forc_t*(pgcm/forc_pbot)**(rair/cp)
  th = clm%forc_t*(100000./clm%forc_pbot)**(rair/cpair) ! potential T
  thv = th*(1.+0.61*clm%forc_q)             ! virtual potential T
  ! Original line
  !ur = max(1.0,sqrt(clm%forc_u*clm%forc_u+clm%forc_v*clm%forc_v))    ! limit must set to 1  , otherwise,
  ur = max(dble(1.0),sqrt(clm%forc_u*clm%forc_u+clm%forc_v*clm%forc_v))  ! NBE

  ! Initialization variables

  nmozsgn = 0
  obuold = 0.

  dth   = thm-clm%t_grnd
  dqh   = clm%forc_q-qsatg
  dthv  = dth*(1.+0.61*clm%forc_q)+0.61*th*dqh
  zldis = clm%forc_hgt_u-0.

  call clm_obuini(ur, thv, dthv, zldis, z0mg, um, obu)

  niters = 3

  do iter = 1, niters         ! begin stability iteration
     tgbef = clm%t_grnd
     if (clm%t_grnd > tfrz) then
        tksur = tkwat
     else
        tksur = tkice
     endif

     ! Evaluated stability-dependent variables using moz from prior iteration

     call clm_obult (0.0d0, z0mg, z0hg, z0qg, obu, um, ustar, temp1, temp2, clm)
     obuold = obu

     ! Get derivative of fluxes with respect to ground temperature

     ram    = 1./(ustar*ustar/um)
     rah    = 1./(temp1*ustar)
     raw    = 1./(temp2*ustar)

     stftg3 = emg*sb*tgbef*tgbef*tgbef

     ax  = clm%sabg + emg*clm%forc_lwrad + 3.*stftg3*tgbef &
          + clm%forc_rho*cpair/rah*thm &
          - htvp*clm%forc_rho/raw*(qsatg-qsatgdT*tgbef - clm%forc_q) &
          + tksur*clm%t_soisno(1)/dzsur

     bx  = 4.*stftg3 + clm%forc_rho*cpair/rah &
          + htvp*clm%forc_rho/raw*qsatgdT + tksur/dzsur

     clm%t_grnd = ax/bx

     ! Surface fluxes of momentum, sensible and latent heat
     ! using ground temperatures from previous time step

     clm%eflx_sh_grnd = clm%forc_rho*cpair*(clm%t_grnd-thm)/rah
     clm%qflx_evap_soi = clm%forc_rho*(qsatg+qsatgdT*(clm%t_grnd-tgbef)-clm%forc_q)/raw

     call clm_qsadv(clm%t_grnd,clm%forc_pbot,eg,degdT,qsatg,qsatgdT)
     dth=thm-clm%t_grnd
     dqh=clm%forc_q-qsatg

     tstar = temp1*dth
     qstar = temp2*dqh

     dthv=dth*(1.+0.61*clm%forc_q)+0.61*th*dqh
     thvstar=tstar*(1.+0.61*clm%forc_q) + 0.61*th*qstar
     zeta=zldis*vkc * grav*thvstar/(ustar**2*thv)

     if (zeta >= 0.) then     !stable
        ! zeta = min(2.,max(zeta,0.01))
        zeta = min(dble(2.0d+0),max(zeta,dble(0.01d+0)))  ! NBE

     else                     !unstable
        ! zeta = max(-100.,min(zeta,-0.01))
        zeta = max(dble(-100.0d+0),min(zeta,dble(-0.01d+0))) ! NBE
     endif
     obu = zldis/zeta

     if (dthv >= 0.) then
        ! um = max(ur,0.1)
        um = max(ur,dble(0.1d+0)) ! NBE
     else
        wc = beta1*(-grav*ustar*thvstar*zii/thv)**0.333
        um = sqrt(ur*ur+wc*wc)
     endif

     if (obuold*obu < 0.) nmozsgn = nmozsgn+1
     if (nmozsgn >= 4) EXIT

  enddo

  ! If there is snow on the ground and t_grnd > tfrz: reset t_grnd = tfrz.  Reevaluate ground fluxes.
  ! Energy imbalance used to melt snow.  h2osno > 0.5 prevents spurious fluxes

  if (clm%h2osno > 0.5 .AND. clm%t_grnd > tfrz) then
     clm%t_grnd = tfrz
     clm%eflx_sh_grnd = clm%forc_rho*cpair*(clm%t_grnd-thm)/rah
     clm%qflx_evap_soi = clm%forc_rho*(qsatg+qsatgdT*(clm%t_grnd-tgbef) & 
          - clm%forc_q)/raw    !note that qsatg and qsatgdT should be f(tgbef)
  endif

  ! Net longwave from ground to atmosphere

  clm%eflx_lwrad_out = (1.-emg)*clm%forc_lwrad + stftg3*(-3.*tgbef+4.*clm%t_grnd)

  ! Radiative temperature

  clm%t_rad = (clm%eflx_lwrad_out/sb)**0.25

  ! Ground heat flux

  clm%eflx_soil_grnd = clm%sabg + clm%forc_lwrad - clm%eflx_lwrad_out - &
       clm%eflx_sh_grnd - htvp*clm%qflx_evap_soi

  clm%taux   = -clm%forc_rho*clm%forc_u/ram
  clm%tauy   = -clm%forc_rho*clm%forc_v/ram

  clm%eflx_sh_tot   = clm%eflx_sh_grnd
  clm%qflx_evap_tot = clm%qflx_evap_soi
  clm%eflx_lh_tot   = htvp*clm%qflx_evap_soi
  clm%eflx_lh_grnd  = htvp*clm%qflx_evap_soi

  ! 2 m height air temperature

  clm%t_ref2m   = (clm%t_grnd + temp1*dth * 1./ &
       vkc *log((2.+z0hg)/z0hg))

  ! Energy residual for snow melting

  if (clm%h2osno > 0. .AND. clm%t_grnd >= tfrz) then
     ! hm = min( clm%h2osno*hfus/clm%dtime, max(clm%eflx_soil_grnd,0.) )
     hm = min( clm%h2osno*hfus/clm%dtime, max(clm%eflx_soil_grnd,dble(0.0d+0)) ) ! NBE
  else
     hm = 0.
  endif
  qmelt = hm/hfus             ! snow melt (mm/s)

  ! ========================================================================
  !*[3] LAKE LAYER TEMPERATURE
  ! ========================================================================

  ! Lake density

  do j = 1, nlevlak
     rhow(j) = 1000.*( 1.0 - 1.9549e-05*(abs(clm%t_soisno(j)-277.))**1.68 )
  enddo

  ! Eddy diffusion +  molecular diffusion coefficient:
  ! eddy diffusion coefficient used for unfrozen deep lakes only

  cwat = cpliq*denh2o
  km = tkwat/cwat

  fin = beta(idlak)*clm%sabg + clm%forc_lwrad - (clm%eflx_lwrad_out+clm%eflx_sh_tot+clm%eflx_lh_tot+hm)
  ! u2m = max(1.0,ustar/vkc*log(2./z0mg))
  u2m = max(dble(1.0d+0),ustar/vkc*log(2./z0mg)) ! NBE

  ws = 1.2e-03 * u2m
  ks = 6.6*sqrt(abs(sin(clm%lat)))*(u2m**(-1.84))

  do j = 1, nlevlak-1
     drhodz = (rhow(j+1)-rhow(j)) / (clm%z(j+1)-clm%z(j))
     n2 = -grav / rhow(j) * drhodz
     num = 40. * n2 * (vkc*clm%z(j))**2
     ! den = max( (ws**2) * exp(-2.*ks*clm%z(j)), 1.e-10 )
     den = max( (ws**2) * exp(-2.*ks*clm%z(j)), dble(1.e-10) ) ! NBE
     !ri = ( -1. + sqrt( max(1.+num/den, 0.) ) ) / 20.
     ri = ( dble(-1.0d+0) + sqrt( max(dble(1.0d+0)+num/den, dble(0.0d+0)) ) ) / dble(20.0d+0) ! NBE
     if (idlak == 1 .AND. clm%t_grnd > tfrz) then
        ke = vkc*ws*clm%z(j)/p0 * exp(-ks*clm%z(j)) / (1.+37.*ri*ri)
     else
        ke = 0.
     endif
     kme(j) = km + ke 
  enddo

  kme(nlevlak) = kme(nlevlak-1)

  ! Heat source term: unfrozen lakes only

  do j = 1, nlevlak
     zin  = clm%z(j) - 0.5*clm%dz(j)
     zout = clm%z(j) + 0.5*clm%dz(j)
     !in  = exp( -eta(idlak)*max(  zin-za(idlak),0. ) )
     in  = exp( -eta(idlak)*max(  zin-za(idlak),dble(0.0d+0) ) ) ! NBE
     !out = exp( -eta(idlak)*max( zout-za(idlak),0. ) )
     out = exp( -eta(idlak)*max( zout-za(idlak),dble(0.0d+0) ) ) ! NBE

     ! Assumed solar absorption is only in the considered depth
     if (j == nlevlak) out = 0.  
     if (clm%t_grnd > tfrz) then
        phidum = (in-out) * clm%sabg * (1.-beta(idlak))
     else if (j == 1) then
        phidum= clm%sabg * (1.-beta(idlak))
     else
        phidum = 0.
     endif
     phi(j) = phidum
  enddo

  ! Sum cwat*t_soisno*dz for energy check

  ocvts = 0.
  do j = 1, nlevlak
     ocvts = ocvts + cwat*clm%t_soisno(j)*clm%dz(j) 
  enddo

  ! Set up vector r and vectors a, b, c that define tridiagonal matrix

  j = 1
  m2 = clm%dz(j)/kme(j) + clm%dz(j+1)/kme(j+1)
  m3 = clm%dtime/clm%dz(j)
  r(j) = clm%t_soisno(j) + (fin+phi(j))*m3/cwat - (clm%t_soisno(j)-clm%t_soisno(j+1))*m3/m2
  a(j) = 0.
  b(j) = 1. + m3/m2
  c(j) = -m3/m2

  j = nlevlak
  m1 = clm%dz(j-1)/kme(j-1) + clm%dz(j)/kme(j)
  m3 = clm%dtime/clm%dz(j)
  r(j) = clm%t_soisno(j) + phi(j)*m3/cwat + (clm%t_soisno(j-1)-clm%t_soisno(j))*m3/m1
  a(j) = -m3/m1
  b(j) = 1. + m3/m1
  c(j) = 0.

  do j = 2, nlevlak-1
     m1 = clm%dz(j-1)/kme(j-1) + clm%dz(j  )/kme(j  )
     m2 = clm%dz(j  )/kme(j  ) + clm%dz(j+1)/kme(j+1)
     m3 = clm%dtime/clm%dz(j)
     r(j) = clm%t_soisno(j) + phi(j)*m3/cwat &
          +(clm%t_soisno(j-1)-clm%t_soisno(j))*m3/m1 - (clm%t_soisno(j)-clm%t_soisno(j+1))*m3/m2

     a(j) = -m3/m1
     b(j) = 1. + m3/m1 + m3/m2
     c(j) = -m3/m2
  enddo

  ! Solve for t_soisno: a, b, c, r, u 

  call clm_tridia (nlevlak ,a ,b ,c ,r ,clm%t_soisno(1:nlevlak)) 

  ! Convective mixing: make sure cwat*dz*ts is conserved.  Mixing

  if (idlak == 1 .AND. clm%t_grnd > tfrz) then
     do j = 1, nlevlak-1
        if (rhow(j) > rhow(j+1)) then
           tav = 0.
           nav = 0.
           do i = 1, j+1
              tav = tav + clm%t_soisno(i)*clm%dz(i)
              nav = nav + clm%dz(i)
           enddo
           tav = tav/nav
           do i = 1, j+1
              clm%t_soisno(i) = tav
              rhow(i) = 1000.*( 1.0 - 1.9549e-05*(abs(clm%t_soisno(i)-277.))**1.68 )
           enddo
        endif
     enddo
  endif

  ! Sum cwat*t_soisno*dz and total energy into lake for energy check

  ncvts = 0.
  do j = 1, nlevlak
     ncvts = ncvts + cwat*clm%t_soisno(j)*clm%dz(j) 
     fin = fin + phi(j)
  enddo

  clm%errsoi = (ncvts-ocvts) / clm%dtime - fin

  ! ========================================================================
  ! [4] snow on the lake ice 
  ! ========================================================================

  qflx_evap_grnd = 0.
  qflx_sub_snow = 0.
  qflx_dew_snow = 0.
  qflx_dew_grnd = 0.

  if (clm%qflx_evap_soi >= 0.) then

     ! Sublimation: do not allow for more sublimation than there is snow
     ! after melt.  Remaining surface evaporation used for infiltration.

     qflx_sub_snow = min( clm%qflx_evap_soi, clm%h2osno/clm%dtime-qmelt )
     qflx_evap_grnd = clm%qflx_evap_soi - qflx_sub_snow

  else

     if (clm%t_grnd < tfrz-0.1) then
        qflx_dew_snow = abs(clm%qflx_evap_soi)
     else
        qflx_dew_grnd = abs(clm%qflx_evap_soi)
     endif

  endif

  ! Update snow pack

  clm%h2osno = clm%h2osno + (clm%forc_snow-qmelt-qflx_sub_snow+qflx_dew_snow)*clm%dtime
  !clm%h2osno = max( clm%h2osno, 0. )
  clm%h2osno = max( clm%h2osno, dble(0.0d+0) )

  ! No snow if lake unfrozen

  if (clm%t_grnd > tfrz) clm%h2osno = 0.

  ! Snow height and fractional coverage

  clm%snowdp = clm%h2osno/250.       !assumed a constant snow bulk density = 250.

  ! ========================================================================
  ! determine ending water balance for lake points
  ! ========================================================================

  clm%endwb = clm%h2osno

  ! ========================================================================
  ! [5] set other clm values for lake points
  ! ========================================================================

  ! the following are needed for global average on history tape 
  ! note time invariant variables set in initialization phase:
  !    z, dz, snl, h2osoi_liq, and h2osoi_ice 

  clm%t_veg = clm%forc_t  ! to be consistent with treatment of t_veg for bare soil points

  clm%eflx_sh_veg     = 0.    
  clm%eflx_lh_vegt    = 0.   
  clm%eflx_lh_vege    = 0.    
  clm%eflx_lwrad_net  = clm%eflx_lwrad_out -  clm%forc_lwrad
  clm%eflx_snomelt    = qmelt*hfus

  clm%h2ocan          = 0.  
  clm%qflx_evap_veg   = 0.
  clm%qflx_tran_veg   = 0.
  clm%qflx_infl       = 0. 
  clm%qflx_snomelt    = qmelt
  clm%qflx_surf       = 0.
  clm%qflx_drain      = 0.
  clm%qflx_qirr       = 0.
  clm%qflx_qrgwl      = clm%forc_rain + clm%forc_snow - clm%qflx_evap_tot - (clm%endwb-clm%begwb)/clm%dtime

  clm%qflx_prec_grnd  = clm%forc_rain + clm%forc_snow
  clm%qflx_prec_intr  = 0.

  clm%btran           = 0.     

  ! put in for consistency with LSM

  cf = clm%forc_pbot/(8.314*thm)*1.e06 
  clm%rssun = 1./clm%bp * cf
  clm%rssha = 1./clm%bp * cf

end subroutine clm_lake


