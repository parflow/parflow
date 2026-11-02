!#include <misc.h>

subroutine clm_stomata(mpe,       apar,     ei,        ea,      &
                       tgcm,      o2,       co2,                &
                       btran,     rb,       rs,        psn,     &
                       qe25,      kc25,     ko25,      vcmx25,  &
                       akc,       ako,      avcmx,     bp,      &
                       mp,        foln,     folnmx,    c3psn,  clm)

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
!  Leaf stomatal resistance and leaf photosynthesis.
!
! REVISION HISTORY:
! date last revised: March 1996 - lsm version 1
! author:            Gordon Bonan
! standardized:      J. Truesdale, Feb. 1996
! reviewed:          G. Bonan, Feb. 1996
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision
!  March 2026: Reed Maxwell; PFT photosynthesis, Rd, smooth colimitation,
!              niter=5, Medlyn stomatal conductance @RMM 2026
!=========================================================================
! $Id: clm_stomata.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : tfrz
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm !CLM 1-D Module

  real(r8), intent(in) :: &
       mpe,               & ! prevents division by zero errors
       ei,                & ! vapor pressure inside leaf (sat vapor press at t_veg) (pa)
       ea,                & ! vapor pressure of canopy air (pa)
       apar,              & ! par absorbed per unit lai (w/m**2)
       o2,                & ! atmospheric o2 concentration (pa)
       co2,               & ! atmospheric co2 concentration (pa)
       tgcm,              & ! air temperature at agcm reference height (kelvin)
       btran,             & ! soil water transpiration factor (0 to 1)
       foln,              & ! foliage nitrogen concentration (%)
       qe25,              & ! quantum efficiency at 25c (umol co2 / umol photon)
       ko25,              & ! o2 michaelis-menten constant at 25c (pa)
       kc25,              & ! co2 michaelis-menten constant at 25c (pa)
       vcmx25,            & ! maximum rate of carboxylation at 25c (umol co2/m**2/s)
       ako,               & ! q10 for ko25
       akc,               & ! q10 for kc25
       avcmx,             & ! q10 for vcmx25
       bp,                & ! minimum leaf conductance (umol/m**2/s)
       mp,                & ! slope for conductance-to-photosynthesis relationship
       folnmx,            & ! foliage nitrogen concentration when f(n)=1 (%)
       c3psn                ! photosynthetic pathway: 0. = c4, 1. = c3

  real(r8), intent(inout) :: rb  ! boundary layer resistance (s/m)
  real(r8), intent(out)   :: rs  ! leaf stomatal resistance (s/m)
  real(r8), intent(out)   :: psn ! foliage photosynthesis (umol co2 /m**2/ s) [always +]

!=== Local Variables =====================================================

  integer, parameter :: niter = 3  ! number of iterations

  integer  iter            ! iteration index

  real(r8) ab,            & ! used in statement functions
       bc,                & ! used in statement functions
       f1,                & ! generic temperature response (statement function)
       f2,                & ! generic temperature inhibition (statement function)
       tc,                & ! foliage temperature (degree celsius)
       cs,                & ! co2 concentration at leaf surface (pa)
       kc,                & ! co2 michaelis-menten constant (pa)
       ko,                & ! o2 michaelis-menten constant (pa)
       a,b,c,q,           & ! intermediate calculations for rs
       r1,r2,             & ! roots for rs
       fnf,               & ! foliage nitrogen adjustment factor (0 to 1)
       ppf,               & ! absorb photosynthetic photon flux (umol photons/m**2/s)
       wc,                & ! rubisco limited photosynthesis (umol co2/m**2/s)
       wj,                & ! light limited photosynthesis (umol co2/m**2/s)
       we,                & ! export limited photosynthesis (umol co2/m**2/s)
       cp,                & ! co2 compensation point (pa)
       ci,                & ! internal co2 (pa)
       ci_old,            & ! previous iteration ci for convergence check
       awc,               & ! intermediate calculation for wc
       vcmx,              & ! maximum rate of carboxylation (umol co2/m**2/s)
       j,                 & ! electron transport (umol co2/m**2/s)
       cea,               & ! constrain ea or else model blows up
       cf,                & ! s m**2/umol -> s/m
       rsmax0,            & ! maximum stomatal resistance [s/m]
       rd,                & ! dark respiration (umol co2/m**2/s) @RMM 2026
       psn_gross,         & ! gross photosynthesis before Rd @RMM 2026
       ai,                & ! intermediate for smooth colimitation @RMM 2026
       theta_cj,          & ! curvature for Wc-Wj colimitation @RMM 2026
       theta_ip,          & ! curvature for (Wc,Wj)-We colimitation @RMM 2026
       vpd_pa,            & ! vapor pressure deficit [Pa] for Medlyn @RMM 2026
       g1_pa                ! Medlyn g1 in Pa^0.5 units @RMM 2026

!=== End Variable List ===================================================

  f1(ab,bc) = ab**((bc-25.)/10.)
  f2(ab) = 1. + exp((-2.2e05+710.*(ab+273.16))/(8.314*(ab+273.16)))

! Initialize rs=rsmax and psn=0 because calculations are performed only
! when apar > 0, in which case rs <= rsmax and psn >= 0
! Set constants

  rsmax0 = 2.e4
  cf = clm%forc_pbot/(8.314*tgcm)*1.e06

  if (apar <= 0.) then          ! night time
     rs = min(rsmax0, 1./bp * cf)
     psn = 0.
     return
  else                          ! day time
     fnf = min(foln/max(mpe,folnmx), 1.0d0)
     tc = clm%t_veg-tfrz
     ppf = 4.6*apar
     j = ppf*qe25
     kc = kc25 * f1(akc,tc)
     ko = ko25 * f1(ako,tc)
     awc = kc * (1.+o2/ko)
     cp = 0.5*kc/ko*o2*0.21
     vcmx = vcmx25 * f1(avcmx,tc) / f2(tc) * fnf * btran

! First guess ci

     ci = 0.7*co2*c3psn + 0.4*co2*(1.-c3psn)

! rb: s/m -> s m**2 / umol

     rb = rb/cf

! Constrain ea

     cea = max(0.25*ei*c3psn+0.40*ei*(1.-c3psn), min(ea,ei) )

! ci iteration — select original or improved physics path @RMM 2026

     if (.not. clm%photosyn_custom) then

        ! === Original CLM3 code path (bit-for-bit identical) ===
        do iter = 1, niter
           wj = max(ci-cp,0.d0)*j/(ci+2.*cp)*c3psn + j*(1.-c3psn)
           wc = max(ci-cp,0.d0)*vcmx/(ci+awc)*c3psn + vcmx*(1.-c3psn)
           we = 0.5*vcmx*c3psn + 4000.*vcmx*ci/clm%forc_pbot*(1.-c3psn)
           psn = min(wj,wc,we)
           cs = max( co2-1.37*rb*clm%forc_pbot*psn, mpe )
           a = mp*psn*clm%forc_pbot*cea / (cs*ei) + bp
           b = ( mp*psn*clm%forc_pbot/cs + bp ) * rb - 1.
           c = -rb
           if (b >= 0.) then
              q = -0.5*( b + sqrt(b*b-4.*a*c) )
           else
              q = -0.5*( b - sqrt(b*b-4.*a*c) )
           endif
           r1 = q/a
           r2 = c/q
           rs = max(r1,r2)
           ci = max( cs-psn*clm%forc_pbot*1.65*rs, 0.d0 )
        enddo

     else

        ! === Improved physics (CLM4.5-era) @RMM 2026 ===
        do iter = 1, 5
           wj = max(ci-cp,0.d0)*j/(ci+2.*cp)*c3psn             &
                + j*(1.-c3psn)
           wc = max(ci-cp,0.d0)*vcmx/(ci+awc)*c3psn             &
                + vcmx*(1.-c3psn)
           we = 0.5*vcmx*c3psn                                   &
                + 4000.*vcmx*ci/clm%forc_pbot*(1.-c3psn)
           theta_cj = 0.98d0*c3psn + 0.80d0*(1.d0 - c3psn)
           theta_ip = 0.95d0
           ai = (wc + wj - sqrt(max((wc+wj)**2                  &
                - 4.d0*theta_cj*wc*wj, 0.d0)))                  &
                / (2.d0*theta_cj)
           psn_gross = (ai + we - sqrt(max((ai+we)**2           &
                       - 4.d0*theta_ip*ai*we, 0.d0)))           &
                       / (2.d0*theta_ip)
           rd = 0.015d0 * vcmx
           psn = max(psn_gross - rd, 0.d0)
           cs = max( co2-1.37*rb*clm%forc_pbot*psn, mpe )
           if (clm%stomata_scheme == 1) then
              vpd_pa = max(ei - cea, 100.d0)
              g1_pa = clm%g1_medlyn * sqrt(1000.d0)
              a = bp + 1.6d0*(1.d0 + g1_pa/sqrt(vpd_pa))        &
                  * psn * clm%forc_pbot / cs
              b = (bp + 1.6d0*(1.d0 + g1_pa/sqrt(vpd_pa))       &
                  * psn * clm%forc_pbot / cs) * rb - 1.d0
              c = -rb
           else
              a = mp*psn*clm%forc_pbot*cea / (cs*ei) + bp
              b = ( mp*psn*clm%forc_pbot/cs + bp ) * rb - 1.
              c = -rb
           endif
           if (b >= 0.) then
              q = -0.5*( b + sqrt(b*b-4.*a*c) )
           else
              q = -0.5*( b - sqrt(b*b-4.*a*c) )
           endif
           r1 = q/a
           r2 = c/q
           rs = max(r1,r2)
           ci_old = ci
           ci = max( cs-psn*clm%forc_pbot*1.65*rs, 0.d0 )
           if (abs(ci - ci_old) < 0.1d0) exit
        enddo

     endif

     ! rs, rb:  s m**2 / umol -> s/m
     rs = min(rsmax0, rs*cf)
     ! multiply stomatal resistance for beetle kill or not @CAP 2014-02-24
     rs = rs*clm%bkmult

     rb = rb*cf

  endif

end subroutine clm_stomata
