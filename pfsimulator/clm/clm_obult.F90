!#include <misc.h>

subroutine clm_obult (displa,  z0m,      z0h,    z0q,    obu,  &
                      um,      ustar,    temp1,  temp2,  clm)

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
!  Calculation of the friction velocity, relation for potential 
!  temperature and humidity profiles of surface boundary layer. 
!  The scheme is based on the work of Zeng et al. (1998): 
!  Intercomparison of bulk aerodynamic algorithms for the computation 
!  of sea surface fluxes using TOGA CORE and TAO data. J. Climate, 
!  Vol. 11, 2628-2644.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_obult.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

! Declare Modules and data structures

  use precision
  use clmtype
  use clm_varcon, only : vkc
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm   !CLM 1-D Module
  real(r8), intent(in) :: &
       displa,       & ! displacement height [m]
       z0m,          & ! roughness length, momentum [m]
       z0h,          & ! roughness length, sensible heat [m]
       z0q,          & ! roughness length, latent heat [m]
       obu,          & ! monin-obukhov length (m)
       um              ! wind speed including the stability effect [m/s]

  real(r8), intent(out) :: &
       ustar,       & ! friction velocity [m/s]
       temp1,       & ! relation for potential temperature profile
       temp2          ! relation for specific humidity profile

!=== Local Variables =====================================================

  real(r8)  zldis,    & ! reference height "minus" zero displacement heght [m]
       clm_psi,       & ! stability function for unstable case
       zetam,         & ! transition point of flux-gradient relation (wind profile)
       zetat,         & ! transition point of flux-gradient relation (temp. profile)
       zeta             ! dimensionless height used in Monin-Obukhov theory

!=== End Variable List ===================================================

! Adjustment factors for unstable (moz < 0) or stable (moz > 0) conditions.
! Wind profile

  zldis=clm%forc_hgt_u-displa
  if (zldis < 0.0d0) zldis = 5.0d0
  zeta=zldis/obu
  zetam=1.574

  if (zeta < -zetam) then           ! zeta < -1
     ustar=vkc*um/(log(-zetam*obu/z0m)- &
          clm_psi(1,-zetam) +clm_psi(1,z0m/obu) &
          +1.14*((-zeta)**0.333-(zetam)**0.333))
  else if (zeta < 0.) then         ! -1 <= zeta < 0
     ustar=vkc*um/(log(zldis/z0m)- &
          clm_psi(1,zeta)+clm_psi(1,z0m/obu))
  else if (zeta <= 1.) then        !  0 <= ztea <= 1
     ustar=vkc*um/(log(zldis/z0m) + &
          5.*zeta -5.*z0m/obu)
  else                             !  1 < zeta, phi=5+zeta
     ustar=vkc*um/(log(obu/z0m)+5.-5.*z0m/obu &
          +(5.*log(zeta)+zeta-1.))
  endif

! Temperature profile

  zldis=clm%forc_hgt_t-displa
  if (zldis < 0.0d0) zldis = 5.0d0
  zeta=zldis/obu
  zetat=0.465
  if (zeta < -zetat) then           ! zeta < -1
     temp1=vkc/(log(-zetat*obu/z0h)-clm_psi(2,-zetat) &
          +clm_psi(2,z0h/obu) &
          +0.8*((zetat)**(-0.333)-(-zeta)**(-0.333)))
  else if (zeta < 0.) then         ! -1 <= zeta < 0
     temp1=vkc/(log(zldis/z0h)-clm_psi(2,zeta)+ &
          clm_psi(2,z0h/obu))
  else if (zeta <= 1.) then        !  0 <= ztea <= 1
     temp1=vkc/(log(zldis/z0h)+5.*zeta-5.*z0h/obu)
  else                             !  1 < zeta, phi=5+zeta
     temp1=vkc/(log(obu/z0h)+5.-5.*z0h/obu &
          +(5.*log(zeta)+zeta-1.))
  endif

! Humidity profile

  zldis=clm%forc_hgt_q-displa
  if (zldis < 0.0d0) zldis = 5.0d0
  zeta=zldis/obu
  zetat=0.465
  if (zeta < -zetat) then          ! zeta < -1
     temp2=vkc/(log(-zetat*obu/z0q)- &
          clm_psi(2,-zetat)+clm_psi(2,z0q/obu) &
          +0.8*((zetat)**(-0.333)-(-zeta)**(-0.333)))
  else if (zeta < 0.) then         ! -1 <= zeta < 0
     temp2=vkc/(log(zldis/z0q)- &
          clm_psi(2,zeta)+clm_psi(2,z0q/obu))
  else if (zeta <= 1.) then        !  0 <= ztea <= 1
     temp2=vkc/(log(zldis/z0q)+5.*zeta-5.*z0q/obu)
  else                             !  1 < zeta, phi=5+zeta
     temp2=vkc/(log(obu/z0q)+5.-5.*z0q/obu &
          +(5.*log(zeta)+zeta-1.))
  endif

end subroutine clm_obult
