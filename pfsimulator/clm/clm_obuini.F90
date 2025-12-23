!#include <misc.h>

subroutine clm_obuini(ur,      thv,     dthv,      zldis,       &
                      z0m,     um,      obu        )

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
!  Initialization of the Monin-Obukhov length.
!  The scheme is based on the work of Zeng et al. (1998): 
!  Intercomparison of bulk aerodynamic algorithms for the computation 
!  of sea surface fluxes using TOGA CORE and TAO data. J. Climate, 
!  Vol. 11, 2628-2644.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================

! Declare Modules and data structures

  use precision
  use clm_varcon, only : grav
  implicit none

!=== Arguments ===========================================================      

  real(r8), intent(in) :: &
       ur,         &! wind speed at reference height [m/s]
       thv,        &! virtual potential temperature (kelvin)
       dthv,       &! diff of vir. poten. temp. between ref. height and surface
       zldis,      &! reference height "minus" zero displacement heght [m]
       z0m          ! roughness length, momentum [m]

  real(r8), intent(out) :: &
       um,         &! wind speed including the stability effect [m/s]
       obu          ! monin-obukhov length (m)

!=== Local Variables =====================================================

  real(r8)  wc,         &! convective velocity [m/s]
       rib,        &! bulk Richardson number
       zeta,       &! dimensionless height used in Monin-Obukhov theory
       ustar        ! friction velocity [m/s]     

!=== End Variable List ===================================================

! Initial values of u* and convective velocity

  ustar=0.06
  wc=0.5
  if (dthv >= 0.) then
     um=max(ur,dble(0.1))
  else
     um=sqrt(ur*ur+wc*wc)
  endif

  rib=grav*zldis*dthv/(thv*um*um)

  if (rib >= 0.) then      ! neutral or stable
     zeta = rib*log(zldis/z0m)/(1.-5.*min(rib,dble(0.19)))
     zeta = min(dble(2.),max(zeta,dble(0.01) ))
  else                    !unstable
     zeta=rib*log(zldis/z0m)
     zeta = max(dble(-100.),min(zeta,dble(-0.01) ))
  endif

  obu=zldis/zeta

end subroutine clm_obuini
