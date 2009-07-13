!#include <misc.h>

subroutine clm_condcq (raw,        rbw,      rdw,               &   
     rpp,        wtaq,     wtlq,              &
     wtgq,       wtaq0,    wtlq0,    wtgq0,   &
     wtalq,      wtgaq,    wtglq,     clm     )

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
  !  Provides dimensional and non-dimensional latent heat 
  !  conductances for canopy and soil flux calculations.  Latent fluxes 
  !  differs from the sensible heat flux due to stomatal resistance.
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
  !=========================================================================

  ! Declare Modules and data structures

  use precision
  use clmtype
  implicit none

  !=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm  !CLM 1-D Module

  real(r8), intent(in) ::  &
       raw,             & ! aerodynamical resistance [s/m]
       rbw,             & ! leaf boundary layer resistance [s/m]
       rdw,             & ! latent heat resistance between ground and bottom of canopy
       rpp                ! fraction of potential evaporation from leaf [-]

  real(r8), intent(out) :: &
       wtaq,            & ! latent heat conduactance for air [m/s]
       wtlq,            & ! latent heat conduactance for leaf [m/s]
       wtgq,            & ! latent heat conduactance for ground [m/s]
       wtaq0,           & ! normalized latent heat conduactance for air [-]
       wtlq0,           & ! normalized latent heat conduactance for leaf [-]
       wtgq0,           & ! normalized heat conduactance for ground [-]
       wtalq,           & ! normalized latent heat cond. for air and leaf [-]
       wtglq,           & ! normalized latent heat cond. for leaf and ground [-]
       wtgaq              ! normalized latent heat cond. for air and ground [-]

  !=== Local Variables =====================================================

  real(r8)  wtsqi            ! latent heat resistance for air, grd and leaf [-]

  !=== End Variable List ===================================================

  wtaq  = clm%frac_veg_nosno/raw                                ! air
  wtlq  = clm%frac_veg_nosno*(clm%elai+clm%esai)/rbw * rpp      ! leaf
  wtgq  = clm%frac_veg_nosno/rdw                                ! ground
  wtsqi = 1./(wtaq+wtlq+wtgq)

  wtgq0 = wtgq*wtsqi                    ! ground
  wtlq0 = wtlq*wtsqi                    ! leaf
  wtaq0 = wtaq*wtsqi                    ! air

  wtglq = wtgq0+wtlq0                   ! ground + leaf
  wtgaq = wtaq0+wtgq0                   ! air + ground
  wtalq = wtaq0+wtlq0                   ! air + leaf

end subroutine clm_condcq
