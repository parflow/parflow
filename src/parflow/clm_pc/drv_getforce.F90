!#include <misc.h>

subroutine drv_getforce (drv,tile,clm)

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
!  Access meteorological data - this current version reads 1D forcing
!  and distributes it to the clm domain (spatially constant).  This routine
!  must be modified to allow for spatially variable forcing, or coupling to
!  a GCM.
!
!  The user may likely want to modify this subroutine significantly,
!  to include such things as space/time intrpolation of forcing to the
!  CLM grid, reading of spatially variable binary data, etc.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: drv_getforce.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use clm_varcon, only : tfrz, tcrit
  implicit none

!=== Arguments ===========================================================

  type (drvdec) ,intent(inout) :: drv              
  type (tiledec),intent(inout) :: tile(drv%nch)
  type (clm1d)  ,intent(inout) :: clm (drv%nch)

!=== Local Variables =====================================================

  real(r8) solar     ! incident solar radiation [w/m2]
  real(r8) prcp      ! precipitation [mm/s]
  integer t          ! Tile looping variable

!=== End Variable List ===================================================

!=== Increment Time Step Counter

  clm%istep=clm%istep+1 

! Valdai - 1D Met data

  read (11,*) solar, clm(1)%forc_lwrad, prcp, clm(1)%forc_t, clm(1)%forc_u, &
              clm(1)%forc_v, clm(1)%forc_pbot, clm(1)%forc_q

  clm(1)%forc_rho   = clm(1)%forc_pbot/(clm(1)%forc_t*2.8704e2)

  clm(1)%forc_solad(1) = solar*35./100.    !forc_sols
  clm(1)%forc_solad(2) = solar*35./100.    !forc_soll
  clm(1)%forc_solai(1) = solar*15./100.    !forc_solsd
  clm(1)%forc_solai(2) = solar*15./100.    !forc_solad

! Next set upper limit of air temperature for snowfall at 275.65K.
! This cut-off was selected based on Fig. 1, Plate 3-1, of Snow
! Hydrology (1956).

  if (prcp > 0.) then
     if(clm(1)%forc_t > (tfrz + tcrit))then
        clm(1)%itypprc = 1
        clm(1)%forc_rain = prcp
        clm(1)%forc_snow = 0.
     else
        clm(1)%itypprc = 2
        clm(1)%forc_rain = 0.
        clm(1)%forc_snow = prcp
     endif
  else
     clm(1)%itypprc = 0
     clm(1)%forc_rain = 0.
     clm(1)%forc_snow = 0
  endif

!=== Extend forcing to entire CLM tile space uniformly
!=== This should be modified for spatially variable forcing

  do t=2,drv%nch
     clm(t)%itypprc       = clm(1)%itypprc
     clm(t)%forc_rain     = clm(1)%forc_rain
     clm(t)%forc_snow     = clm(1)%forc_snow
     clm(t)%forc_lwrad    = clm(1)%forc_lwrad
     clm(t)%forc_t        = clm(1)%forc_t
     clm(t)%forc_u        = clm(1)%forc_u
     clm(t)%forc_v        = clm(1)%forc_v
     clm(t)%forc_pbot     = clm(1)%forc_pbot
     clm(t)%forc_q        = clm(1)%forc_q
     clm(t)%forc_rho      = clm(1)%forc_rho
     clm(t)%forc_solad(1) = clm(1)%forc_solad(1) 
     clm(t)%forc_solad(2) = clm(1)%forc_solad(2) 
     clm(t)%forc_solai(1) = clm(1)%forc_solai(1) 
     clm(t)%forc_solai(2) = clm(1)%forc_solai(2) 
  enddo

end subroutine drv_getforce
