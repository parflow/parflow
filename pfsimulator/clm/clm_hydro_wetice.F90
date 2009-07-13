!#include <misc.h>

subroutine clm_hydro_wetice (clm)

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
  !  Calculate hydrology for ice and wetland
  !
  ! REVISION HISTORY:
  !  7 November 2000: Mariana Vertenstein; Initial code
  !
  !=========================================================================
  ! $Id: clm_hydro_wetice.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : istwet, istice
  implicit none

  !=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm !CLM 1-D Module

  !=========================================================================

  ! Wetland and land ice runoff

  clm%qflx_drain  = 0. 
  clm%qflx_surf   = 0.
  clm%qflx_infl   = 0.
  clm%qflx_qirr   = 0.
  clm%qflx_qrgwl  = clm%forc_rain + clm%forc_snow - clm%qflx_evap_tot - (clm%endwb - clm%begwb)/clm%dtime

end subroutine clm_hydro_wetice
