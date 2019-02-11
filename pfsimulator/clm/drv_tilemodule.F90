!#include <misc.h>

module drv_tilemodule 
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
!  Module for tile space variable specification.
!
! REVISION HISTORY:
!  15 Jan 2000: Paul Houser; Initial code
!=========================================================================
! $Id: drv_tilemodule.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clm_varpar, only : max_nlevsoi
  implicit none

  public tiledec
  type tiledec

!=== TILE SPACE User-Defined Parameters ====================================

     integer  :: col           ! Grid Column of Tile
     integer  :: row           ! Grid Row of Tile
     integer  :: vegt          ! Vegetation Type of Tile
     integer  :: pveg          ! Predominance of vegetation clas in grid
     real(r8) :: fgrd          ! Fraction of grid covered by a given veg type (%/100)

     real(r8) :: sand(max_nlevsoi) ! Percent sand in soil (vertically average)
     real(r8) :: clay(max_nlevsoi) ! Percent clay in soil (vertically average)

     real(r8) :: scalez        ! Soil layer thickness discretization (m)
     real(r8) :: hkdepth       ! length scale for Ksat decrease(m)
     real(r8) :: roota         ! Temporary CLM vegetation parameter
     real(r8) :: rootb         ! Temporary CLM vegetation parameter

!=== End Variable List ===================================================

  end type tiledec

end module drv_tilemodule
