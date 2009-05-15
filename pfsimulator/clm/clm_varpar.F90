!#include <misc.h>

module clm_varpar

!----------------------------------------------------------------------- 
! 
! Purpose: 
! land surface model parameters
! 
! Method: 
! 
! Author: Mariana Vertenstein
! 
!-----------------------------------------------------------------------

  use precision
  implicit none

! define level parameters

  integer, parameter :: nlevsoi     =  10   !number of soil levels
  integer, parameter :: nlevlak     =  10   !number of lake levels
  integer, parameter :: nlevsno     =  5    !number of maximum snow levels

  integer, parameter :: numrad      =   2   !number of solar radiation bands: vis, nir
  integer, parameter :: numcol      =   8   !number of soil color types

end module clm_varpar
