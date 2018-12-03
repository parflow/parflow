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

  integer,parameter :: max_nlevsoi     =  20   !number of soil levels, should be set from input
  integer,parameter :: max_nlevlak     =  20   !number of lake levels, should be set from input

  integer:: nlevsoi     =  100   !number of soil levels, should be set from input
  integer:: nlevlak     =  100   !number of lake levels, should be set from input
  integer, parameter :: nlevsno     =  5    !number of maximum snow levels

  integer, parameter :: numrad      =   2   !number of solar radiation bands: vis, nir
  integer, parameter :: numcol      =   8   !number of soil color types

end module clm_varpar
