!#include <misc.h>

module infnan
!-------------------------------------------------------------------------
! Purpose:
!	Set parameters for the floating point flags "inf" Infinity
!	and "nan" not-a-number. As well as "bigint" the point
!	at which integers start to overflow. These values are used
!	to initialize arrays with as a way to detect if arrays
!	are being used before being set.
!-------------------------------------------------------------------------
  use precision
! Using trick here to make inf and nan values based on octal byte
! representations and EQUIVALNCE.
! The more natural parameter statements caused compilation issues
!  real(r8), parameter :: inf = O'17740000000'
!  real(r8), parameter :: nan = O'17757777777'
  integer :: int_infinity
  integer :: int_nan
  data int_infinity/O'17740000000'/
  data int_nan/O'17757777777'/
  real(r8) inf
  real(r8) nan
  EQUIVALENCE (inf,int_infinity)
  EQUIVALENCE (nan,int_nan)

  integer, parameter  :: bigint = 100000000
end module infnan



  



