!#include <misc.h>

subroutine drv_t2gi(t,       g,       nc,     nr,               &
                    nch,     fgrd,    col,    row)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! drv_t2g.f and drv_g2t.f: 
!
! DESCRIPTION:
!  Transfer variables between grid and tile space.
!
! REVISION HISTORY:
!  15 Oct 1999: Paul Houser; Initial Code
!=========================================================================
! $Id: drv_t2g.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  implicit none

  integer nch,nc,nr,c,r,i,col(nch),row(nch)
  integer t(nch),g(nc,nr)
  real(r8) fgrd(nch)

  do c=1,nc
     do r=1,nr
        g(c,r)=0
     enddo
  enddo

  do i=1,nch
     g(col(i),row(i))=g(col(i),row(i))+t(i)*fgrd(i)
  enddo
  return
end subroutine drv_t2gi

subroutine drv_t2gr(t,       g,       nc,     nr,               &
                    nch,     fgrd,    col,    row)
  use precision
  implicit none
  integer nch,nc,nr,c,r,i,col(nch),row(nch)
  real(r8) t(nch),g(nc,nr)
  real(r8) fgrd(nch)

  do c=1,nc
     do r=1,nr
        g(c,r)=0.0
     enddo
  enddo

  do i=1,nch
     g(col(i),row(i))=g(col(i),row(i))+t(i)*fgrd(i)
  enddo
  return
end subroutine drv_t2gr


subroutine drv_g2ti(t,       g,       nc,     nr,               &
                    nch,     col,    row)
  use precision
  implicit none
  integer nch,nc,nr,i,col(nch),row(nch)
  integer t(nch),g(nc,nr)

  do i=1,nch
     t(i)=g(col(i),row(i))
  enddo
  return
end subroutine drv_g2ti

subroutine drv_g2tr(t,       g,       nc,     nr,               &
                    nch,     col,    row)
  use precision
  implicit none
  integer nch,nc,nr,i,col(nch),row(nch)
  real(r8) t(nch),g(nc,nr)

  do i=1,nch
     t(i)=g(col(i),row(i))
  enddo
  return
end subroutine drv_g2tr




