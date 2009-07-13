!#include <misc.h>

subroutine drv_g2clm(u,drv,grid,tile,clm)

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
!  Transfer variables into CLM tile space from grid space. 
!
! REVISION HISTORY:
!  15 Jan 2000: Paul Houser; Initial code
!=========================================================================
! $Id: drv_g2clm.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use drv_gridmodule      ! Grid-space variables
  use clmtype             ! CLM tile variables
  use clm_varcon, only : istdlak, istslak
  implicit none

!=== Arguments ===========================================================

  type (drvdec)  :: drv              
  type (tiledec) :: tile
  type (griddec) :: grid(drv%nc,drv%nr)   
  type (clm1d)   :: clm

!=== Local Variables =====================================================

  integer  :: r,c       ! Loop counters
  real(r8) :: u         ! Tempoary UNDEF Variable  

!=== End Variable Definition =============================================

!=== Transfer variables that are identical for each tile in a grid 
!===  to tile space  - for some aplications, the assumption
!===  of spatially constant information across a grid for these
!===  variables may be incorrect, and should be modified by the user.

  c=tile%col
  r=tile%row

! CLM Forcing parameters (read into 2-D grid module variables)

  if (grid(c,r)%forc_hgt_u      /= u) clm%forc_hgt_u =grid(c,r)%forc_hgt_u  
  if (grid(c,r)%forc_hgt_t      /= u) clm%forc_hgt_t =grid(c,r)%forc_hgt_t  
  if (grid(c,r)%forc_hgt_q      /= u) clm%forc_hgt_q =grid(c,r)%forc_hgt_q  

! CLM Vegetation parameters (read into 2-D grid module variables)

  if (grid(c,r)%dewmx           /= u) clm%dewmx      =grid(c,r)%dewmx    

! CLM Soil parameters	(read into 2-D grid module variables)

  if (grid(c,r)%smpmax          /= u) clm%smpmax     =grid(c,r)%smpmax    
  if (grid(c,r)%scalez          /= u) tile%scalez    =grid(c,r)%scalez   
  if (grid(c,r)%hkdepth         /= u) tile%hkdepth   =grid(c,r)%hkdepth    
  if (grid(c,r)%wtfact          /= u) clm%wtfact     =grid(c,r)%wtfact    
  if (grid(c,r)%trsmx0          /= u) clm%trsmx0     =grid(c,r)%trsmx0    

! Roughness lengths (read into 2-D grid module variables)

  if (grid(c,r)%zlnd            /= u) clm%zlnd       =grid(c,r)%zlnd    
  if (grid(c,r)%zsno            /= u) clm%zsno       =grid(c,r)%zsno    
  if (grid(c,r)%csoilc          /= u) clm%csoilc     =grid(c,r)%csoilc    

! Numerical finite-difference parameters (read into 2-D grid module variables)

  if (grid(c,r)%capr            /= u) clm%capr       =grid(c,r)%capr    
  if (grid(c,r)%cnfac           /= u) clm%cnfac      =grid(c,r)%cnfac    
  if (grid(c,r)%smpmin          /= u) clm%smpmin     =grid(c,r)%smpmin    
  if (grid(c,r)%ssi             /= u) clm%ssi        =grid(c,r)%ssi    
  if (grid(c,r)%wimp            /= u) clm%wimp       =grid(c,r)%wimp    
  if (grid(c,r)%pondmx          /= u) clm%pondmx     =grid(c,r)%pondmx 

  return
end subroutine drv_g2clm



