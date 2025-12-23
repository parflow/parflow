!#include <misc.h>

subroutine drv_readvegpf (drv,grid,tile,clm)

  !=========================================================================
  !
  !  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely  
  !  L                        M  available land surface process model.  
  !  M --COMMON LAND MODEL--  C  
  !  C                        L  CLM WEB INFO: http://www.clm.org?
  !  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
  !
  !=========================================================================
  ! DESCRIPTION:	  
  !   Read in vegetation class paramters from input file and assign to
  !   CLM variables.
  !
  ! INPUT DATA FORMAT:
  !  FORTRAN PARAMETER NAME, description (not read in)
  !  values (number of types in vegetation classification)
  !  
  !  This is free format, in any order.  drv_readvegp.f skips any comment lines
  !
  ! REVISION HISTORY:
  !  6 May 1999: Paul Houser; initial code
  !  15 Jan 2000: Paul Houser; revised for F90
  !=========================================================================
  ! $Id: drv_readvegpf.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_gridmodule      ! Grid space module
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use clm_varcon, only : istwet, istice, istdlak , istslak
  implicit none

  !=== Arguments ===========================================================

  type (drvdec)  :: drv              
  type (griddec) :: grid(drv%nc,drv%nr)
  type (tiledec) :: tile(drv%nch)
  type (clm1d)   :: clm (drv%nch)

  !=== Local Variables =====================================================

  character(15) :: vname   ! variable name read from clm_in.dat
  integer :: ioval,t       ! Read error code; tile space counter

  !=== End Variable List ===================================================

  ! Open and read 1-D  CLM input file
  open(9, file=drv%vegpf, form='formatted', status = 'old',action='read')


  ! Setup defaults; this prevents use of unitialized state
  do t=1,drv%nch 
     clm(t)%irrig = 0  !default - no irrigation
  end do

  ioval=0
  do while (ioval == 0)

     vname='!'
     read(9,'(a15)',iostat=ioval)vname
     if (vname == 'itypwat'  ) call drv_vpi(drv,tile,clm%itypwat)
     if (vname == 'lai0')      call drv_vpr(drv,tile,clm%minlai) 
     if (vname == 'lai')       call drv_vpr(drv,tile,clm%maxlai) 
     clm%tlai=clm%maxlai
     if (vname == 'sai')       call drv_vpr(drv,tile,clm%tsai  )
     if (vname == 'z0m')       call drv_vpr(drv,tile,clm%z0m   )
     if (vname == 'displa')    call drv_vpr(drv,tile,clm%displa)
     if (vname == 'dleaf')     call drv_vpr(drv,tile,clm%dleaf )
     if (vname == 'roota')     call drv_vpr(drv,tile,tile%roota)
     if (vname == 'rootb')     call drv_vpr(drv,tile,tile%rootb)
     if (vname == 'rhol_vis')  call drv_vpr(drv,tile,clm%rhol(1))
     if (vname == 'rhol_nir')  call drv_vpr(drv,tile,clm%rhol(2))
     if (vname == 'rhos_vis')  call drv_vpr(drv,tile,clm%rhos(1))
     if (vname == 'rhos_nir')  call drv_vpr(drv,tile,clm%rhos(2))
     if (vname == 'taul_vis')  call drv_vpr(drv,tile,clm%taul(1))
     if (vname == 'taul_nir')  call drv_vpr(drv,tile,clm%taul(2))
     if (vname == 'taus_vis')  call drv_vpr(drv,tile,clm%taus(1))
     if (vname == 'taus_nir')  call drv_vpr(drv,tile,clm%taus(2))
     if (vname == 'xl')        call drv_vpr(drv,tile,clm%xl)
     if (vname == 'vw')        call drv_vpr(drv,tile,clm%vw)
     if (vname == 'irrig')     call drv_vpi(drv,tile,clm%irrig)    ! @IMF
     if (vname == 'bkmult')    call drv_vpr(drv,tile,clm%bkmult)   ! @CAP 2014-02-24
     ! initialize lakpoi from itypwat variable

     do t=1,drv%nch 

        if (clm(t)%itypwat == istdlak .or. clm(t)%itypwat == istslak) then
           clm(t)%lakpoi = .true.
        else
           clm(t)%lakpoi = .false.
        endif

        if (tile(t)%vegt == 18) then  !bare soil index
           clm(t)%baresoil = .true.
        else
           clm(t)%baresoil = .false.
        endif

        ! IMF: Irrigation flag for each veg type added to drv_vegp.dat,  
        !      read from file in above routine)
        !      (irrig=0 -> no irrigation, irrig=1 -> irrigate)
        ! clm(t)%irrig = .false.  !for now - no irrigation 

     end do

  enddo
  close(9) 

end subroutine drv_readvegpf

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely  
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://www.clm.org?
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! drv_vp.f:
!
! DESCRIPTION:
! The following subroutine simply reads and distributes spatially-constant
!  data from drv_vegp.dat into clm arrays.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_vpi(drv,tile,clmvar)  

  ! Declare Modules and data structures
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  implicit none
  type (drvdec)           :: drv              
  type (tiledec)          :: tile(drv%nch)

  integer t
  integer clmvar(drv%nch)
  integer ivar(drv%nt)

  read(9,*)ivar
  do t=1,drv%nch
     clmvar(t)=ivar(tile(t)%vegt)
  enddo

end subroutine drv_vpi


subroutine drv_vpr(drv,tile,clmvar)  

  ! Declare Modules and data structures
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  implicit none
  type (drvdec)           :: drv              
  type (tiledec)          :: tile(drv%nch)

  integer t
  real(r8) clmvar(drv%nch)
  real(r8) rvar(drv%nt)

  read(9,*)rvar
  do t=1,drv%nch
     clmvar(t)=rvar(tile(t)%vegt)
  enddo

end subroutine drv_vpr
