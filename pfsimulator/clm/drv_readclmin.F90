!#include <misc.h>

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
!  1-D user defined CLM parameters.
!  NOTE: For 2-D runs, it is recommended that drv_readclmin.f or drv_main.f be 
!  modified to allow the specification of spatially-variable parameters.  
!  This file is still used for all spatially-constant variabiles when
!  running CLM in 2-D. 
!
!  This subroutine works entirely in grid space.  It assigns initial spatially
!  constant values to the entire model domain in grid space based on values
!  contained in the drv_clmin.dat file.  If spatially variable grid fields are
!  to be read in, then this should be done near the end of this routine.
!  if spatially-variable tile space fields are desired, then they should be 
!  read in at ... PRH
!
!  NOTE on INDEXES: There are several index soil and vegetation values.  These
!  are followed by several parameters that are defined by the index.  If a
!  -999.0 is placed in the fields after the index, then the default index
!  value will be used.  However, if it is desired to override this default
!  value, simply replace the -999.0 with the value you desire.
!
! INPUT DATA FORMAT:
!  FORTRAN PARAMETER NAME, VALUE, description (not read in)
!  This is free format, in any order.  readclmin.f skips any comment lines
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!  15 Jan 2000: Paul Houser; significant revision for new CLM version
!   3 March 2000:     Jon Radakovich; Revision for diagnostic output
!=========================================================================
! $Id: drv_readclmin.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

subroutine drv_readclmin(drv,grid,rank,clm_write_logs)

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_gridmodule      ! Grid-space variables
  implicit none

!=== Arguments ===========================================================

  type (drvdec)  :: drv              
  type (griddec) :: grid(drv%nc,drv%nr)
  integer :: clm_write_logs

!=== Local Variables =====================================================

  integer :: c,r           ! Column and Row dimensions [-]
  character(15) :: vname   ! variable name read from clm_in.dat
  integer :: ioval         ! Read error code

  integer :: rank
  character*100 RI

!=== End Variable List ===================================================

   write(RI,*) rank

   !=== Reading of the clm_in file occurs in 2 steps:
   !===  (1) Read in CLM Domain, and allocate module sizes
   !===  (2) Read in CLM parameters and assign to grid module variables


   !=== Step (1): Open and read domain info into drv_module from clm_in input file

  ! open(10, file='drv_clmin.dat.'//trim(adjustl(RI)), form='formatted', status = 'old',action='read')
  open(10, file='drv_clmin.dat', form='formatted', status = 'old',action='read')
   ioval=0
   do while(IOVAL == 0)
      vname='!'
      read(10,'(a15)',iostat=ioval) vname

      ! CLM domain keys
      if (vname == 'maxt')        call drv_get1divar(drv%maxt)  
      if (vname == 'mina')        call drv_get1drvar(drv%mina)  
      if (vname == 'udef')        call drv_get1drvar(drv%udef)  
      if (vname == 'vclass')      call drv_get1divar(drv%vclass)  

      ! CLM file names
      if (vname == 'vegtf')       call drv_get1dcvar(drv%vegtf)  
      if (vname == 'vegpf')       call drv_get1dcvar(drv%vegpf)  
      if (vname == 'metf1d')      call drv_get1dcvar(drv%metf1d)  
      if (vname == 'poutf1d')     call drv_get1dcvar(drv%poutf1d)  
      if (vname == 'outf1d')      call drv_get1dcvar(drv%outf1d)
      if (vname == 'rstf')        call drv_get1dcvar(drv%rstf)  

      ! Run timing parameters
      if (vname == 'startcode')   call drv_get1divar(drv%startcode)  
      if (vname == 'ts') then
         print *,"CLM Error : ts value specified in drv_clmin.dat file."
         print *,"        ts should not be specified in drv_clmin.dat file for Parflow/CLM runs."
         call abort
      endif
      if (vname == 'sss')         call drv_get1divar(drv%sss)  
      if (vname == 'smn')         call drv_get1divar(drv%smn)  
      if (vname == 'shr')         call drv_get1divar(drv%shr)  
      if (vname == 'sda')         call drv_get1divar(drv%sda)  
      if (vname == 'smo')         call drv_get1divar(drv%smo)  
      if (vname == 'syr')         call drv_get1divar(drv%syr)  
      if (vname == 'ess')         call drv_get1divar(drv%ess)  
      if (vname == 'emn')         call drv_get1divar(drv%emn)  
      if (vname == 'ehr')         call drv_get1divar(drv%ehr)  
      if (vname == 'eda')         call drv_get1divar(drv%eda)  
      if (vname == 'emo')         call drv_get1divar(drv%emo)  
      if (vname == 'eyr')         call drv_get1divar(drv%eyr)  

      ! IC Source: (1) restart file, (2) drv_clmin.dat (this file)
      if (vname == 'clm_ic')      call drv_get1divar(drv%clm_ic)  
 
      ! CLM initial conditions (Read into 1D drv_module variables)
      if (vname == 't_ini')       call drv_get1drvar(drv%t_ini)  
      if (vname == 'h2osno_ini')  call drv_get1drvar(drv%h2osno_ini)  
      if (vname == 'sw_ini')      call drv_get1drvar(drv%sw_ini)
      if (vname == 'surfind')     call drv_get1divar(drv%surfind)
      if (vname == 'soilind')     call drv_get1divar(drv%soilind)
      if (vname == 'snowind')     call drv_get1divar(drv%snowind) 

   enddo
   close(10)


   !=== Open and read 1D  CLM input file
!   open(10, file='drv_clmin.dat.'//trim(adjustl(RI)), form='formatted', status = 'old',action='read')
   open(10, file='drv_clmin.dat', form='formatted', status = 'old',action='read')

   ioval=0
   do while (ioval == 0)
      vname='!'
      read(10,'(a15)',iostat=ioval)vname
      c=drv%nc
      r=drv%nr

      ! CLM Forcing parameters (read into 2-D grid module variables)
      if (vname == 'forc_hgt_u')  call drv_get2drvar(c,r,grid%forc_hgt_u)
      if (vname == 'forc_hgt_t')  call drv_get2drvar(c,r,grid%forc_hgt_t)
      if (vname == 'forc_hgt_q')  call drv_get2drvar(c,r,grid%forc_hgt_q)

      ! CLM Vegetation parameters (read into 2-D grid module variables)
      if (vname == 'dewmx')       call drv_get2drvar(c,r,grid%dewmx)
      if (vname == 'rootfr')      call drv_get2drvar(c,r,grid%rootfr)

      ! CLM Soil parameters (read into 2-D grid module variables)
      if (vname == 'smpmax')      call drv_get2drvar(c,r,grid%smpmax)
      if (vname == 'scalez')      call drv_get2drvar(c,r,grid%scalez)
      if (vname == 'hkdepth')     call drv_get2drvar(c,r,grid%hkdepth)
      if (vname == 'wtfact')      call drv_get2drvar(c,r,grid%wtfact)
      if (vname == 'trsmx0')      call drv_get2drvar(c,r,grid%trsmx0)

      ! Roughness lengths (read into 2-D grid module variables)
      if (vname == 'zlnd')        call drv_get2drvar(c,r,grid%zlnd)
      if (vname == 'zsno')        call drv_get2drvar(c,r,grid%zsno)
      if (vname == 'csoilc')      call drv_get2drvar(c,r,grid%csoilc)

      ! Numerical finite-difference parameters (read into 2-D grid module variables)
      if (vname == 'capr')        call drv_get2drvar(c,r,grid%capr)
      if (vname == 'cnfac')       call drv_get2drvar(c,r,grid%cnfac)
      if (vname == 'smpmin')      call drv_get2drvar(c,r,grid%smpmin)
      if (vname == 'ssi')         call drv_get2drvar(c,r,grid%ssi)
      if (vname == 'wimp')        call drv_get2drvar(c,r,grid%wimp)
      if (vname == 'pondmx')      call drv_get2drvar(c,r,grid%pondmx)

   enddo
   close(10)

if (clm_write_logs==1) then
   !=== Open files for time series output
   ! If restarting from a restart file then assume append to old output file
   if (drv%startcode == 1) then  ! Append to old output file
      open(20,file=trim(adjustl(drv%outf1d))//"."//trim(adjustl(RI)), form='formatted', position='append')
   else
      open(20,file=trim(adjustl(drv%outf1d))//"."//trim(adjustl(RI))) !,form='unformatted')
   endif
endif

   !=== Read in 2-D and 3-D (GRID SPACE) parameter arrays here (to overwrite 1-D arrays read above)
   !===  NOTE TO USER: READ IN YOUR 2-D PARAMETERS & INITIAL CONDITIONS HERE


end subroutine drv_readclmin



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
! The following subroutine simply reads and distributes spatially-constant
!  data from clm_in.dat into clm arrays.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_get2divar(nc,nr,clmvar)  
  implicit none
  character*15 vname  
  integer nc,nr,x,y
  integer clmvar(nc,nr)
  integer ivar

  backspace(10)
  read(10,*) vname,ivar
  do x=1,nc
     do y=1,nr
        clmvar(x,y)=ivar
     enddo
  enddo
end subroutine drv_get2divar



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
! The following subroutine simply reads and distributes spatially-constant
!  data from clm_in.dat into clm arrays.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_get2drvar(nc,nr,clmvar)  
  use precision
  implicit none  
  character*15 vname  
  integer nc,nr,x,y
  real(r8) clmvar(nc,nr)
  real(r8) rvar

  backspace(10)
  read(10,*) vname,rvar
  do x=1,nc
     do y=1,nr
        clmvar(x,y)=rvar
     enddo
  enddo
end subroutine drv_get2drvar




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
! The following subroutine simply reads data from clm_in.dat drv module.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_get1divar(drvvar)  
  use precision
  implicit none
  character*15 vname  
  integer drvvar

  backspace(10)
  read(10,*) vname,drvvar
end subroutine drv_get1divar




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
! The following subroutine simply reads data from clm_in.dat drv module.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_get1drvar(drvvar)  
  use precision
  implicit none
  character*15 vname  
  real(r8) drvvar

  backspace(10)
  read(10,*) vname,drvvar
end subroutine drv_get1drvar




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
! The following subroutine simply reads data from clm_in.dat drv module.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_get1dcvar(drvvar)  
  use precision
  implicit none
  character*15 vname  
  character*40 drvvar

  backspace(10)
  read(10,*) vname,drvvar
end subroutine drv_get1dcvar





