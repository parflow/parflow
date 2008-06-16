!#include <misc.h>

program drv_main

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
!  Off-line, tile-based, land surface driver.  
!  This program facilitates input and output to/from the CLM. 
!  See clm_main.F90 for a full CLM description and philosophy statement.
!  This driver code is provided as an example of how the user might
!  facilitate input and output from the CLM model.  It is intended that
!  the user modify this driver for specific applications, but the user
!  is encouraged to adopt its general structure.
!
! TILE SPACE:
!  Sub-grid processes are modeled on homogenous sub-grid regions called
!  tiles.  Grid-space is translated to a tile-space vector based on 
!  sub-grid vegetation information.  The land model calculations are
!  performed in this tile space vector, and fluxes are aggregated back 
!  to grid space for output or coupling to the atmosphere. 
!
! CLM DRIVER INPUT FILES:
!  1-D Parameter File  (drv_clmin.dat)
!    Contains CLM run specifications and all spatially constant parameters
!
!  2-D Parameter File  (drv_vegm.dat)
!    Contains 2-D spatially variable parameters in grid space
!
!  Veg Class File      (drv_vegp.dat)
!    Contains vegetation class parameters, can be modified for various classifications
!
!
! REVISION HISTORY:
!  15 December 1999:  Paul Houser and Jon Radakovich; Initial Code 
!   3 March 2000:     Jon Radakovich; Revision for diagnostic output
!=========================================================================
! $Id: drv_main.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use dfport
  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use drv_gridmodule      ! Grid-space variables
  use clmtype             ! CLM tile variables
  use casctype
  use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
  implicit none

  type (drvdec)           :: drv              
  type (tiledec), pointer :: tile(:)
  type (griddec), pointer :: grid(:,:)   
  type (clm1d),   pointer :: clm(:)
  type (casc2D),  pointer :: casc(:,:)     

!=== Local Variables =====================================================

  integer :: t,m           ! tile space counter
  integer :: l             ! Stefan: layer counter 
  integer :: r,c           ! row,column indices
  integer :: ierr          ! error output 
  character(15) :: vname   ! variable name read from clm_in.dat
  integer :: ioval         ! Read error code
  integer :: I_err
  integer :: k,j
  integer :: ntroute       !@ number of sub-timesteps for overland routing
  integer :: timeroute     !@ counter of sub-timesteps for overland routing

!=== End Variable List ===================================================


!=========================================================================
!=== Initialize CLM
!=========================================================================

!=== Read in grid size domain
print *,"Read in grid size domain"
  open(10,file='drv_clmin.dat',form='formatted',status='old')
  ioval=0
  do while(ioval==0)
     vname='!'
     read(10,'(a15)',iostat=ioval)vname
     if (vname == 'nc') call drv_get1divar(drv%nc)  
     if (vname == 'nr') call drv_get1divar(drv%nr)  
     if (vname == 'nt') call drv_get1divar(drv%nt)  
  enddo
  close(10)

!=== Allocate Memory for Grid Module
print *,"Allocate Memory for Grid Module"

  allocate (grid(drv%nc,drv%nr),stat=ierr) ; call drv_astp(ierr) 
  do r=1,drv%nr     !rows
     do c=1,drv%nc  !columns
        allocate (grid(c,r)%fgrd(drv%nt))
        allocate (grid(c,r)%pveg(drv%nt))
     enddo      
  enddo         

!=== Read in the clm input file (drv_clmin.dat)
print *,"Read in the clm input file (drv_clmin.dat)"

  call drv_readclmin (drv, grid)  
  
!=== Allocate memory for subgrid tile space
!=== This is done twice, because tile space size is initially unknown        
!=== First - allocate max possible size, then allocate calculated size 

!@ Stefan: I change the size of drv%nch right at the beginning, because we have 1 tile per grid cell
print *,"Allocate memory"
  drv%nch = drv%nr*drv%nc !drv%nt*
  allocate (tile(drv%nch),stat=ierr); call drv_astp(ierr) 
  allocate (clm (drv%nch),stat=ierr); call drv_astp(ierr)
  
  write(*,*)"Call vegetation-data-read"
  call drv_readvegtf (drv, grid, tile, clm)  !Determine actual NCH

  write(*,*)"Allocate Arrays"
  deallocate (tile,clm)                      !Save memory
  allocate (tile(drv%nch), stat=ierr); call drv_astp(ierr) 
  allocate (clm (drv%nch), stat=ierr); call drv_astp(ierr)
!@ Stefan: I allocate casc and mask here
  allocate (casc(drv%nc,drv%nr), stat=ierr); call drv_astp(ierr)

!=== Set clm diagnostic indices and allocate space

  clm%surfind = drv%surfind 
  clm%soilind = drv%soilind
  clm%snowind = drv%snowind

  do t=1,drv%nch 
     allocate (clm(t)%diagsurf(1:drv%surfind             ),stat=ierr); call drv_astp(ierr) 
     allocate (clm(t)%diagsoil(1:drv%soilind,1:nlevsoi   ),stat=ierr); call drv_astp(ierr)
     allocate (clm(t)%diagsnow(1:drv%snowind,-nlevsno+1:0),stat=ierr); call drv_astp(ierr)
  end do

!=== Initialize clm derived type components
  write(*,*)"Call clm_typini"
  call clm_typini(drv%nch, clm)

!=== Read in vegetation data and set tile information accordingly
  write(*,*)"Read in vegetation data and set tile information accordingly"
  call drv_readvegtf (drv, grid, tile, clm)

!=== Transfer grid variables to tile space 
  write(*,*)"Transfer grid variables to tile space ", drv%nch
  do t = 1, drv%nch
    call drv_g2clm (casc(tile(t)%col,tile(t)%row),drv%udef, drv, grid, tile(t), clm(t))   
  enddo

!=== Read vegetation parameter data file for IGBP classification
  write(*,*)"Read vegetation parameter data file for IGBP classification"
  call drv_readvegpf (drv, grid, tile, clm)  

!=== Initialize CLM and DIAG variables
print *,"Initialize CLM and DIAG variables"
  do t=1,drv%nch 
     clm%kpatch = t
     call drv_clmini (drv, grid, tile(t), clm(t))           !Initialize CLM Variables
  enddo

!@ Call to subroutine that reads in information on which cells are (in-)active due to topo
  !call topomask(clm,drv)
  drv%dx = 20.0d0
  drv%dy = 20.0d0
  drv%dz = 0.2d0

!@ Call to subroutine that reads in 2D array(s) of input data (e.g. hksat)
print *,"Call to subroutine that reads in 2D array(s) of input data (e.g. porosity)"
  call read_array(casc,drv,clm)
    
!@ Call to subroutine to open (2D-) output files
print *,"Open (2D-) output files"
  call open_files(clm,drv) 

!=== Read restart file or set initial conditions
  call drv_restart(1,drv,tile,clm)  !(1=read,2=write)

! run pf one time to initialize
 if (drv%clm_ic /= 1) then
  write(*,*)"========= Initialize PARFLOW =========="
  I_err = SYSTEM("tclsh pfprep2.tcl")
 endif
 call pfreadout(clm,drv) 

!=========================================================================
!=== Time looping
!=========================================================================

drv%endtime=0

  do while (drv%endtime==0) !Continue until Endtime is reached

     call drv_tick(drv)

     !=== Read in the atmospheric forcing for off-line run
     call drv_getforce(drv,tile,clm)

     do t = 1, drv%nch     !Tile loop
       !if(clm(t)%planar_mask == 1) 
       call clm_main (clm(t), drv%day) !@ only call if there is an active CLM cell
     enddo ! End of the space vector loop 
     !@=== Ceck Courant-condition =============
     !call courant(casc,clm,drv,ntroute)
	 !=== End check Courant-condition

!@=== Check available storage and partition precip into infil and runoff
     call runoff_infl(clm,drv)

!@== Define sub-timestep for overland routing
     !drv%dt = float(drv%ts) / float(ntroute)
	 !write(*,*)"new dtime:",drv%dt

!@== Stefan: call overland routing routines
    !call overland(casc,clm,drv,ntroute)
           
!@== Stefan: Start of couple for quasi-distributed model
!@== Stefan: watch out for the sequence of the loop, which  must be consistent with Parflow!?
     print *,"Call the couple"
     call pf_couple(drv,clm)
     print *,"=============Couple finished================="
     
!== @Stefan: End of couple for quasi-distributed model      

     !=== Write CLM Output (timeseries model results)
     !=== note that drv_almaout needs to be completed

!    call drv_almaout (drv, tile, clm) !@ This routine was already inactivated in the original tar file 
     call drv_1dout (drv, tile, clm)

!@== Stefan: call 2D output routine
     call drv_2dout (casc, drv,tile, clm)

     !=== Write Daily Restarts

     if (drv%gmt==0..or.drv%endtime==1) call drv_restart(2,drv,tile,clm)
   
     !=== Return required surface fields to atmospheric model (return to grid space)

     call drv_clm2g (drv, grid, tile, clm)

     !=== Write spatially-averaged BC's and IC's to file for user

     if (clm(1)%istep==1) call drv_pout(drv,tile,clm)
     
 enddo ! End the time loop for the model time steps
 
!@==  Call to subroutine to close (2D-) output files
  call close_files(clm,drv)

end program drv_main








