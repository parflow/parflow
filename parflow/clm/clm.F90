!#include <misc.h>

 subroutine clm_lsm(pressure,saturation,evap_trans,topo,porosity,dt,time,pdx,pdy,pdz,ix,iy,nx,ny,nz,nx_f,ny_f,nz_f,ip,npp,npq,npr,rank,clm_dump_interval,clm_1d_out, clm_output_dir, clm_output_dir_length,clm_bin_output_dir)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  	
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use drv_gridmodule      ! Grid-space variables
  use clmtype             ! CLM tile variables
!  use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
  use clm_varpar

  implicit none
!  include 'mpif.h'

  type (drvdec)           :: drv              
  type (tiledec),pointer :: tile(:)
  type (griddec),pointer :: grid(:,:)   
  type (clm1d),pointer :: clm(:)
  
!  type (tiledec), allocatable :: tile(:)
!  type (griddec),allocatable :: grid(:,:)   
!  type (clm1d),allocatable :: clm(:)
  
!=== Local Variables =====================================================

  integer :: t,m           ! tile space counter
  integer :: l             ! Stefan: layer counter 
  integer :: r,c           ! row,column indices
  integer :: ierr          ! error output 
  character(15) :: vname   ! variable name read from clm_in.dat
  integer :: ioval         ! Read error code
  integer :: I_err

  
  integer :: nx,ny,nz,nx_f,ny_f,nz_f,steps
!  real(r8),allocatable :: pressure_data(:,:,:)   ! pressure on CLM grid (nx,ny,1:nlevsoi)
!  real(r8),allocatable :: saturation_data(:,:,:) ! saturation on CLM grid (ny,ny,1:nlevsoi)
!  real(r8),allocatable :: evap_trans_data(:,:,:) ! ET flux (combined) on CLM grid (nx,ny,1:nlevsoi)! 
!  real(r8),allocatable :: porosity_data(:,:,:)
  real(r8) :: pressure((nx+2)*(ny+2)*(nz+2))     ! pressure head, from parflow on grid w/ ghost nodes for current proc
  real(r8) :: saturation((nx+2)*(ny+2)*(nz+2))   ! saturation from parflow, on grid w/ ghost nodes for current proc
  real(r8) :: evap_trans((nx+2)*(ny+2)*(nz+2))   ! ET flux from CLM to ParFlow on grid w/ ghost nodes for current proc
  real(r8) :: topo((nx+2)*(ny+2)*(nz+2))         ! mask from ParFlow 0 for inactive, 1 for active, on grid w/ ghost nodes for current proc
  real(r8) :: porosity((nx+2)*(ny+2)*(nz+2))     ! porosity from ParFlow, on grid w/ ghost nodes for current proc
  real(r8) :: dt                                 ! parflow dt in parflow time units not CLM time units
  real(r8) :: time                               ! parflow time in parflow units
  real(r8) :: otime               
  real(r8) :: pdx,pdy,pdz                        ! parflow DX, DY and DZ in parflow units
  integer  :: ix                                 ! parflow ix, starting point for local grid on global grid
  integer  :: iy                                 ! parflow iy, starting point for local grid on global grid
  integer  :: ip                               
  integer  :: npp,npq,npr                        !@ number of processors in x,y,z
  integer  :: rank		                         ! processor rank, from ParFlow
  integer  :: clm_dump_interval                  ! dump inteval for CLM output, passed from PF, always in interval of CLM timestep, not time
  integer  :: clm_1d_out                         ! whether to dump 1d output 0=no, 1=yes
  integer :: clm_output_dir_length
  character (LEN=clm_output_dir_length) :: clm_output_dir                ! output dir location
  integer :: clm_bin_output_dir
  integer  :: error
real elapsed(2)
  integer  :: j_incr,k_incr                      ! increment for j and k to convert 1D vector to 3D i,j,k array

  integer  :: i,j,k,ll,tt
  integer, allocatable  :: counter(:,:) 
  character*100 :: RI

save

!=== End Variable List ===================================================


!=========================================================================
!=== Initialize CLM
!=========================================================================

!=== Read in grid size domain from PF
drv%dx = pdx
drv%dy = pdy
drv%dz = pdz
drv%nc = nx
drv%nr = ny
drv%nt = 18

!clm_1d_out = 0

write(RI,*) rank

print*, 'clm dump interval', clm_dump_interval
print*, 'clm dump dir:', clm_output_dir
print*, 'clm 1d:',clm_1d_out
print*, 'clm dump lgnth: ', clm_output_dir_length

if (time == 0.0d0) then ! Check if initialization necessary 
!open(6,file='clm.out.txt')
 print *,"INITIALIZATION"
allocate( counter(nx,ny))



!=== Allocate Memory for Grid Module
!print *,"Allocate Memory for Grid Module"

  allocate (grid(drv%nc,drv%nr),stat=ierr) ; call drv_astp(ierr) 
  do r=1,drv%nr     !rows
     do c=1,drv%nc  !columns
        allocate (grid(c,r)%fgrd(drv%nt))
        allocate (grid(c,r)%pveg(drv%nt))
     enddo      
  enddo         

!=== Read in the clm input file (drv_clmin.dat)
!print *,"Read in the clm input file (drv_clmin.dat)"

  call drv_readclmin (drv, grid,rank)  
  
!=== Allocate memory for subgrid tile space
!=== This is done twice, because tile space size is initially unknown        
!=== First - allocate max possible size, then allocate calculated size 

!@ Stefan: I change the size of drv%nch right at the beginning, because we have 1 tile per grid cell
!print *,"Allocate memory"
  drv%nch = drv%nr*drv%nc*drv%nt
  allocate (tile(drv%nch),stat=ierr); call drv_astp(ierr) 
  allocate (clm (drv%nch),stat=ierr); call drv_astp(ierr)
  
  write(*,*)"Call vegetation-data-read"
  call drv_readvegtf (drv, grid, tile, clm, rank)  !Determine actual NCH

  write(*,*)"Allocate Arrays", drv%nch
  deallocate (tile,clm)                      !Save memory
  allocate (tile(drv%nch), stat=ierr); call drv_astp(ierr) 
  allocate (clm (drv%nch), stat=ierr); call drv_astp(ierr)
  
! @RMM open balance and log files- don't write these at every timestep
! print*, "open files" 
 open (166,file='clm_elog.txt.'//trim(adjustl(RI)))

 open (199,file='balance.txt.'//trim(adjustl(RI)))
 write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"
   
! @RMM
! since we've moved the PF-CLM temp vars into the clmtype struture they are allocated in the above statements, prev alloc below
!allocate (pressure_data(nx,ny,nlevsoi),saturation_data(nx,ny,nlevsoi),evap_trans_data(nx,ny,nlevsoi),porosity_data(nx,ny,nlevsoi))

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
  call drv_readvegtf (drv, grid, tile, clm, rank)

!=== Transfer grid variables to tile space 
  write(*,*)"Transfer grid variables to tile space ", drv%nch
  do t = 1, drv%nch
    call drv_g2clm (drv%udef, drv, grid, tile(t), clm(t))   
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

!@ Call to subroutine that reads in 2D array(s) of input data (e.g. hksat)
!print *,"Call to subroutine that reads in 2D array(s) of input data (e.g. porosity)"
!  call read_array(drv,clm,rank)

!@ Initialize the CLM topography mask 
!@ RMM this is two components: 1) a x-y mask of 0 o 1 for active inactive and 
!@ RMM  2) a z/k mask that takes three values (1)= top of LS/PF domain (2)= top-nlevsoi and
!@ RMM  (3) the bottom of the LS/PF domain.
print *,"Initialize the CLM topography mask"
print *,"DIMENSIONS",nx,nx_f,drv%nc,drv%nr,drv%nch,ny,ny_f, nz, nz_f, ip

j_incr = nx_f 
k_incr = (nx_f * ny_f)
!print*, j_incr,k_incr
do t=1,drv%nch
i=tile(t)%col
j=tile(t)%row
counter(i,j) = 0
  clm(t)%topo_mask(3) = 1
!  print*, t, i, j,ip 
  do k = nz, 1, -1 ! PF loop over z
   l = 1+i + j_incr*(j) + k_incr*(k)
!   print*, l, i,j,k, topo(l), clm(t)%topo_mask(1)
   if (topo(l) == 1) then
     counter(i,j) = counter(i,j) + 1
     if (counter(i,j) == 1) then 
	 clm(t)%topo_mask(1) = k
     clm(t)%planar_mask = 1
	 end if
   !else
   !  clm(t)%topo_mask(nz-k+1) = 0
   endif
   if (topo(l) == 0 .and. topo(l+k_incr) == 1) clm(t)%topo_mask(3) = k+1
!     print*, clm(t)%topo_mask(1), clm(t)%topo_mask(2), clm(t)%topo_mask(3)
  enddo
  clm(t)%topo_mask(2) = clm(t)%topo_mask(1)-nlevsoi
!  print*, clm(t)%topo_mask(1), clm(t)%topo_mask(2), clm(t)%topo_mask(3)
   
enddo


!set up watsat
j_incr = nx_f 
k_incr = (nx_f * ny_f)
do t=1,drv%nch  ! loop over clm tile space
!convert t to i,j index
i=tile(t)%col
j=tile(t)%row
! loop from 1, number of soil layers (in CLM)
  do k = 1, nlevsoi
! convert clm space to parflow space, note that PF space has ghost nodes
  l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
  	! 	l = 1+i + j_incr*(j-1) + k_incr*(clm(t)%topo_mask(1)-k)
  clm(t)%watsat(k)=porosity(l)
 ! print*, i,j,k,t,l,clm(t)%topo_mask(1),porosity(l),clm(t)%watsat(k)

  end do !k
end do !t

!! below is for debugging.
!! it allows a check of old CLM mapping from PF->CLM and 
!! compares to new CLM mapping
!
!t = 0
l = ip
!print*,ip
!do k=1,nz ! PF loop over z
!  do j=1,ny
!    do i=1,nx
!    t = t + 1
! find t from i,j
!do tt = 1, drv%nch
!if ( (i==tile(tt)%col).and.(j==tile(tt)%row) ) t = tt
!end do !  tt
!    l = l + 1
!	   ll = 1+i + j_incr*(j) + k_incr*(k)
!if (k==clm(t)%topo_mask(1)) then
!print*, k,ll,l,clm(t)%topo_mask(1)
!print*, clm(t)%watsat(1),porosity(l)
!end if
!	print*,i,j,k,l,ll,topo(l),porosity(l)
!     saturation_data(i,j,k) = saturation(l)
!     evap_trans_data(i,j,k) = evap_trans(l)
!     pressure_data(i,j,k) = pressure(l)
!     mask_data(i,j,k) = topo(l)
!     porosity_data(i,j,k) = porosity(l)
!    enddo
!  l = l +  nx_f - nx
!  enddo
!l = l + (nx_f * ny_f) - (ny * nx_f)
!enddo

!=== Read restart file or set initial conditions
!@ But first, get the original start time from drv_clmin.dat
!call drv_date2time(otime,drv%doy,drv%day,drv%gmt, &
!             drv%syr,drv%smo,drv%sda,drv%shr,drv%smn,drv%sss)

!print *,"Read restart file",etime(elapsed)
  call drv_restart(1,drv,tile,clm,rank)  !(1=read,2=write)

!call MPI_BCAST(clm,drv%nch,clm1d,0,MPI_COMM_WORLD,error)

!@ Jump to correct line in forcing file
do i = 1, clm(1)%istep
 read(11,*)
enddo

endif !======= End of the initialization ================

j_incr = nx_f 
k_incr = (nx_f * ny_f)

!=== Assign Parflow timestep in case it was cut ===
!if (dt /= 0.0d0) drv%ts = dt * 3600.0d0
!clm%dtime = dble(drv%ts)

!print*, "implied array copy of clm%qlux/old/veg"
! clm%qflx_infl_old = clm%qflx_infl
! clm%qflx_tran_veg_old = clm%qflx_tran_veg

print *,"Call the Readout"
! call ParFlow --> CLM couple code
! maps ParFlow space to CLM space @RMM
 call pfreadout(clm,drv,tile,saturation,pressure,rank,ix,iy,nx,ny,nz,j_incr, k_incr, ip) 

!=========================================================================
!=== Time looping
!=========================================================================
drv%endtime = 0

     call drv_tick(drv)

     !=== Read in the atmospheric forcing for off-line run
     !print *," Read in the atmospheric forcing for off-line run"
     call drv_getforce(drv,tile,clm)

     do t = 1, drv%nch     !Tile loop
	  clm(t)%qflx_infl_old = clm(t)%qflx_infl
      clm(t)%qflx_tran_veg_old = clm(t)%qflx_tran_veg
       if(clm(t)%planar_mask == 1) call clm_main (clm(t), drv%day) !@ only call if there is an active CLM cell
     enddo ! End of the space vector loop 
           
     !=== Write CLM Output (timeseries model results)
     !=== note that drv_almaout needs to be completed

!    call drv_almaout (drv, tile, clm) !@ This routine was already inactivated in the original tar file 
	if (clm_1d_out == 1) &
     call drv_1dout (drv, tile,clm,rank)

!@== Stefan: call 2D output routine
! @RMM now we only call for every clm_dump_interval steps (not 
! time units, integer units)
!print*, clm(1)%istep, clm_dump_interval, mod(clm(1)%istep,clm_dump_interval)
     if (mod(clm(1)%istep,clm_dump_interval)==0)  then
! @ RMM 9-08 move file open to outside initialization loop
! @ RMM  this is now done every timestep specified by pf input file
!@ Call to subroutine to open (2D-) output files
!print *,"Open (2D-) output files"
!print*, pressure(111),saturation(111),evap_trans(111),topo(111),vname,ierr,drv%dx,drv%nc,drv%nr
!print*, clm(1)
!print *,clm(1)%istep
  call open_files(clm,drv,rank,ix,iy,clm(1)%istep,clm_output_dir, clm_output_dir_length,clm_bin_output_dir) 
  call drv_2dout (drv,grid,clm,rank)
!@==  Call to subroutine to close (2D-) output files
!@==  RMM modified to open/close files (but to include istep) every 
!@== time step 
!!if (drv%endtime /= 0)  call close_files(clm,drv,rank)
 !print*, "close files"
  call close_files(clm,drv,rank)
end if  ! mod of istep and dump_interval

     !=== Write Daily Restarts

     if (drv%gmt==0..or.drv%endtime==1) call drv_restart(2,drv,tile,clm,rank)
!     call drv_restart(2,drv,tile,clm,rank)
 
! call PF couple, this transfers ET from CLM to ParFlow 
! as evap_trans flux	     
     call pf_couple(drv,clm,tile,evap_trans,saturation, pressure, porosity, nx,ny,nz,j_incr, k_incr,ip)   


     !=== Return required surface fields to atmospheric model (return to grid space)
print*, "drv_clm2g"
     call drv_clm2g (drv, grid, tile, clm)

     !=== Write spatially-averaged BC's and IC's to file for user
print*, "drv_pout"
     if (clm(1)%istep==1) call drv_pout(drv,tile,clm,rank)
print*, "drv pout"     
! enddo ! End the time loop for the model time steps
 

if (rank == 0) then
  open (1234,file="global_nt.scr",action='write')
  write(1234,*)"set i =",clm(1)%istep
  write(1234,*)"set endt =",drv%endtime
  close (1234)
endif

!@RMM if at end of simulation, close all files
     if (drv%endtime==1) then
	 close(166)
	 close(199)
	 end if
print*, 'return'

end subroutine clm_lsm 
