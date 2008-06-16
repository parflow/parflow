!#include <misc.h>

 subroutine clm_lsm(pressure,saturation,temperature,evap_trans,latent_heat,forc_t,topo,porosity, &
                    dt,time,pdx,pdy,pdz,ix,iy,nx,ny,nz,nx_f,ny_f,nz_f,ip,npp,npq,npr,rank)

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
  use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
  implicit none
!  include 'mpif.h'

  type (drvdec)           :: drv              
  type (tiledec),pointer :: tile(:)
  type (griddec),pointer :: grid(:,:)   
  type (clm1d),pointer :: clm(:)

!=== Local Variables =====================================================

  integer :: t,m           ! tile space counter
  integer :: l             ! Stefan: layer counter 
  integer :: r,c           ! row,column indices
  integer :: ierr          ! error output 
  character(15) :: vname   ! variable name read from clm_in.dat
  integer :: ioval         ! Read error code
  integer :: I_err
  integer :: ntroute       !@ number of sub-timesteps for overland routing
  integer :: timeroute     !@ counter of sub-timesteps for overland routing
  
  integer :: nx,ny,nz,nx_f,ny_f,nz_f,steps
  real(r8) :: pressure_data(nx,ny,nz),saturation_data(nx,ny,nz),evap_trans_data(nx,ny,nz),mask_data(nx,ny,nz),porosity_data(nx,ny,nz)
  real(r8) :: temperature_data(nx,ny,nz),latent_heat_data(nx,ny,nz),forc_t_data(nx,ny,nz)
  real(r8) :: pressure((nx+2)*(ny+2)*(nz+2)),saturation((nx+2)*(ny+2)*(nz+2)),evap_trans((nx+2)*(ny+2)*(nz+2)),topo((nx+2)*(ny+2)*(nz+2))
  real(r8) :: temperature((nx+2)*(ny+2)*(nz+2)),latent_heat((nx+2)*(ny+2)*(nz+2)),forc_t((nx+2)*(ny+2)*(nz+2))
  real(r8) :: porosity((nx+2)*(ny+2)*(nz+2))
  real(r8) :: dt,time,otime,pdx,pdy,pdz
  integer  :: i,j,k,j_incr,k_incr,ip,ix,iy
  integer  :: npp,npq,npr !@ number of processors in x,y,z
  integer  :: rank,error
  integer  :: counter(nx,ny) 
  character*100 :: RI

!allocate (pressure_data(nx,ny,nz),saturation_data(nx,ny,nz),evap_trans_data(nx,ny,nz),mask_data(nx,ny,nz),porosity_data(nx,ny,nz))

!=== End Variable List ===================================================
ix = ix + 1 !Correction for CLM/Fortran space
iy = iy + 1 !Correction for CLM/Fortran space
!call MPI_COMM_RANK(MPI_COMM_WORLD, rank, error)
j_incr = nx_f - nx
k_incr = (nx_f * ny_f) - (ny * nx_f)
t = 0
l = ip
do k=1,nz ! PF loop over z
  do j=1,ny
    do i=1,nx
    t = t + 1
    l = l + 1
     saturation_data(i,j,k) = saturation(l)
     evap_trans_data(i,j,k) = evap_trans(l)
     pressure_data(i,j,k) = pressure(l)
     temperature_data(i,j,k) = temperature(l)
     mask_data(i,j,k) = topo(l)
     porosity_data(i,j,k) = porosity(l)
    enddo
  l = l + j_incr
  enddo
l = l + k_incr
enddo

!=========================================================================
!=== Initialize CLM
!=========================================================================

!=== Read in grid size domain
drv%dx = pdx
drv%dy = pdy
drv%dz = pdz
drv%nc = nx
drv%nr = ny
drv%nt = 18

write(RI,*) rank

if (time == 0.0d0) then ! Check if initialization necessary 
! print *,"INITIALIZATION"

!  open(10,file='drv_clmin.dat.'//trim(adjustl(RI)),form='formatted',status='old',action='read')

!  ioval=0
!  do while(ioval==0)
!     vname='!'
!     read(10,'(a15)',iostat=ioval)vname
!     if (vname == 'nc') call drv_get1divar(drv%nc)  
!     if (vname == 'nr') call drv_get1divar(drv%nr)  
!     if (vname == 'nt') call drv_get1divar(drv%nt)  
!  enddo
!  close(10)

!==== Adjust array size with respect to processor topology in x and y 
!==== This is done only if nx, ny are read in from a file 
!==== I assume that domain is NOT split in the z-direction
!drv%nc = drv%nc / npp
!drv%nr = drv%nr / npq


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
!print *,"Initialize CLM and DIAG variables"
  do t=1,drv%nch 
     clm%kpatch = t
     call drv_clmini (drv, grid, tile(t), clm(t))           !Initialize CLM Variables
  enddo

!@ Call to subroutine that reads in information on which cells are (in-)active due to topo
  !call topomask(clm,drv)

!@ Call to subroutine that reads in 2D array(s) of input data (e.g. hksat)
!print *,"Call to subroutine that reads in 2D array(s) of input data (e.g. porosity)"
  call read_array(drv,clm,rank)

!@ Initialize the CLM topography mask
!print *,"Initialize the CLM topography mask"
!print *,"DIMENSIONS",nx,nx_f,drv%nc,drv%nr,drv%nch
counter = 0
do t=1,drv%nch
i=tile(t)%col
j=tile(t)%row
  do k = nz, 1, -1 ! PF loop over z
   if (mask_data(i,j,k) == 1 .and. counter(i,j) < 10) then
     counter(i,j) = counter(i,j) + 1
     clm(t)%topo_mask(nz-k+1) = counter(i,j)
     clm(t)%planar_mask = 1
   else
     clm(t)%topo_mask(nz-k+1) = 0
   endif
  clm(t)%watsat(nz-k+1)=porosity_data(i,j,k)
  enddo
enddo

!@ Call to subroutine to open (2D-) output files
!print *,"Open (2D-) output files"
  call open_files(clm,drv,rank,ix,iy) 

!=== Read restart file or set initial conditions
!@ But first, get the original start time from drv_clmin.dat
!call drv_date2time(otime,drv%doy,drv%day,drv%gmt, &
!             drv%syr,drv%smo,drv%sda,drv%shr,drv%smn,drv%sss)

!print *,"Read restart file"
  call drv_restart(1,drv,tile,clm,rank)  !(1=read,2=write)

!call MPI_BCAST(clm,drv%nch,clm1d,0,MPI_COMM_WORLD,error)

!@ Jump to correct line in forcing file
do i = 1, clm(1)%istep
 read(11,*)
enddo

endif !======= End of the initialization ================

!=== Assign Parflow timestep in case it was cut ===
!if (dt /= 0.0d0) drv%ts = dt * 3600.0d0
!clm%dtime = dble(drv%ts)

 clm%qflx_infl_old = clm%qflx_infl
 clm%qflx_tran_veg_old = clm%qflx_tran_veg

!print *,"Call the Readout"
 call pfreadout(clm,drv,tile,saturation_data,pressure_data,temperature_data,rank,ix,iy) 

!=========================================================================
!=== Time looping
!=========================================================================
drv%endtime = 0

     call drv_tick(drv)

     !=== Read in the atmospheric forcing for off-line run
     !print *," Read in the atmospheric forcing for off-line run"
     call drv_getforce(drv,tile,clm)

     do t = 1, drv%nch     !Tile loop
       if(clm(t)%planar_mask == 1) call clm_main (clm(t), drv%day) !@ only call if there is an active CLM cell
     enddo ! End of the space vector loop 
           
     !=== Write CLM Output (timeseries model results)
     !=== note that drv_almaout needs to be completed

!    call drv_almaout (drv, tile, clm) !@ This routine was already inactivated in the original tar file 
     call drv_1dout (drv, tile,clm,rank)

!@== Stefan: call 2D output routine
     call drv_2dout (drv,grid,clm,rank)

     !=== Write Daily Restarts

!     if (drv%gmt==0..or.drv%endtime==1) call drv_restart(2,drv,tile,clm,rank)
     call drv_restart(2,drv,tile,clm,rank)
     
     call pf_couple(drv,clm,tile,evap_trans_data,latent_heat_data,forc_t_data)   

l = ip 
do k=1,nz ! PF loop over z
t = 0
  do j=1,ny
    do i=1,nx
    t = t + 1
    l = l + 1
    evap_trans(l) = evap_trans_data(i,j,k)
    latent_heat(l) = latent_heat_data(i,j,k)
    forc_t(l) = forc_t_data(i,j,k)
    enddo
  l = l + j_incr
  enddo
l = l + k_incr
enddo

!@Infrastructure to write and test the mask, et, and topography
!do t=1,drv%nch
!i=tile(t)%col
!j=tile(t)%row
!  do k=1,nz
!    if (i == 400 .and. j == 1 .and. rank == 0)then
!         write(777,'(3i,f,i2,2f,i)')i,j,k,(clm(t)%pf_press(nz-k+1)/1000.0d0),clm(t)%topo_mask(nz-k+1),clm(t)%pf_vol_liq(nz-k+1), &
!                                                  evap_trans_data(i,j,k),tile(t)%vegt
!    endif
!  enddo
!enddo

     !=== Return required surface fields to atmospheric model (return to grid space)

     call drv_clm2g (drv, grid, tile, clm)

     !=== Write spatially-averaged BC's and IC's to file for user

     if (clm(1)%istep==1) call drv_pout(drv,tile,clm,rank)
     
! enddo ! End the time loop for the model time steps
 
!@==  Call to subroutine to close (2D-) output files
if (drv%endtime /= 0)  call close_files(clm,drv,rank)

if (rank == 0) then
  open (1234,file="global_nt.scr",action='write')
  write(1234,*)"set i =",clm(1)%istep
  write(1234,*)"set endt =",drv%endtime
  close (1234)
endif

end subroutine clm_lsm 
