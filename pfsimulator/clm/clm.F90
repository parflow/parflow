!#include <misc.h>

subroutine clm_lsm(pressure,saturation,evap_trans,topo,porosity,istep_pf,dt,time,pdx,pdy,     &
pdz,ix,iy,nx,ny,nz,nx_f,ny_f,nz_f,ip,npp,npq,npr,rank,sw_pf,lw_pf,prcp_pf,tas_pf,u_pf,        &
v_pf,patm_pf,qatm_pf,eflx_lh_pf,eflx_lwrad_pf,eflx_sh_pf,eflx_grnd_pf,qflx_tot_pf,            &
qflx_grnd_pf,qflx_soi_pf,qflx_eveg_pf,qflx_tveg_pf,qflx_in_pf,swe_pf,t_g_pf, t_soi_pf,        &
clm_dump_interval,clm_1d_out,clm_output_dir,clm_output_dir_length,clm_bin_output_dir,         &
write_CLM_binary,beta_typepf,veg_water_stress_typepf,wilting_pointpf,field_capacitypf,        &
res_satpf,irr_typepf, irr_cyclepf, irr_ratepf, irr_startpf, irr_stoppf, irr_thresholdpf,      &
qirr_pf,qirr_inst_pf,irr_flag_pf,irr_thresholdtypepf)

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

  integer :: t             ! tile space counter
  integer :: l             ! Stefan: layer counter 
  integer :: r,c           ! row,column indices
  integer :: ierr          ! error output 

  integer :: nx,ny,nz,nx_f,ny_f,nz_f
  !  real(r8),allocatable :: pressure_data(:,:,:)   ! pressure on CLM grid (nx,ny,1:nlevsoi)
  !  real(r8),allocatable :: saturation_data(:,:,:) ! saturation on CLM grid (ny,ny,1:nlevsoi)
  !  real(r8),allocatable :: evap_trans_data(:,:,:) ! ET flux (combined) on CLM grid (nx,ny,1:nlevsoi)! 
  !  real(r8),allocatable :: porosity_data(:,:,:)
  real(r8) :: pressure((nx+2)*(ny+2)*(nz+2))     ! pressure head, from parflow on grid w/ ghost nodes for current proc
  real(r8) :: saturation((nx+2)*(ny+2)*(nz+2))   ! saturation from parflow, on grid w/ ghost nodes for current proc
  real(r8) :: evap_trans((nx+2)*(ny+2)*(nz+2))   ! ET flux from CLM to ParFlow on grid w/ ghost nodes for current proc
  real(r8) :: topo((nx+2)*(ny+2)*(nz+2))         ! mask from ParFlow 0 for inactive, 1 for active, on grid w/ ghost nodes for current proc
  real(r8) :: porosity((nx+2)*(ny+2)*(nz+2))     ! porosity from ParFlow, on grid w/ ghost nodes for current proc
  !  real(r8) :: res_sat((nx+2)*(ny+2)*(nz+2))      ! residual saturation from ParFlow, on grid w/ ghost nodes for current proc
  real(r8) :: dt                                 ! parflow dt in parflow time units not CLM time units
  real(r8) :: time                               ! parflow time in parflow units
  real(r8) :: pdx,pdy,pdz                        ! parflow DX, DY and DZ in parflow units
  integer  :: istep_pf                           ! istep, now passed from PF
  integer  :: ix                                 ! parflow ix, starting point for local grid on global grid
  integer  :: iy                                 ! parflow iy, starting point for local grid on global grid
  integer  :: ip                               
  integer  :: npp,npq,npr                        !@ number of processors in x,y,z
  integer  :: rank                               ! processor rank, from ParFlow
  
  !  surface fluxes form CLM
  !  eflx_lh added by RMM, copied for other CLM fluxes...
  real(r8) :: eflx_lh_pf((nx+2)*(ny+2)*3)        ! e_flux   (lh)    output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: eflx_lwrad_pf((nx+2)*(ny+2)*3)     ! e_flux   (lw)    output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: eflx_sh_pf((nx+2)*(ny+2)*3)        ! e_flux   (sens)  output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: eflx_grnd_pf((nx+2)*(ny+2)*3)      ! e_flux   (grnd)  output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: qflx_tot_pf((nx+2)*(ny+2)*3)       ! h2o_flux (total) output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: qflx_grnd_pf((nx+2)*(ny+2)*3)      ! h2o_flux (grnd)  output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: qflx_soi_pf((nx+2)*(ny+2)*3)       ! h2o_flux (soil)  output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: qflx_eveg_pf((nx+2)*(ny+2)*3)      ! h2o_flux (veg-e) output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: qflx_tveg_pf((nx+2)*(ny+2)*3)      ! h2o_flux (veg-t) output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: qflx_in_pf((nx+2)*(ny+2)*3)        ! h2o_flux (infil) output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: swe_pf((nx+2)*(ny+2)*3)            ! swe              output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: t_g_pf((nx+2)*(ny+2)*3)            ! t_grnd           output var to send to ParFlow, on grid w/ ghost nodes for current proc but nz=1 (2D)
  real(r8) :: t_soi_pf((nx+2)*(ny+2)*(nlevsoi+2))!tsoil             output var to send to ParFlow, on grid w/ ghost nodes for current proc, but nz=10 (3D)
  real(r8) :: sw_pf((nx+2)*(ny+2)*3)             ! SW rad, passed from PF
  real(r8) :: lw_pf((nx+2)*(ny+2)*3)             ! LW rad, passed from PF
  real(r8) :: prcp_pf((nx+2)*(ny+2)*3)           ! Precip, passed from PF
  real(r8) :: tas_pf((nx+2)*(ny+2)*3)            ! Air temp, passed from PF
  real(r8) :: u_pf((nx+2)*(ny+2)*3)              ! u-wind, passed from PF
  real(r8) :: v_pf((nx+2)*(ny+2)*3)              ! v-wind, passed from PF
  real(r8) :: patm_pf((nx+2)*(ny+2)*3)           ! air pressure, passed from PF
  real(r8) :: qatm_pf((nx+2)*(ny+2)*3)           ! air specific humidity, passed from PF

  ! IMF -- For passing irrigation amounts to write as silo in PF
  real(r8) :: irr_flag_pf((nx+2)*(ny+2)*3)       ! irrigation flag for deficit-based scheduling -- 1 = irrigate, 0 = no-irrigate
  real(r8) :: qirr_pf((nx+2)*(ny+2)*3)           ! irrigation applied above ground -- spray or drip (2D)
  real(r8) :: qirr_inst_pf((nx+2)*(ny+2)*(nlevsoi+2))! irrigation applied below ground -- 'instant' (3D)

  integer  :: clm_dump_interval                  ! dump inteval for CLM output, passed from PF, always in interval of CLM timestep, not time
  integer  :: clm_1d_out                         ! whether to dump 1d output 0=no, 1=yes
  integer  :: clm_output_dir_length
  character (LEN=clm_output_dir_length) :: clm_output_dir ! output dir location
  integer  :: clm_bin_output_dir
  integer  :: write_CLM_binary                   ! whether to write CLM output as binary 

  integer  :: beta_typepf                        ! beta formulation for bare soil Evap 0=none, 1=linear, 2=cos
  integer  :: veg_water_stress_typepf            ! veg transpiration water stress formulation 0=none, 1=press, 2=sm
  real(r8) :: wilting_pointpf                    ! wilting point in m if press-type, in saturation if soil moisture type
  real(r8) :: field_capacitypf                   ! field capacity for water stress same as units above
  real(r8) :: res_satpf                          ! residual saturation from ParFlow

  integer  :: irr_typepf                         ! irrigation type flag (0=none,1=spray,2=drip,3=instant)
  integer  :: irr_cyclepf                        ! irrigation cycle flag (0=constant,1=deficit)
  real(r8) :: irr_ratepf                         ! irrigation application rate for spray and drip [mm/s]
  real(r8) :: irr_startpf                        ! irrigation daily start time for constant cycle
  real(r8) :: irr_stoppf                         ! irrigation daily stop tie for constant cycle
  real(r8) :: irr_thresholdpf                    ! irrigation threshold criteria for deficit cycle (units of soil moisture content)
  integer  :: irr_thresholdtypepf                ! irrigation threshold criteria type -- top layer, bottom layer, column avg

  integer  :: j_incr,k_incr                      ! increment for j and k to convert 1D vector to 3D i,j,k array
  integer  :: i,j,k
  integer, allocatable  :: counter(:,:) 
  character*100 :: RI

  save

  !=== End Variable List ===================================================

  !=========================================================================
  !=== Initialize CLM
  !=========================================================================

  print*, "clm.F90: rank =", rank, "   istep =", istep_pf
  write(RI,*) rank

  !=== Read in grid size domain from PF
  drv%dx = pdx
  drv%dy = pdy
  drv%dz = pdz
  drv%nc = nx
  drv%nr = ny
  drv%nt = 18
  drv%ts = dt*3600.d0    !  assume PF in hours, CLM in seconds
  
!  clm_1d_out = 0
!  print*, 'clm dump interval', clm_dump_interval
!  print*, 'clm dump dir:', clm_output_dir
!  print*, 'clm 1d:',clm_1d_out
!  print*, 'clm dump lgnth: ', clm_output_dir_length
!  print*, 'dx:',drv%dx, pdx
!  print*, 'dy:',drv%dy, pdy
!  print*, 'dz:',drv%dz, pdz
!  print*, 'nr:',drv%nr, nx
!  print*, 'nc:',drv%nc, ny
!  print*, 'dt:',drv%ts, dt
!  print*, 'time:',time
!  i=3
!  j=3
!  k=10
!  l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2)*(k)
!  print*, 'press(l):',pressure(l)
   
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

     call drv_readclmin (drv,grid,rank)  

     !=== Allocate memory for subgrid tile space
     !=== This is done twice, because tile space size is initially unknown        
     !=== First - allocate max possible size, then allocate calculated size 

     !@ Stefan: I change the size of drv%nch right at the beginning, because we have 1 tile per grid cell
     !print *,"Allocate memory"
     ! drv%nch = drv%nr*drv%nc*drv%nt
     ! allocate (tile(drv%nch),stat=ierr); call drv_astp(ierr) 
     ! allocate (clm (drv%nch),stat=ierr); call drv_astp(ierr)

     ! write(*,*)"Call vegetation-data-read"
     ! call drv_readvegtf (drv, grid, tile, clm, rank)  !Determine actual NCH
     ! deallocate (tile,clm)                      !Save memory

     ! IMF: Because we only use one tile per grid cell, we don't need to call readvegtf to determine actual nch
     !      nch is just equal to number of cells (nr*nc)
     !      (revert to previous setup if we ever implement multiple tiles/cell in PF.CLM)
     drv%nch = drv%nr*drv%nc
     write(*,*)"Allocate Arrays", drv%nch
     allocate (tile(drv%nch), stat=ierr); call drv_astp(ierr) 
     allocate (clm (drv%nch), stat=ierr); call drv_astp(ierr)

     ! @RMM open balance and log files- don't write these at every timestep
     ! print*, "open files" 
     open (166,file='clm_elog.txt.'//trim(adjustl(RI)))
     open (199,file='balance.txt.'//trim(adjustl(RI)))
     write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"

     ! @RMM
     ! since we've moved the PF-CLM temp vars into the clmtype struture they are allocated in the above statements, prev alloc below
     ! allocate (pressure_data(nx,ny,nlevsoi),saturation_data(nx,ny,nlevsoi),evap_trans_data(nx,ny,nlevsoi),porosity_data(nx,ny,nlevsoi))

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
     call clm_typini(drv%nch,clm,istep_pf)

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
        call drv_clmini (drv, grid, tile(t), clm(t), istep_pf) !Initialize CLM Variables
     enddo

     !@ Call to subroutine that reads in information on which cells are (in-)active due to topo
     !call topomask(clm,drv)

     !@ Call to subroutine that reads in 2D array(s) of input data (e.g. hksat)
     !print *,"Call to subroutine that reads in 2D array(s) of input data (e.g. porosity)"
     !call read_array(drv,clm,rank)

     !@ Initialize the CLM topography mask 
     !@ RMM this is two components: 1) a x-y mask of 0 o 1 for active inactive and 
     !@ RMM  2) a z/k mask that takes three values (1)= top of LS/PF domain (2)= top-nlevsoi and
     !@ RMM  (3) the bottom of the LS/PF domain.
     print *,"Initialize the CLM topography mask"
     print *,"DIMENSIONS",nx,nx_f,drv%nc,drv%nr,drv%nch,ny,ny_f, nz, nz_f, ip

     j_incr = (nx+2) 
     k_incr = (nx+2) * (ny+2)
     do t=1,drv%nch
        i=tile(t)%col
        j=tile(t)%row
        counter(i,j) = 0
        clm(t)%topo_mask(3) = 1
        !  print*, t, i, j,ip 
        do k = nz, 1, -1 ! PF loop over z
           l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2)*(k)
            
           if (topo(l) == 1) then
              counter(i,j) = counter(i,j) + 1
              if (counter(i,j) == 1) then 
                 clm(t)%topo_mask(1) = k
                 clm(t)%planar_mask = 1
              end if
              !else
              !  clm(t)%topo_mask(nz-k+1) = 0
           endif
           !    print*, l, i,j,k, topo(l), clm(t)%topo_mask(1)
           if (topo(l) == 0 .and. topo(l+k_incr) == 1) clm(t)%topo_mask(3) = k+1
           !    print*, clm(t)%topo_mask(1), clm(t)%topo_mask(2), clm(t)%topo_mask(3)
        enddo
        clm(t)%topo_mask(2) = clm(t)%topo_mask(1)-nlevsoi
        !  print*, clm(t)%topo_mask(1), clm(t)%topo_mask(2), clm(t)%topo_mask(3)

     enddo

     ! loop over clm tile space 
     ! set up watsat and residual sat  (RMM)
     ! set irrigation flags (type, cycle, rate, start, stop, threshold)  (IMF)
     j_incr = nx_f 
     k_incr = (nx_f * ny_f)
     do t=1,drv%nch  ! loop over clm tile space

        ! for beta and veg stress formulations
        clm(t)%beta_type = beta_typepf
        clm(t)%vegwaterstresstype = veg_water_stress_typepf
        clm(t)%wilting_point = wilting_pointpf
        clm(t)%field_capacity = field_capacitypf
        clm(t)%res_sat = res_satpf

        ! for irrigation
        clm(t)%irr_type  = irr_typepf
        clm(t)%irr_cycle = irr_cyclepf
        clm(t)%irr_rate  = irr_ratepf
        clm(t)%irr_start = irr_startpf
        clm(t)%irr_stop  = irr_stoppf
        clm(t)%irr_threshold  = irr_thresholdpf     
        clm(t)%threshold_type = irr_thresholdtypepf
 
        ! set clm watsat, tksatu from PF porosity
        ! convert t to i,j index
        i=tile(t)%col        
        j=tile(t)%row
        do k = 1, nlevsoi ! loop over clm soil layers (1->nlevsoi)
           ! convert clm space to parflow space, note that PF space has ghost nodes
           l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
           ! l = 1+i + j_incr*(j-1) + k_incr*(clm(t)%topo_mask(1)-k)
           clm(t)%watsat(k)=porosity(l)
           ! if (k==1) clm(t)%res_sat = res_sat(l)
           clm(t)%tksatu(k)=clm(t)%tkmg(k)*0.57**clm(t)%watsat(k)
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
     call drv_restart(1,drv,tile,clm,rank,istep_pf)  !(1=read,2=write)

     !call MPI_BCAST(clm,drv%nch,clm1d,0,MPI_COMM_WORLD,error)
     !@ Jump to correct line in forcing file
     !IMF: forcing I/O now handled in PF, passed to CLM
     !clm(1)%istep = 1440
     !print*, clm(1)%istep
     !do i = 1, clm(1)%istep-1
     !   read(11,*)
     !enddo
  endif !======= End of the initialization ================

  j_incr = nx_f 
  k_incr = (nx_f * ny_f)

  !=== Assign Parflow timestep in case it was cut ===
  !if (dt /= 0.0d0) drv%ts = dt * 3600.0d0
  !clm%dtime = dble(drv%ts)

  !print*, "implied array copy of clm%qlux/old/veg"
  !clm%qflx_infl_old = clm%qflx_infl
  !clm%qflx_tran_veg_old = clm%qflx_tran_veg

  !print *,"Call the Readout"
  !call ParFlow --> CLM couple code
  !maps ParFlow space to CLM space @RMM
  call pfreadout(clm,drv,tile,saturation,pressure,rank,ix,iy,nx,ny,nz,j_incr, k_incr, ip) 

  !=========================================================================
  !=== Time looping
  !=========================================================================
  drv%endtime = 0

  call drv_tick(drv)

  ! IMF: forcing arrays passed to drv_getforce, forcing no longer read in drv_getforce
  !=== Read in the atmospheric forcing for off-line run
  !print *," Read in the atmospheric forcing for off-line run"
  !call drv_getforce(drv,tile,clm)
  call drv_getforce(drv,tile,clm,nx,ny,sw_pf,lw_pf,prcp_pf,tas_pf,u_pf,v_pf,patm_pf,qatm_pf,istep_pf)

  do t = 1, drv%nch     !Tile loop
     clm(t)%qflx_infl_old = clm(t)%qflx_infl
     clm(t)%qflx_tran_veg_old = clm(t)%qflx_tran_veg                    ! IMF: added gmt to next line for irrig schedule 
     if(clm(t)%planar_mask == 1) call clm_main (clm(t),drv%day,drv%gmt) ! @ only call if there is an active CLM cell
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
  if (mod(istep_pf,clm_dump_interval)==0)  then
     
     !IMF only call if write_CLM_binary==True 
     if (write_CLM_binary==1) then

        ! @ RMM 9-08 move file open to outside initialization loop
        ! @ RMM  this is now done every timestep specified by pf input file
        !@ Call to subroutine to open (2D-) output files
        !print *,"Open (2D-) output files"
        !print*, pressure(111),saturation(111),evap_trans(111),topo(111),vname,ierr,drv%dx,drv%nc,drv%nr
        !print*, clm(1)
        !print *,clm(1)%istep
        call open_files(clm,drv,rank,ix,iy,istep_pf,clm_output_dir, clm_output_dir_length,clm_bin_output_dir) 
        call drv_2dout (drv,grid,clm,rank)
        !@==  Call to subroutine to close (2D-) output files
        !@==  RMM modified to open/close files (but to include istep) every 
        !@== time step 
        !!if (drv%endtime /= 0)  call close_files(clm,drv,rank)
        !print*, "close files"
        call close_files(clm,drv,rank)
     end if ! write_CLM_binary
  end if  ! mod of istep and dump_interval

  !=== copy arrays from clm-to-pf for printing in pf as pfb or silo
  j_incr = (nx+2) 
  do t=1,drv%nch
     i=tile(t)%col
     j=tile(t)%row
     l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2) 
     eflx_lh_pf(l)=clm(t)%eflx_lh_tot
     eflx_lwrad_pf(l)=clm(t)%eflx_lwrad_out
     eflx_sh_pf(l)=clm(t)%eflx_sh_tot
     eflx_grnd_pf(l)=clm(t)%eflx_soil_grnd
     qflx_tot_pf(l)=clm(t)%qflx_evap_tot
     qflx_grnd_pf(l)=clm(t)%qflx_evap_grnd
     qflx_soi_pf(l)=clm(t)%qflx_evap_soi
     qflx_eveg_pf(l)=clm(t)%qflx_evap_veg 
     qflx_tveg_pf(l)=clm(t)%qflx_tran_veg
     qflx_in_pf(l)=clm(t)%qflx_infl 
     swe_pf(l)=clm(t)%h2osno 
     t_g_pf(l)=clm(t)%t_grnd
     qirr_pf(l)=clm(t)%qflx_qirr
     irr_flag_pf(l)=clm(t)%irr_flag
  enddo

  !3D arrays (tsoil)
  j_incr = (nx+2)
  k_incr = (nx+2) * (ny+2)
  do t=1,drv%nch            ! Loop over CLM tile space
     i=tile(t)%col
     j=tile(t)%row
     do k = 1,nlevsoi       ! Loop from 1 -> number of soil layers (in CLM)
        l = 1+i + j_incr*(j) + k_incr*(nlevsoi-(k-1))
        t_soi_pf(l)=clm(t)%t_soisno(k)
        qirr_inst_pf(l)=clm(t)%qflx_qirr_inst(k)
     enddo
  enddo

  !=== Write Daily Restarts
  print*, 'time =', time, 'gmt =', drv%gmt, 'endtime =', drv%endtime
  if (drv%gmt==0..or.drv%endtime==1) call drv_restart(2,drv,tile,clm,rank,istep_pf)
  ! call drv_restart(2,drv,tile,clm,rank)
  ! call PF couple, this transfers ET from CLM to ParFlow 
  ! as evap_trans flux	     
  call pf_couple(drv,clm,tile,evap_trans,saturation,pressure,porosity,nx,ny,nz,j_incr,k_incr,ip,istep_pf)   

  !=== Return required surface fields to atmospheric model (return to grid space)
  call drv_clm2g (drv, grid, tile, clm)

  !=== Write spatially-averaged BC's and IC's to file for user
  if (istep_pf==1) call drv_pout(drv,tile,clm,rank)

! IMF istep is passed from PF to CLM, this is no longer needed
!  if (rank == 0) then
!     open (1234,file="global_nt.scr",action='write')
!     write(1234,*)"set i =",clm(1)%istep
!     write(1234,*)"set endt =",drv%endtime
!     close (1234)
!  endif

  !@RMM if at end of simulation, close all files
  if (drv%endtime==1) then
     close(166)
     close(199)
  end if
! print*, 'return'
! close(11)

end subroutine clm_lsm
