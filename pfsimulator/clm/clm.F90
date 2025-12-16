!#include <misc.h>

subroutine clm_lsm(pressure,saturation,evap_trans,top,bottom,porosity,pf_dz_mult,istep_pf,dt,time,           &
start_time,pdx,pdy,pdz,ix,iy,nx,ny,nz,nx_f,ny_f,nz_f,nz_rz,ip,npp,npq,npr,gnx,gny,rank,sw_pf,lw_pf,    &
prcp_pf,tas_pf,u_pf,v_pf,patm_pf,qatm_pf,lai_pf,sai_pf,z0m_pf,displa_pf,                               &
slope_x_pf,slope_y_pf,                                                                                 &
eflx_lh_pf,eflx_lwrad_pf,eflx_sh_pf,eflx_grnd_pf,                                                     &
qflx_tot_pf,qflx_grnd_pf,qflx_soi_pf,qflx_eveg_pf,qflx_tveg_pf,qflx_in_pf,swe_pf,t_g_pf,               &
t_soi_pf,clm_dump_interval,clm_1d_out,clm_forc_veg,clm_output_dir,clm_output_dir_length,clm_bin_output_dir,         &
write_CLM_binary,slope_accounting_CLM,beta_typepf,veg_water_stress_typepf,wilting_pointpf,field_capacitypf,                 &
res_satpf,irr_typepf, irr_cyclepf, irr_ratepf, irr_startpf, irr_stoppf, irr_thresholdpf,               &
qirr_pf,qirr_inst_pf,irr_flag_pf,irr_thresholdtypepf,soi_z,clm_next,clm_write_logs,                    &
clm_last_rst,clm_daily_rst,rz_water_stress_typepf, pf_nlevsoi, pf_nlevlak)

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
  use clm_varpar

  implicit none

  type (drvdec)          :: drv
  type (tiledec),pointer :: tile(:)
  type (griddec),pointer :: grid(:,:)
  type (clm1d),pointer   :: clm(:)

  ! IMF...
  ! This added call to set-up parameters...
  ! use clm_varpar
  !=== Parameters ==========================================================
  ! integer :: nz_rz                               ! number of layers, now passed from ParFlow 
  ! call clm_varpar(
  ! integer, parameter :: nlevsoi     =  nz_rz     !number of soil levels
  ! integer, parameter :: nlevlak     =  10        !number of lake levels
  ! integer, parameter :: nlevsno     =  5    !number of maximum snow levels
  ! integer, parameter :: numrad      =   2   !number of solar radiation bands: vis, nir
  ! integer, parameter :: numcol      =   8   !number of soil color types

  integer  :: pf_nlevsoi                         ! number of soil levels, passed from PF
  integer  :: pf_nlevlak                         ! number of lake levels, passed from PF
 
  !=== Local Variables =====================================================

  ! basic indices, counters
  integer  :: t                                   ! tile space counter
  integer  :: l,ll                                ! layer counter 
  integer  :: r,c                                 ! row,column indices
  integer  :: ierr                                ! error output 

  ! values passed from parflow
  integer  :: nx,ny,nz,nx_f,ny_f,nz_f,nz_rz
  integer  :: soi_z                               ! NBE: Specify layer should be used for reference temperature
  real(r8) :: pressure((nx+2)*(ny+2)*(nz+2))     ! pressure head, from parflow on grid w/ ghost nodes for current proc
  real(r8) :: saturation((nx+2)*(ny+2)*(nz+2))   ! saturation from parflow, on grid w/ ghost nodes for current proc
  real(r8) :: evap_trans((nx+2)*(ny+2)*(nz+2))   ! ET flux from CLM to ParFlow on grid w/ ghost nodes for current proc
  real(r8) :: top((nx+2)*(ny+2)*(3))             ! top Z index from ParFlow, -1 for inactive, on grid w/ ghost nodes for current proc
  real(r8) :: bottom((nx+2)*(ny+2)*(3))          ! bottom Z index from ParFlow, -1 for inactive, on grid w/ ghost nodes for current proc
  real(r8) :: porosity((nx+2)*(ny+2)*(nz+2))     ! porosity from ParFlow, on grid w/ ghost nodes for current proc
  real(r8) :: pf_dz_mult((nx+2)*(ny+2)*(nz+2))   ! dz multiplier from ParFlow on PF grid w/ ghost nodes for current proc
  real(r8) :: dt                                 ! parflow dt in parflow time units not CLM time units
  real(r8) :: time                               ! parflow time in parflow units
  real(r8) :: start_time                         ! starting time in parflow units
  real(r8) :: pdx,pdy,pdz                        ! parflow DX, DY and DZ in parflow units
  integer  :: istep_pf                           ! istep, now passed from PF
  integer  :: ix                                 ! parflow ix, starting point for local grid on global grid
  integer  :: iy                                 ! parflow iy, starting point for local grid on global grid
  integer  :: ip                               
  integer  :: npp,npq,npr                        ! number of processors in x,y,z
  integer  :: gnx, gny                           ! global grid, nx and ny
  integer  :: rank                               ! processor rank, from ParFlow

  integer :: clm_next                           ! NBE: Passing flag to sync outputs
  integer :: d_stp                              ! NBE: Dummy for CLM restart
  integer :: clm_write_logs                     ! NBE: Enable/disable writing of the log files
  integer :: clm_last_rst                       ! NBE: Write all the CLM restart files or just the last one
  integer :: clm_daily_rst                      ! NBE: Write daily restart files or hourly

  ! surface fluxes & forcings
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
  real(r8) :: t_soi_pf((nx+2)*(ny+2)*(pf_nlevsoi+2))!tsoil             output var to send to ParFlow, on grid w/ ghost nodes for current proc, but nz=10 (3D)
  real(r8) :: sw_pf((nx+2)*(ny+2)*3)             ! SW rad, passed from PF
  real(r8) :: lw_pf((nx+2)*(ny+2)*3)             ! LW rad, passed from PF
  real(r8) :: prcp_pf((nx+2)*(ny+2)*3)           ! Precip, passed from PF
  real(r8) :: tas_pf((nx+2)*(ny+2)*3)            ! Air temp, passed from PF
  real(r8) :: u_pf((nx+2)*(ny+2)*3)              ! u-wind, passed from PF
  real(r8) :: v_pf((nx+2)*(ny+2)*3)              ! v-wind, passed from PF
  real(r8) :: patm_pf((nx+2)*(ny+2)*3)           ! air pressure, passed from PF
  real(r8) :: qatm_pf((nx+2)*(ny+2)*3)           ! air specific humidity, passed from PF
  real(r8) :: lai_pf((nx+2)*(ny+2)*3)            ! BH: lai, passed from PF
  real(r8) :: sai_pf((nx+2)*(ny+2)*3)            ! BH: sai, passed from PF
  real(r8) :: z0m_pf((nx+2)*(ny+2)*3)            ! BH: z0m, passed from PF
  real(r8) :: displa_pf((nx+2)*(ny+2)*3)         ! BH: displacement height, passed from PF
  real(r8) :: irr_flag_pf((nx+2)*(ny+2)*3)       ! irrigation flag for deficit-based scheduling -- 1 = irrigate, 0 = no-irrigate
  real(r8) :: qirr_pf((nx+2)*(ny+2)*3)           ! irrigation applied above ground -- spray or drip (2D)
  real(r8) :: qirr_inst_pf((nx+2)*(ny+2)*(pf_nlevsoi+2))! irrigation applied below ground -- 'instant' (3D)

  real(r8) :: slope_x_pf((nx+2)*(ny+2)*3)        ! Slope in x-direction from PF
  real(r8) :: slope_y_pf((nx+2)*(ny+2)*3)        ! Slope in y-direction from PF

  ! output keys
  integer :: clm_dump_interval                  ! dump interval for CLM output, passed from PF, always in interval of CLM timestep, not time
  integer  :: clm_1d_out                         ! whether to dump 1d output 0=no, 1=yes
  integer  :: clm_forc_veg                       ! BH: whether vegetation (LAI, SAI, z0m, displa) is being forced 0=no, 1=yes
  integer  :: clm_output_dir_length              ! for output directory
  integer  :: clm_bin_output_dir                 ! output directory
  integer  :: write_CLM_binary                   ! whether to write CLM output as binary 
  integer  :: slope_accounting_CLM               ! account for slope is solar zenith angle calculations
  character (LEN=clm_output_dir_length) :: clm_output_dir ! output dir location

  ! ET keys
  integer  :: beta_typepf                        ! beta formulation for bare soil Evap 0=none, 1=linear, 2=cos
  integer  :: veg_water_stress_typepf            ! veg transpiration water stress formulation 0=none, 1=press, 2=sm
  integer  :: rz_water_stress_typepf             ! RZ transpiration limit formulation 0=none, 1=distributed discussed in Ferguson, Jefferson et al EES 2016
  real(r8) :: wilting_pointpf                    ! wilting point in m if press-type, in saturation if soil moisture type
  real(r8) :: field_capacitypf                   ! field capacity for water stress same as units above
  real(r8) :: res_satpf                          ! residual saturation from ParFlow

  ! irrigation keys
  integer  :: irr_typepf                         ! irrigation type flag (0=none,1=spray,2=drip,3=instant)
  integer  :: irr_cyclepf                        ! irrigation cycle flag (0=constant,1=deficit)
  real(r8) :: irr_ratepf                         ! irrigation application rate for spray and drip [mm/s]
  real(r8) :: irr_startpf                        ! irrigation daily start time for constant cycle
  real(r8) :: irr_stoppf                         ! irrigation daily stop tie for constant cycle
  real(r8) :: irr_thresholdpf                    ! irrigation threshold criteria for deficit cycle (units of soil moisture content)
  integer  :: irr_thresholdtypepf                ! irrigation threshold criteria type -- top layer, bottom layer, column avg

  ! local indices & counters
  integer  :: i,j,k,k1,j1,l1                     ! indices for local looping
  integer  :: bj,bl                              ! indices for local looping !BH

  integer  :: j_incr,k_incr                      ! increment for j and k to convert 1D vector to 3D i,j,k array
  real(r8) :: total
  character*100 :: RI
  real(r8) :: u         ! Tempoary UNDEF Variable

  real(r8) pf_porosity(pf_nlevsoi)  !porosity from PF, replaces watsat clm var

  save

  !=== End Variable List ===================================================

  !=========================================================================
  !=== Initialize CLM
  !=========================================================================

  !=== Open CLM text output
  write(RI,*)  rank

! NBE: Throughout clm.F90, any writes to unit 999 are now prefaced with the logical to disable the
!       writing of the log files. This greatly reduces the number of files created during a run.
  if (clm_write_logs==1) open(999, file="clm_output.txt."//trim(adjustl(RI)), action="write")
  if (clm_write_logs==1) write(999,*) "clm.F90: rank =", rank, "   istep =", istep_pf

  !=== Specify grid size using values passed from PF
  drv%dx = pdx
  drv%dy = pdy
  drv%dz = pdz
  drv%nc = nx
  drv%nr = ny                   
  drv%nt = 18                  ! 18 IGBP land cover classes
  drv%ts = dt*3600.d0          ! Assume PF in hours, CLM in seconds
  j_incr = nx_f
  k_incr = nx_f*ny_f

  !=== levels passed from PF
  nlevsoi = pf_nlevsoi
  nlevlak = pf_nlevlak

  !=== Check if initialization is necessary
  if (time == start_time) then 
     
     if (clm_write_logs==1) write(999,*) "INITIALIZATION"

!RMM: writing a CLM.out.clm.log file with basic information only from the master node (0 processor)
!
  if (rank==0) then
  open(9919, file="CLM.out.clm.log",action="write")
  write(9919,*) "******************************"
  write(9919,*) " CLM log basic output"
  write(9919,*)
  write(9919,*) "CLM starting istep =", istep_pf
  end if ! CLM log

     !=== Allocate Memory for Grid Module
     allocate (grid(drv%nc,drv%nr),stat=ierr) ; call drv_astp(ierr) 
     do r=1,drv%nr                              ! rows
        do c=1,drv%nc                           ! columns
           grid(c,r)%smpmax = u                 ! SGS Added initialization to address valgrind issues
           grid(c,r)%scalez = u
           grid(c,r)%hkdepth = u
           grid(c,r)%wtfact = u
           grid(c,r)%trsmx0 = u
           grid(c,r)%pondmx = u
           allocate (grid(c,r)%fgrd(drv%nt))
           allocate (grid(c,r)%pveg(drv%nt))
        enddo                                   ! columns
     enddo                                      ! rows

     !=== Read in the clm input (drv_clmin.dat)
     call drv_readclmin (drv,grid,rank,clm_write_logs)

     if (rank==0) then
       write(9919,*) "CLM startcode for date (1=restart, 2=defined):", drv%startcode
       write(9919,*) "CLM IC (1=restart, 2=defined):", drv%clm_ic
    !=== @RMM check for error in IC or starting time
       if (drv%startcode == 0) stop
       if (drv%clm_ic == 0) stop


     end if
     !=== Allocate memory for subgrid tile space
     !=== LEGACY =============================================================================================
     !=== (Keeping around in case we go back to multiple tiles per cell)

     !=== This is done twice, because tile space size is initially unknown        
     !=== First - allocate max possible size, then allocate calculated size 
     !=== Allocate maximum NCH
     ! if (clm_write_logs==1) write(999,*) "Allocate arrays -- using maximum NCH"
     ! drv%nch = drv%nr*drv%nc*drv%nt
     ! allocate (tile(drv%nch),stat=ierr); call drv_astp(ierr) 
     ! allocate (clm (drv%nch),stat=ierr); call drv_astp(ierr)

     !=== Read vegetation data to determine actual NCH
     ! if (clm_write_logs==1) write(999,*) "Call vegetation-data-read (drv_readvegtf), determines actual NCH"
     ! call drv_readvegtf (drv, grid, tile, clm, rank)               !Determine actual NCH
     ! deallocate (tile,clm)                                         !Deallocate to save memory

     !=== Allocate for calculated NCH
     ! if (clm_write_logs==1) write(999,*) "Allocate arrays -- actual NCH"
     ! allocate (tile(drv%nch),stat=ierr); call drv_astp(ierr)
     ! allocate (clm (drv%nch),stat=ierr); call drv_astp(ierr)


     !=== CURRENT =============================================================================================
     !=== Because we only use one tile per grid cell, we don't need to call readvegtf to determine actual nch
     !    (nch is just equal to number of cells (nr*nc))
     drv%nch = drv%nr*drv%nc
     if (clm_write_logs==1) write(999,*) "Allocate arrays -- using NCH =", drv%nch
     allocate (tile(drv%nch), stat=ierr); call drv_astp(ierr) 
     allocate (clm (drv%nch), stat=ierr); call drv_astp(ierr)


     !=== Open balance and log files - don't write these at every timestep
    ! open (166,file='clm_elog.txt.'//trim(adjustl(RI)))
    ! open (199,file='balance.txt.'//trim(adjustl(RI)))
    ! write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"


     !=== Set clm diagnostic indices and allocate space
     clm%surfind = drv%surfind 
     clm%soilind = drv%soilind
     clm%snowind = drv%snowind

     do t=1,drv%nch 
        allocate (clm(t)%diagsurf(1:drv%surfind             ),stat=ierr); call drv_astp(ierr) 
        allocate (clm(t)%diagsoil(1:drv%soilind,1:nlevsoi   ),stat=ierr); call drv_astp(ierr)
        allocate (clm(t)%diagsnow(1:drv%snowind,-nlevsno+1:0),stat=ierr); call drv_astp(ierr)
     end do

     !====================================================
     !NBE: Define the reference layer for the seasonal soi
     clm%soi_z = soi_z                  ! Probably out of place
     if (clm_write_logs==1) write(999,*) "Check soi_z",clm%soi_z

     !=== Initialize clm derived type components
     if (clm_write_logs==1) write(999,*) "Call clm_typini"
     call clm_typini(drv%nch,clm,istep_pf)
     
     if (clm_write_logs==1) then
     write(999,*) "DIMENSIONS:"
     write(999,*) 'local NX:',nx,' NX with ghost:',nx_f,' IX:', ix
     write(999,*) 'local NY:',ny,' NY with ghost:',ny_f,' IY:',iy
     write(999,*) 'PF    NZ:',nz, 'NZ with ghost:',nz_f
     write(999,*) 'global  NX:',gnx, ' global NY:', gny
     write(999,*) 'DRV-NC:',drv%nc,' DRV-NR:',drv%nr, 'DRV-NCH:',drv%nch
     write(999,*) ' Processor Number:',rank, ' local vector start:',ip
     endif
     !=== Read in vegetation data and set tile information accordingly
     if (clm_write_logs==1) write(999,*) "Read in vegetation data and set tile information accordingly"
     call drv_readvegtf (drv, grid, tile, clm, nx, ny, ix, iy, gnx, gny, rank)


     !=== Transfer grid variables to tile space 
     if (clm_write_logs==1) write(999,*) "Transfer grid variables to tile space ", drv%nch
     do t = 1, drv%nch
        call drv_g2clm (drv%udef, drv, grid, tile(t), clm(t))   
     enddo

     !=== Read vegetation parameter data file for IGBP classification
     if (clm_write_logs==1) write(999,*) "Read vegetation parameter data file for IGBP classification"
     call drv_readvegpf (drv, grid, tile, clm)  


     !=== Initialize CLM and DIAG variables
     if (clm_write_logs==1) write(999,*) "Initialize CLM and DIAG variables"
     do t=1,drv%nch 
        clm%kpatch = t

        !=== Initialize the CLM topography mask  @RMM  moved up from loop below
        !    This is two components:
        !    1) a x-y mask of 0 o 1 for active inactive and
        !    2) a z/k mask that takes three values
        !      (1)= top of LS/PF domain
        !      (2)= top-nlevsoi and
        !      (3)= the bottom of the LS/PF domain.
        if (clm_write_logs==1 .and. t==1) write(999,*) "Initialize the CLM topography mask"

        i=tile(t)%col
        j=tile(t)%row
        clm(t)%topo_mask(3) = 1

        l = 1+i + j_incr*(j) + k_incr
        if (top(l) > 0) then
           clm(t)%topo_mask(1) = 1+top(l)
           clm(t)%topo_mask(3) = 1+bottom(l)
           clm(t)%planar_mask = 1
        endif
        clm(t)%topo_mask(2) = clm(t)%topo_mask(1)-nlevsoi

        ! set clm watsat, tksatu from PF porosity
        do k = 1, nlevsoi ! loop over clm soil layers (1->nlevsoi)
           ! convert clm space to parflow space, note that PF space has ghost nodes
           l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
           ! put ParFlow porosity in a temp variable passed to clm_ini
           pf_porosity(k)       = porosity(l)
           !print*, 'k=',k,'l=',l,'porosity=',porosity(l),'pf_poro=',pf_porosity(k)

           !clm(t)%tksatu(k)       = clm(t)%tkmg(k)*0.57**clm(t)%watsat(k)
        end do !k

        call drv_clmini (drv, grid, pf_porosity,tile(t), clm(t), istep_pf, clm_forc_veg) !Initialize CLM Variables
     enddo ! t

     !=== IMF:
     !    Check planar mask...
     ! open(161,file='planar_mask.txt', action='write')
     ! do t=1,drv%nch
     !    i=tile(t)%col
     !    j=tile(t)%row
     !    write(161,*) t, i, j, clm(t)%planar_mask
     ! enddo ! t
     ! close(161)
     
     !=== IMF:
     !    Set up variable DZ over root column
     !    -- Copy dz multipliers for root zone cells from PF grid to 1D array
     !    -- Then loop to recompute clm(t)%z(j), clm(t)%dz(j), clm(t)%zi(j) 
     !       (replaces values set in drv_clmini)
     do t = 1,drv%nch

        i = tile(t)%col
        j = tile(t)%row

		!!!! BH: modification of the interfaces depths and layers thicknesses to match PF definitions
	    clm(t)%zi(0)            = 0.   
    
        ! check if cell is active
        if (clm(t)%planar_mask == 1) then

           ! reset node depths (clm%z) based on variable dz multiplier
           do k = 1, nlevsoi
              l                 = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
	      clm(t)%dz(k)	= drv%dz * pf_dz_mult(l) ! basile
              if (k==1) then
                 clm(t)%z(k)    = 0.5 * drv%dz * pf_dz_mult(l)
	      	clm(t)%zi(k)	= drv%dz * pf_dz_mult(l) ! basile
              else
                 total          = 0.0
                 do k1 = 1, k-1
                    l1          = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k1-1))
                    total       = total + (drv%dz * pf_dz_mult(l1))
                 enddo
                 clm(t)%z(k)       = total + (0.5 * drv%dz * pf_dz_mult(l))
		clm(t)%zi(k)	= total + drv%dz * pf_dz_mult(l)! basile
 
              endif
    
           enddo


          !! BH : the following is the previous version: commented
          ! ! set dz values (node thickness)
          ! ! (computed from node depths as in original CLM -- not always equal to PF dz values!)
          ! clm(t)%dz(1)            = 0.5*(clm(t)%z(1)+clm(t)%z(2))         !thickness b/n two interfaces
          ! do k = 2,nlevsoi-1
          !    clm(t)%dz(k)         = 0.5*(clm(t)%z(k+1)-clm(t)%z(k-1))
          ! enddo
          ! clm(t)%dz(nlevsoi)      = clm(t)%z(nlevsoi)-clm(t)%z(nlevsoi-1)
!
          ! ! set zi values (interface depths)
          ! ! (computed from node depths as in original CLM -- not always equal to PF interfaces!)
          ! clm(t)%zi(0)            = 0.                             !interface depths
          ! do k = 1, nlevsoi-1
          !    clm(t)%zi(k)         = 0.5*(clm(t)%z(k)+clm(t)%z(k+1))
          ! enddo
          ! clm(t)%zi(nlevsoi)      = clm(t)%z(nlevsoi) + 0.5*clm(t)%dz(nlevsoi)
!
 !!          ! PRINT CHECK
!          do k = 1, nlevsoi
!             l                 = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
!             if (clm_write_logs==1) write(999,*) "DZ CHECK -- ", i, j, k, l, pf_dz_mult(l), clm(t)%dz(k), clm(t)%z(k), &
!                 clm(t)%zi(k),clm(t)%rootfr(k)
!           enddo
          !! BH : commented (end)
		  
		   !! BH: Overwrite Rootfr disttribution: start
           !! BH: the following overwrites the root fraction definition which is previously set up in drv_clmini.F90 
		   !! BH: but based on constant DZ, regardless of pf_dz_mult.
           do bj = 1, nlevsoi-1
           clm(t)%rootfr(bj) = .5*( exp(-tile(t)%roota*clm(t)%zi(bj-1))  &
                           + exp(-tile(t)%rootb*clm(t)%zi(bj-1))  &
                           - exp(-tile(t)%roota*clm(t)%zi(bj  ))  &
                           - exp(-tile(t)%rootb*clm(t)%zi(bj  )) )
           enddo
           clm(t)%rootfr(nlevsoi)=.5*( exp(-tile(t)%roota*clm(t)%zi(nlevsoi-1))&
                               + exp(-tile(t)%rootb*clm(t)%zi(nlevsoi-1)))

           ! reset depth variables assigned by user in clmin file 
           do bl=1,nlevsoi
              if (grid(tile(t)%col,tile(t)%row)%rootfr /= drv%udef) &
                 clm(t)%rootfr(bl)=grid(tile(t)%col,tile(t)%row)%rootfr    
           enddo

 !!          ! PRINT CHECK
           !do k = 1, nlevsoi
           !  l                 = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
           !  if (clm_write_logs==1) write(999,*) "DZ CHECK -- ", i, j, k, l, pf_dz_mult(l), clm(t)%dz(k), clm(t)%z(k), &
           !      clm(t)%zi(k),clm(t)%rootfr(k)
           !enddo
           !! BH: Overwrite Rootfr disttribution: end
		   
		   endif ! active/inactive

     enddo !t 
   
   !! Loop over the tile space to assign slopes

      do t=1,drv%nch

        i=tile(t)%col
        j=tile(t)%row
      ll =  (1+i) + (nx+2)*(j) + (nx+2)*(ny+2)
      if (slope_accounting_CLM==1) then
      clm(t)%slope_x = slope_x_pf(ll)
      clm(t)%slope_y = slope_y_pf(ll)
      else
      clm(t)%slope_x = 0.0d0
      clm(t)%slope_y = 0.0d0
      end if
      end do ! t

     !=== Loop over CLM tile space to set keys/constants from PF
     !    (watsat, residual sat, irrigation keys)
     do t=1,drv%nch  

        ! check if cell is active
        if (clm(t)%planar_mask == 1) then

           ! for beta and veg stress formulations
           clm(t)%beta_type          = beta_typepf
           clm(t)%vegwaterstresstype = veg_water_stress_typepf  ! none, pressure, sat
           clm(t)%rzwaterstress      = rz_water_stress_typepf   ! limit T by layer (1) or not (0, default)
           clm(t)%wilting_point      = wilting_pointpf
           clm(t)%field_capacity     = field_capacitypf
           clm(t)%res_sat            = res_satpf

           ! for irrigation
           clm(t)%irr_type           = irr_typepf
           clm(t)%irr_cycle          = irr_cyclepf
           clm(t)%irr_rate           = irr_ratepf
           clm(t)%irr_start          = irr_startpf
           clm(t)%irr_stop           = irr_stoppf
           clm(t)%irr_threshold      = irr_thresholdpf     
           clm(t)%threshold_type     = irr_thresholdtypepf
 
           ! set clm watsat, tksatu from PF porosity   @RMM moved this code up before clm_ini
           ! convert t to i,j index
 !          i=tile(t)%col
 !          j=tile(t)%row
!           do k = 1, nlevsoi ! loop over clm soil layers (1->nlevsoi)
!              ! convert clm space to parflow space, note that PF space has ghost nodes
!              l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
!              clm(t)%watsat(k)       = porosity(l)
!              clm(t)%tksatu(k)       = clm(t)%tkmg(k)*0.57**clm(t)%watsat(k)
!                print*,i,j
!              print*, 'k=',k,'watsat=',clm(t)%watsat(k),'porosity=',porosity(l),'pf_poro=',pf_porosity(k)
!           end do !k

        endif ! active/inactive

     end do !t

     !=== Read restart file or set initial conditions
     call drv_restart(1,drv,tile,clm,rank,istep_pf)        ! (1=read,2=write)

  endif !======= End of the initialization ================


  !=========================================================================
  !=== Time looping
  !=========================================================================

  !=== Call routine to copy PF variables to CLM space 
  !    (converts saturation to soil moisture)
  !    (converts pressure from m to mm)
  !    (converts soil moisture to mass of h2o)
  call pfreadout(clm,drv,tile,saturation,pressure,rank,ix,iy,nx,ny,nz,j_incr,k_incr,ip)

  !=== Advance time (CLM calendar time keeping routine)
  drv%endtime = 0
  call drv_tick(drv)

!RMM: writing a CLM.log.out file with basic information only from the master node (0 processor)
!
  if (rank==0) then
  write(9919,*)
  write(9919,*) "CLM starting time =", time, "gmt =", drv%gmt,"istep_pf =",istep_pf 
  write(9919,*) "CLM day =", drv%da, "month =", drv%mo,"year =", drv%yr
  end if ! CLM log

  
  !=== Read in the atmospheric forcing for off-line run
  !    (values no longer read by drv_getforce, passed from PF)
  !    (drv_getforce is modified to convert arrays from PF input to CLM space)
  !call drv_getforce(drv,tile,clm,nx,ny,sw_pf,lw_pf,prcp_pf,tas_pf,u_pf,v_pf,patm_pf,qatm_pf,istep_pf)
  !BH: modification of drv_getforc to optionally force vegetation (LAI/SAI/Z0M/DISPLA): 
  !BH: this replaces values from clm_dynvegpar called previously from drv_clmini and 
  !BH: replaces values from drv_readvegpf
  call drv_getforce(drv,tile,clm,nx,ny,sw_pf,lw_pf,prcp_pf,tas_pf,u_pf,v_pf, &
  patm_pf,qatm_pf,lai_pf,sai_pf,z0m_pf,displa_pf,istep_pf,clm_forc_veg)

  !=== Actual time loop
  !    (loop over CLM tile space, call 1D CLM at each point)
  do t = 1, drv%nch     
     clm(t)%qflx_infl_old       = clm(t)%qflx_infl
     clm(t)%qflx_tran_veg_old   = clm(t)%qflx_tran_veg
     if (clm(t)%planar_mask == 1) then
        call clm_main (clm(t),drv%day,drv%gmt,clm_forc_veg)
     else
     endif ! Planar mask
  enddo ! End of the space vector loop

  !=== Write CLM Output (timeseries model results)
  if (clm_1d_out == 1) then 
     call drv_1dout (drv, tile,clm,clm_write_logs)
  endif


  !=== Call 2D output routine
  !     Only call for clm_dump_interval steps (not time units, integer units)
  !     Only call if write_CLM_binary is True
  if (mod((istep_pf),clm_dump_interval)==0)  then
     if (write_CLM_binary==1) then

        ! Call subroutine to open (2D-) output files
        call open_files (clm,drv,rank,ix,iy,istep_pf,clm_output_dir,clm_output_dir_length,clm_bin_output_dir) 

        ! Call subroutine to write 2D output
        call drv_2dout  (drv,grid,clm)

        ! Call to subroutine to close (2D-) output files
        call close_files(clm,drv)

     end if ! write_CLM_binary
  end if ! mod of istep and dump_interval
  

  !=== Copy values from 2D CLM arrays to PF arrays for printing from PF (as Silo)
  do t=1,drv%nch
     i=tile(t)%col
     j=tile(t)%row
     l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2) 
     if (clm(t)%planar_mask==1) then
        eflx_lh_pf(l)      = clm(t)%eflx_lh_tot
        eflx_lwrad_pf(l)   = clm(t)%eflx_lwrad_out
        eflx_sh_pf(l)      = clm(t)%eflx_sh_tot
        eflx_grnd_pf(l)    = clm(t)%eflx_soil_grnd
        qflx_tot_pf(l)     = clm(t)%qflx_evap_tot
        qflx_grnd_pf(l)    = clm(t)%qflx_evap_grnd
        qflx_soi_pf(l)     = clm(t)%qflx_evap_soi
        qflx_eveg_pf(l)    = clm(t)%qflx_evap_veg 
        qflx_tveg_pf(l)    = clm(t)%qflx_tran_veg
        qflx_in_pf(l)      = clm(t)%qflx_infl 
        swe_pf(l)          = clm(t)%h2osno 
        t_g_pf(l)          = clm(t)%t_grnd
        qirr_pf(l)         = clm(t)%qflx_qirr
        irr_flag_pf(l)     = clm(t)%irr_flag
     else
        eflx_lh_pf(l)      = -9999.0
        eflx_lwrad_pf(l)   = -9999.0
        eflx_sh_pf(l)      = -9999.0
        eflx_grnd_pf(l)    = -9999.0
        qflx_tot_pf(l)     = -9999.0
        qflx_grnd_pf(l)    = -9999.0
        qflx_soi_pf(l)     = -9999.0
        qflx_eveg_pf(l)    = -9999.0
        qflx_tveg_pf(l)    = -9999.0
        qflx_in_pf(l)      = -9999.0
        swe_pf(l)          = -9999.0
        t_g_pf(l)          = -9999.0
        qirr_pf(l)         = -9999.0
        irr_flag_pf(l)     = -9999.0
     endif
  enddo


  !=== Repeat for values from 3D CLM arrays
  do t=1,drv%nch            ! Loop over CLM tile space
     i=tile(t)%col
     j=tile(t)%row
     if (clm(t)%planar_mask==1) then
        do k = 1,nlevsoi       ! Loop from 1 -> number of soil layers (in CLM)
           l = 1+i + j_incr*(j) + k_incr*(nlevsoi-(k-1))
           t_soi_pf(l)     = clm(t)%t_soisno(k)
           qirr_inst_pf(l) = clm(t)%qflx_qirr_inst(k)
        enddo
     else
        do k = 1,nlevsoi
           l = 1+i + j_incr*(j) + k_incr*(nlevsoi-(k-1))
           t_soi_pf(l)     = -9999.0
           qirr_inst_pf(l) = -9999.0
        enddo
     endif
  enddo



  !=== Write Daily Restarts
  if (clm_write_logs==1) then
  write(999,*) "End of time advance:" 
  write(999,*) 'time =', time, 'gmt =', drv%gmt, 'endtime =', drv%endtime
  endif
 if (rank==0) then
    write(9919,*) "End of time advance:"
    write(9919,*) 'time =', time, 'gmt =', drv%gmt, 'endtime =', drv%endtime
 end if !! rank 0, write log info

  ! if ( (drv%gmt==0.0).or.(drv%endtime==1) ) call drv_restart(2,drv,tile,clm,rank,istep_pf)
  ! ----------------------------------
  ! NBE: Added more control over writing of the RST files
    if (clm_last_rst==1) then
       d_stp=0
    else
       d_stp = istep_pf
    endif
    
    if (clm_daily_rst==1) then

       ! Restarts occur at daily boundaries and at end of the run.
       if ( (drv%gmt==0.0).or.(drv%endtime==1) ) then

          !! @RMM/LEC  add in a TCL file that sets an istep value to better automate restarts
          if (rank==0) then
             write(9919,*) 'Writing restart time =', time, 'gmt =', drv%gmt, 'istep_pf =',istep_pf

             open(393, file="clm_restart.tcl",action="write")
             write(393,*) "set istep ",istep_pf
             close(393)
          end if  !  write istep corresponding to restart step
             
          call drv_restart(2,drv,tile,clm,rank,d_stp)

       end if
    else
       ! Restarts occur at start of each CLM reuse sequence.
       if (clm_next == 1) then

          !! @RMM/LEC  add in a TCL file that sets an istep value to better automate restarts
          if (rank==0) then
             write(9919,*) 'Writing restart time =', time, 'gmt =', drv%gmt, 'istep_pf =',istep_pf

             open(393, file="clm_restart.tcl",action="write")
             write(393,*) "set istep ",istep_pf
             close(393)
          end if  !  write istep corresponding to restart step
             
          call drv_restart(2,drv,tile,clm,rank,d_stp)

       end if

    end if
  ! ---------------------------------

  !=== Call routine to calculate CLM flux passed to PF
  !    (i.e., routine that couples CLM and PF)
  call pf_couple(drv,clm,tile,evap_trans,saturation,pressure,porosity,nx,ny,nz,j_incr,k_incr,ip,d_stp)


  !=== LEGACY ===========================================================================================
  !    (no longer needed because current setup restricts one tile per grid cell) 
  !=== Return required surface fields to atmospheric model (return to grid space)
  !    (accumulates tile fluxes over grid space)
  ! call drv_clm2g (drv, grid, tile, clm)


  !=== Write spatially-averaged BC's and IC's to file for user
  if (clm_write_logs==1) then ! NBE
  if (istep_pf==1) call drv_pout(drv,tile,clm,rank)
  endif

  !=== If at end of simulation, close all files
  if (drv%endtime==1) then
     ! close(166)
     ! close(199)
     if (clm_write_logs==1) close(999)
     if (rank == 0) close (9919)
  end if



end subroutine clm_lsm
