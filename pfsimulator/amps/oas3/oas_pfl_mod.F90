module oas_pfl_mod
  use iso_c_binding
  use mod_oasis
  implicit none
  save
  private

  ! Public members
  public :: oas_pfl_init
  public :: oas_pfl_define
  public :: oas_pfl_finalize
  public :: send_fld2_clm
  public :: receive_fld2_clm

  ! Local comm assigned by OASIS3-MCT to ParFlow 
  integer(c_int), public :: localComm 

  ! Hack to conform with localComm usage in oas3/amps_init.c
#if defined(__INTEL_COMPILER)
  bind(c, name="oas_pfl_vardef_mp_localcomm_")   :: localComm
#else
  bind(c, name="__oas_pfl_vardef_MOD_localcomm") :: localComm
#endif

  ! Local variables
  integer :: comp_id, rank, ierror
  integer :: nlevsoi = 20           ! Number of soil layers in CLM
  integer :: nlevgrnd = 25          ! Number of soil layers in CLM
  integer :: soilliq_id, psi_id, et_id

contains

  subroutine oas_pfl_init(argc) bind(c, name='oas_pfl_init_')
    integer ::  argc  ! unused variable
    
    call oasis_init_comp(comp_id, "oaspfl", ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_init', 'oasis_init_comp failed.')
  
    call oasis_get_localcomm(localComm, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_init', 'oasis_get_localcomm failed.')

    call MPI_Comm_Rank(localComm, rank, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_init', 'MPI_Comm_Rank failed.') 
  end subroutine oas_pfl_init

  subroutine oas_pfl_finalize(argc) bind(c, name='oas_pfl_finalize_')
    integer ::  argc  ! unused variable
    
    call oasis_terminate(ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_finalize', 'oasis_terminate failed.')
  end subroutine oas_pfl_finalize

  subroutine oas_pfl_define(nx, ny, pdx, pdy, ix, iy, sw_lon, sw_lat, nlon, nlat, &
                            pfl_step, pfl_stop) bind(c, name='oas_pfl_define_')
    ! Input args
    integer,       intent(in) :: nx,ny,             & ! parflow, total local grid points, nz=1 for oasis3  
                                 ix,iy,             & ! parflow, starting point for local grid on global grid
                                 nlon, nlat           ! Size of entire parflow grid

    real (kind=8), intent(in) :: sw_lon,sw_lat,     & !  parflow, SW corner in parflow units     
                                 pdx,pdy,           & !  parflow DX, DY and DZ in parflow units
                                 pfl_step, pfl_stop   !  parflow time step, and stop time in hours

    ! Local variables
    integer  :: part_id       ! id returned by oasis_def_partition
    integer  :: il_paral(5)
    integer  :: var_nodims(2)
    integer  :: write_grid_files

    ! TODO: Write Parflow grid parameters to grid file.
    ! if (rank == 0) then
    !   call oasis_start_grids_writing(write_grid_files)
    !   if (write_grid_files == 1) then
    !     call oasis_write_grid(grid_name, nlon, nlat, lglon, lglat)
    !     call oasis_write_corner(grid_name, nlon, nlat, 4, lclon, lclat)
    !     call oasis_write_mask(grid_name, nlon, nlat, mask_land)
    !     call oasis_terminate_grids_writing()
    !   end if
    ! end if

    ! -----------------------------------------------------------------
    ! ... Define partition
    ! -----------------------------------------------------------------
    il_paral(1) = 2            ! Box partition
    il_paral(2) = ix + iy*nlon ! Upper left corner global offset
    il_paral(3) = nx           ! local extent in X
    il_paral(4) = ny           ! local extent in Y
    il_paral(5) = nlon         ! Global extent in X
    call oasis_def_partition (part_id, il_paral, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_define', 'Failed oasis_def_partition')
    
    ! -----------------------------------------------------------------
    ! ... Define coupling fields
    ! -----------------------------------------------------------------
    var_nodims(1) = 1
    var_nodims(2) = nlevsoi 

    call oasis_def_var (et_id, "PFL_ET", part_id, var_nodims, OASIS_In, OASIS_Real, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_define', 'oasis_def_var failed for PFL_ET')

    var_nodims(2) = nlevgrnd
    call oasis_def_var (psi_id, "PFL_PSI", part_id, var_nodims, OASIS_Out, OASIS_Real, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_define', 'oasis_def_var failed for PFL_PSI')
    call oasis_def_var (soilliq_id, "PFL_SOILLIQ", part_id, var_nodims, OASIS_Out, OASIS_Real, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_define', 'oasis_def_var failed for PFL_SOILLIQ')

    call oasis_enddef ( ierror )
    if (ierror /= 0) call oasis_abort (comp_id, 'oas_pfl_define', 'oasis_enddef failed')
    write(6,*) 'oaspfl: - oas_pfl_define : variable definition complete'

  end subroutine oas_pfl_define

  subroutine send_fld2_clm(pressure, saturation, topo, ix, iy, nx, ny, nz, nx_f, ny_f, pstep, porosity, dz) bind(c, name='send_fld2_clm_')
    ! Input args
    integer,      intent(in)  :: nx, ny, nz,                       & ! Subgrid
                                 nx_f, ny_f,                       & !         
                                 ix,iy                               ! Need to write debug netcdf files
    real(kind=8), intent(in)  :: pstep,                            & ! Parflow model time-step in hours
                                 pressure((nx+2)*(ny+2)*(nz+2)),   & ! pressure head (m)
                                 saturation((nx+2)*(ny+2)*(nz+2)), & ! saturation    (-)
                                 topo((nx+2)*(ny+2)*(nz+2)),       & ! mask    (0 for inactive, 1 for active)
                                 porosity((nx+2)*(ny+2)*(nz+2)),   & ! pressure head (m)
                                 dz((nx+2)*(ny+2)*(nz+2))

    ! Local variables                            
    integer                   :: i, j, k, l, g
    integer                   :: isecs                               ! Parflow model time in seconds
    integer                   :: j_incr, k_incr                      ! convert 1D vector to 3D i,j,k array
    integer, allocatable      :: counter(:,:), topo_mask(:,:)        ! Mask for active parflow cells
    real(kind=8), allocatable :: h2osoi_liq_snd(:,:,:) , psi_snd(:,:,:)     ! temporary array

    isecs= nint(pstep*3600.d0)
    j_incr = nx_f
    k_incr = nx_f*ny_f

    allocate(h2osoi_liq_snd(nx,ny,nlevgrnd))
    allocate(psi_snd(nx,ny,nlevgrnd))
    allocate(topo_mask(nx,ny))
    allocate(counter(nx,ny))

    ! Create the masking vector
    topo_mask = 0 
    do i = 1, nx
      do j = 1, ny
        counter(i,j) = 0
        do k = nz, 1, -1                                             ! PF loop over z
            l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2)*(k)
            if (topo(l) > 0) then
              counter(i,j) = counter(i,j) + 1
              if (counter(i,j) == 1) topo_mask(i,j) = k
            end if
        end do
      end do
    end do

    h2osoi_liq_snd = 0
    psi_snd = 0
    do i = 1, nx
      do j = 1, ny
        do k = 1, nlevgrnd
          if (topo_mask(i,j) > 0) then
            l = 1+i + j_incr*(j) + k_incr*(topo_mask(i,j)-(k-1))     !
            ! h2osoi_liq_snd(g,k) = saturation(l)
            h2osoi_liq_snd(i,j,k) = saturation(l)*porosity(l)*dz(l)*1000
            psi_snd(i,j,k) = pressure(l)*1000.0     ! convert from [m] to [mm]
          end if
        end do
      end do
    end do

    !Send the fields

    call oasis_put(soilliq_id, isecs, h2osoi_liq_snd, ierror)
    call oasis_put(psi_id, isecs, psi_snd, ierror)

    deallocate(h2osoi_liq_snd)
    deallocate(psi_snd)
    deallocate(counter)
    deallocate(topo_mask)
  end subroutine send_fld2_clm

  subroutine receive_fld2_clm(evap_trans, topo, ix, iy, nx, ny, nz, nx_f, ny_f, pstep) bind(c, name='receive_fld2_clm_')
    ! Input args
    integer,      intent(in)    :: ix, iy,                          & !
                                   nx, ny, nz,                      & ! Subgrid
                                   nx_f, ny_f
    real(kind=8), intent(in)    :: pstep                              ! Parflow model time-step in hours
    real(kind=8), intent(in)    :: topo((nx+2)*(ny+2)*(nz+2))         ! mask (0 for inactive, 1 for active)
    real(kind=8), intent(inout) :: evap_trans((nx+2)*(ny+2)*(nz+2))   ! source/sink (1/T)

    ! Local variables
    integer                     :: i, j, k, l, g
    integer                     :: isecs                              ! Parflow model time in seconds
    integer                     :: j_incr, k_incr                     ! convert 1D vector to 3D i,j,k array
    integer, allocatable        :: counter(:,:),                    & !
                                   topo_mask(:,:)                     ! Mask for active parflow cells
    real(kind=8), allocatable   :: et_rcv(:,:,:)                      ! ET fluxes from eCLM

    isecs= nint(pstep*3600.d0)
    j_incr = nx_f
    k_incr = nx_f*ny_f

    allocate(topo_mask(nx,ny))
    allocate(counter(nx,ny))
    allocate(et_rcv(nx,ny,nlevsoi))

    topo_mask = 0
    ! Create the masking vector
    do i = 1, nx
      do j = 1, ny
        counter(i,j) = 0
        do k = nz, 1, -1                                                  ! PF loop over z
            l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2)*(k)
            if (topo(l) > 0) THEN
              counter(i,j)=counter(i,j)+1
              if (counter(i,j) .eq. 1) topo_mask(i,j) = k 
            end if 
        end do
      end do
    end do
    
    call oasis_get(et_id, isecs, et_rcv, ierror)

    evap_trans = 0.
    do i = 1, nx
      do j = 1, ny
        do k = 1, nlevsoi 
          if (topo_mask(i,j) > 0) then
            !g = (i-1)*ny + j
            l = 1+i + j_incr*(j) + k_incr*(topo_mask(i,j)-(k-1))
            evap_trans(l) = et_rcv(i,j,k)!et_rcv(g,k)
          end if
        end do
      end do
    end do

    deallocate(et_rcv)
    deallocate(counter)
    deallocate(topo_mask)
  end subroutine receive_fld2_clm
end module oas_pfl_mod