!-----------------------------------------------------------------
! 
! This module includes subroutines for communicating with eCLM 
! via the OASIS3-MCT API.
!
!-----------------------------------------------------------------
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
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  bind(c, name="oas_pfl_vardef_mp_localcomm_")   :: localComm
#else
  bind(c, name="__oas_pfl_vardef_MOD_localcomm") :: localComm
#endif

  ! Local variables used by OASIS
  integer :: comp_id, rank, ierror
  integer :: soilliq_id, psi_id, et_id ! coupling field IDs

  ! TODO: Get these values from eCLM instead of hardcoding it here
  integer :: nlevsoi  = 20 ! Number of hydrologically active soil layers in eCLM
  integer :: nlevgrnd = 25 ! Total number of soil layers in eCLM

contains

  subroutine oas_pfl_init(argc) bind(c, name='oas_pfl_init_')
    !-------------------------------------
    ! Initializes the OASIS3-MCT coupler.
    !-------------------------------------
    integer, intent(inout) ::  argc  ! unused variable
    
    call oasis_init_comp(comp_id, "oaspfl", ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_init', 'oasis_init_comp failed.')
  
    call oasis_get_localcomm(localComm, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_init', 'oasis_get_localcomm failed.')

    call MPI_Comm_Rank(localComm, rank, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_init', 'MPI_Comm_Rank failed.') 
  end subroutine oas_pfl_init

  subroutine oas_pfl_finalize(argc) bind(c, name='oas_pfl_finalize_')
    !---------------------------------------------------------------------
    ! Terminates the OASIS3-MCT coupler and closes the MPI communication.
    !---------------------------------------------------------------------
    integer, intent(inout) ::  argc  ! unused variable
    
    call oasis_terminate(ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_finalize', 'oasis_terminate failed.')
  end subroutine oas_pfl_finalize

  subroutine oas_pfl_define(nx, ny, pdx, pdy, ix, iy, sw_lon, sw_lat, nlon, nlat, &
                            pfl_step, pfl_stop) bind(c, name='oas_pfl_define_')
    !-----------------------------------------------------------------------
    ! Defines grid, partition, and coupled variables for OASIS3-MCT coupler
    !-----------------------------------------------------------------------
    integer,       intent(in) :: nx,ny,             & ! subgrid dimensions
                                 ix,iy,             & ! starting coordinate of the subgrid
                                 nlon, nlat           ! Size of entire parflow grid

    real (kind=8), intent(in) :: sw_lon,sw_lat,     & !  lat & lon of southwest corner (UNUSED)
                                 pdx,pdy,           & !  DX, DY (UNUSED)
                                 pfl_step, pfl_stop   !  parflow time step and stop time in hours (UNUSED)

    ! Local variables
    integer  :: part_id       ! partition id returned by oasis_def_partition
    integer  :: il_paral(5)   ! partition descriptor (input to oasis_def_partition)
    integer  :: var_nodims(2) ! array dimensions of the coupling field (input to oasis_def_var)

    ! 1) Define grid (not necessary*)
    !    * This step is not necessary since coupling fields don't have to be spatially
    !      interpolated between ParFlow and eCLM (i.e. ParFlow and eCLM grids correspond 1-to-1)
    ! if (rank == 0) then
    !   call oasis_start_grids_writing(write_grid_files)
    !   if (write_grid_files == 1) then
    !     call oasis_write_grid(grid_name, nlon, nlat, lglon, lglat)
    !     call oasis_write_corner(grid_name, nlon, nlat, 4, lclon, lclat)
    !     call oasis_write_mask(grid_name, nlon, nlat, mask_land)
    !     call oasis_terminate_grids_writing()
    !   end if
    ! end if

    ! 2) Define partition
    il_paral(1) = 2            ! Box partition
    il_paral(2) = ix + iy*nlon ! Upper left corner global offset
    il_paral(3) = nx           ! local extent in X
    il_paral(4) = ny           ! local extent in Y
    il_paral(5) = nlon         ! Global extent in X
    call oasis_def_partition (part_id, il_paral, ierror)
    if (ierror /= 0) call oasis_abort(comp_id, 'oas_pfl_define', 'oasis_def_partition failed.')
    
    ! 3) Define coupling fields
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
  end subroutine oas_pfl_define

  subroutine send_fld2_clm(pressure, saturation, topo, ix, iy, nx, ny, nz, nx_f, ny_f, pstep, porosity, dz) bind(c, name='send_fld2_clm_')
    !----------------------------------------------------------------------------
    ! Sends pressure head and soil liquid water from ParFlow to eCLM
    ! pressure is converted from [m] to [mm]
    ! soil liquid is calculated as: h2osoi_liq = saturation*porosity*dz*1000 
    !----------------------------------------------------------------------------
    integer,      intent(in)  :: ix, iy,                           & ! starting coordinate of the subgrid (UNUSED)
                                 nx, ny, nz,                       & ! subgrid dimensions
                                 nx_f, ny_f                          ! dimensions of ET subvector
    real(kind=8), intent(in)  :: pstep,                            & ! current model time in hours
                                 pressure((nx+2)*(ny+2)*(nz+2)),   & ! pressure head [m]
                                 saturation((nx+2)*(ny+2)*(nz+2)), & ! saturation [-]
                                 topo((nx+2)*(ny+2)*(nz+2)),       & ! topography mask (0 for inactive, 1 for active)
                                 porosity((nx+2)*(ny+2)*(nz+2)),   & ! porosity [m^3/m^3]
                                 dz((nx+2)*(ny+2)*(nz+2))            ! subsurface layer thickness [m]
                                                                     ! (nx+2)*(ny+2)*(nz+2) = total number of subgrid cells; the
                                                                     !  extra "+2" terms account for the ghost nodes/halo points
    ! Local variables
    integer                   :: seconds_elapsed                     ! current model time in seconds
    integer                   :: i, j, k                             ! index variables for subgrid dimensions (nx, ny, nz)
    integer                   :: l                                   ! index variable for ParFlow fields (e.g. pressure, saturation, etc.)
    integer                   :: z                                   ! subsurface level (z=nz topmost layer, z=1 deepest layer)
    integer                   :: top_z_level(nx,ny)                  ! topmost z level of active ParFlow cells
    real(kind=8), allocatable :: pressure_3d(:,:,:),               & ! pressure head sent to eCLM   [mm]
                                 h2osoi_liq_3d(:,:,:)                ! h2o soil liquid sent to eCLM [mm]

    allocate(pressure_3d(nx,ny,nlevgrnd))                             
    allocate(h2osoi_liq_3d(nx,ny,nlevgrnd))
    pressure_3d = 0.0
    h2osoi_liq_3d = 0.0
    top_z_level = get_top_z_level(nx, ny, nz, topo)

    ! Convert ParFlow fields to 3d array
    do i = 1, nx
      do j = 1, ny
        if (top_z_level(i,j) > 0) then    ! ***** Subsurface level indexing convention *****
          do k = 1, nlevgrnd              !    eCLM: 1=topmost layer, nlevgrnd=deepest layer
            z = top_z_level(i,j) - (k-1)  ! ParFlow: 1=deepest layer, nz=topmost layer
            l = flattened_array_index(i, j, z, nx_f, ny_f)
            pressure_3d(i,j,k) = pressure(l)*1000.0                       ! multiply these quantities by 1000
            h2osoi_liq_3d(i,j,k) = saturation(l)*porosity(l)*dz(l)*1000   ! to convert from [m] to [mm]
          end do
        end if
      end do
    end do

    ! Send ParFlow fields to eCLM
    seconds_elapsed = nint(pstep*3600.d0)
    call oasis_put(soilliq_id, seconds_elapsed, h2osoi_liq_3d, ierror)
    call oasis_put(psi_id, seconds_elapsed, pressure_3d, ierror)

    deallocate(h2osoi_liq_3d)
    deallocate(pressure_3d)
  end subroutine send_fld2_clm

  subroutine receive_fld2_clm(evap_trans, topo, ix, iy, nx, ny, nz, nx_f, ny_f, pstep) bind(c, name='receive_fld2_clm_')
    !-----------------------------------------------
    ! Receives evapotranspiration fluxes from eCLM.
    !-----------------------------------------------
    integer,      intent(in)    :: ix, iy,                        & ! starting coordinate of the subgrid (UNUSED)
                                   nx, ny, nz,                    & ! subgrid dimensions
                                   nx_f, ny_f                       ! dimensions of ET subvector
    real(kind=8), intent(in)    :: pstep,                         & ! current model time in hours
                                   topo((nx+2)*(ny+2)*(nz+2))       ! topography mask (0 for inactive, 1 for active)
    real(kind=8), intent(inout) :: evap_trans((nx+2)*(ny+2)*(nz+2)) ! evapotranspiration [1/hrs]
                                                                    ! (nx+2)*(ny+2)*(nz+2) = total number of subgrid cells; the
                                                                    !  extra "+2" terms account for the ghost nodes/halo points

    ! Local variables
    integer                     :: seconds_elapsed                  ! current model time in seconds
    integer                     :: i, j, k                          ! index variables for subgrid dimensions (nx, ny, nz)
    integer                     :: l                                ! index variable for ParFlow fields (e.g. evap_trans, topo)
    integer                     :: z                                ! subsurface level (z=nz topmost layer, z=1 deepest layer)
    integer                     :: top_z_level(nx,ny)               ! topmost z level of active ParFlow cells
    real(kind=8), allocatable   :: evap_trans_3d(:,:,:)             ! Root ET fluxes received from eCLM [1/hrs]

    ! Receive ET fluxes from eCLM
    allocate(evap_trans_3d(nx,ny,nlevsoi))
    seconds_elapsed = nint(pstep*3600.d0)
    call oasis_get(et_id, seconds_elapsed, evap_trans_3d, ierror)

    ! Save ET fluxes to ParFlow evap_trans vector
    evap_trans = 0.
    top_z_level = get_top_z_level(nx, ny, nz, topo)
    do i = 1, nx
      do j = 1, ny
        if (top_z_level(i,j) > 0) then    ! ***** Subsurface level indexing convention *****
          do k = 1, nlevsoi               !    eCLM: 1=topmost layer, nlevsoi=deepest layer
            z = top_z_level(i,j) - (k-1)  ! ParFlow: 1=deepest layer, nz=topmost layer
            l = flattened_array_index(i, j, z, nx_f, ny_f)
            evap_trans(l) = evap_trans_3d(i,j,k)
          end do
        end if
      end do
    end do

    deallocate(evap_trans_3d)
  end subroutine receive_fld2_clm

  function get_top_z_level(nx, ny, nz, topo)
    !-------------------------------------------------
    ! Get the topmost z level for active cells. 
    ! z = 1 points to the deepest subsurface layer; 
    ! thus a higher z means a layer closer to surface.
    ! For inactive cells, get_top_z_level(nx,ny) = 0.
    !-------------------------------------------------
    integer,      intent(in) :: nx, ny, nz                 ! number of elements along x, y, and z
    real(kind=8), intent(in) :: topo((nx+2)*(ny+2)*(nz+2)) ! topography mask (0 for inactive, 1 for active)

    integer :: i, j, k, l
    integer :: get_top_z_level(nx,ny)
 
    get_top_z_level = 0
    do i = 1, nx
      do j = 1, ny
        find_active_z: do k = nz, 1, -1
            l = flattened_array_index(i, j, k, nx+2, ny+2)
            if (topo(l) > 0) then
              get_top_z_level(i,j) = k
              exit find_active_z
            end if
        end do find_active_z
      end do
    end do
  end function get_top_z_level

  integer function flattened_array_index(i, j, k, nx, ny)
    !----------------------------------------------------
    ! Computes the 1d array index that corresponds with 
    ! the given 3d array indices (i,j,k)
    !----------------------------------------------------
    integer, intent(in) :: i, j, k   ! indices for nx, ny, and nz
    integer, intent(in) :: nx, ny    ! subgrid dimensions

    flattened_array_index = (i+1) + (j*nx) + (k*nx*ny)
  end function flattened_array_index

end module oas_pfl_mod