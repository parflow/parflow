SUBROUTINE oas_pfl_define(nx,ny,pdx,pdy,ix,iy, & 
           sw_lon,sw_lat,nlon,nlat,pfl_step,pfl_stop)

!----------------------------------------------------------------------------
!
! Description
! Define coupled variables for OASIS3 coupler 
!
!
! Method:
! Current Code Owner: TR32, Z4, Prabhakar Shrestha
!  phone: +49 (0)228 73 3152
! e-mail: pshrestha@uni-bonn.de
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.00       2011/10/19 Prabhakar Shrestha
! 2.00       2013/09/27 Prabhakar Shrestha
! Added read of LANDMASK from CLM fracdata.nc,
! Usage of prism libraries
! prism_start_grids_writing, prism_write_grid, prism_write_corner, prism_write_mask
! prism_terminate_grids_writing, prism_def_partition_proto, 
! prism_def_var_proto, prism_enddef_proto, prism_abort_proto 
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!----------------------------------------------------------------------------- 

! Declarations:

! Modules used:
USE oas_pfl_vardef


USE netcdf

!==============================================================================

IMPLICIT NONE

!==============================================================================

!
! values passed from parflow
INTEGER , INTENT(IN)                       ::                   &
    nx,ny,                                                      & ! parflow, total local grid points, nz=1 for oasis3  
    ix,iy,                                                      & ! parflow, starting point for local grid on global grid
    nlon, nlat                                                    ! Size of entire parflow grid

REAL (KIND=8), INTENT(IN)                  ::                   &
    sw_lon,sw_lat,                                              & !  parflow, SW corner in parflow units
    pdx,pdy,                                                    & !  parflow DX, DY and DZ in parflow units
    pfl_step, pfl_stop                                            !  parflow time step, and stop time in hours

! Defined in this routine
CHARACTER(len=4)                           ::  clgrd
INTEGER                                    ::  il_flag            ! Flag for grid writing by proc 0
INTEGER                                    ::  var_nodims(2)      ! used in prism_def_var_proto
INTEGER                                    ::  vshape(4)          ! Shape of array passed to PSMILe 
                                                                  ! 2 x field rank (= 4 because fields are of rank = 2)
INTEGER                                    ::  dim_paral          ! Type of partition
INTEGER, POINTER                           ::  il_paral(:)        ! Define process partition 
!
INTEGER                                    ::  part_id            ! ID returned by prism_def_partition_proto 
INTEGER                                    ::  ii, jj, nn         ! Local Variables

REAL(KIND=8), ALLOCATABLE                  :: lglon(:,:),        &! 
                                              lglat(:,:)          ! Global Grid Centres 
REAL(KIND=8), ALLOCATABLE                  :: lclon(:,:,:),      &!
                                              lclat(:,:,:)        ! Global Grid Corners 

REAL                                       ::  dlat, dlon
!CPS PARFLOW HAS NO GRID INFORMATION
!CPS #ifdef READCLM
INTEGER                         :: readclm = 1         ! 1 or 0 to read clm mask
!CPS #else
!INTEGER                         :: readclm = 0         ! 1 or 0 to read clm mask
!CPS #endif

REAL(KIND=8), ALLOCATABLE                  ::  clmlon(:,:),        &! 
                                               clmlat(:,:)          ! Global Grid Centres
INTEGER                                    ::  status, pflncid,  &!
                                               pflvarid(3),      &! CPS increased to 3,Debug netcdf output
                                               ib, npes
!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!- Begin Subroutine oas_pfl_define 
!------------------------------------------------------------------------------
!
! Read in land mask for each sub-domain to mask recv values from CLM
ALLOCATE( mask_land_sub(nx,ny), stat = ierror )
IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating mask_land_sub')
CALL MPI_Comm_size(localComm, npes, ierror)
DO ib = 0,npes-1
  IF (rank == ib ) THEN
   status = nf90_open("clmgrid.nc", NF90_NOWRITE, pflncid)
   status = nf90_inq_varid(pflncid, "LANDMASK" , pflvarid(3))
   status = nf90_get_var(pflncid, pflvarid(3), mask_land_sub, &
                         start = (/ix+1, iy+1/), &
                         count = (/nx, ny/) )
   status = nf90_close(pflncid)
   mask_land_sub = mask_land_sub
  ENDIF
ENDDO


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 !Global grid definition for OASIS3, written by master process for 
 !the component, i.e rank = 0 
 !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 IF (rank .EQ. 0) THEN                                  !if rank = 0

 pfl_timestep = pfl_step
 pfl_stoptime = pfl_stop
!
 nx_tot = nlat                                          !Assign total parflow domain
 ny_tot = nlon                                          !Assign total parflow domain 

 ALLOCATE( lglon(nlon,nlat), stat = ierror )
 IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating lglon')
 ALLOCATE( lglat(nlon,nlat), stat = ierror )
 IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating lglat')

 ALLOCATE( lclon(nlon,nlat, 4), stat = ierror )
 IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating lclon')
 ALLOCATE( lclat(nlon,nlat, 4), stat = ierror )
 IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating lclat')

 ALLOCATE( mask_land(nlon,nlat), stat = ierror )
 IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating mask_land')

 ! Assumes lats are constant on an i line
 ! Assumes lons are constant on an j line

 !Added option to read clm lat-lon CPS

 IF (readclm == 1) THEN
   ALLOCATE( clmlon(nlon,nlat), stat = ierror )
   IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating clmlon')
   ALLOCATE( clmlat(nlon,nlat), stat = ierror )
   IF (ierror >0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating clmlat')

   status = nf90_open("clmgrid.nc", NF90_NOWRITE, pflncid)
   status = nf90_inq_varid(pflncid, "LONGXY" , pflvarid(1))
   status = nf90_inq_varid(pflncid, "LATIXY" , pflvarid(2))
   status = nf90_inq_varid(pflncid, "LANDMASK" , pflvarid(3))
   status = nf90_get_var(pflncid, pflvarid(1), clmlon) 
   status = nf90_get_var(pflncid, pflvarid(2), clmlat)
   status = nf90_get_var(pflncid, pflvarid(3), mask_land)
   status = nf90_close(pflncid)

   ! Define centers
   DO ii = 1, nlon
   DO jj = 1, nlat
    lglon(ii,jj) = clmlon(ii,jj)
    lglat(ii,jj) = clmlat(ii,jj) 
   END DO
   END DO

!CPS assuming regular grids
 dlon = ABS(lglon(2,1) - lglon(1,1))
 dlat = ABS(lglat(1,2) - lglat(1,1))

 ELSE IF (readclm == 0) THEN

   dlat = pdx/100689.655172d0      !Convert from PARFLOW units in "m" to degrees
   dlon = pdy/60998.5377063d0      !Convert from PARFLOW units in "m" to degrees

   dlat = 0.00898311175 !0.009375
   dlon = 0.00898311175 !0.009375

   ! Define centers
   DO ii = 1, nlon
    lglon(ii,:) = sw_lon + (ii-1)*dlon
   END DO
   DO jj = 1, nlat
    lglat(:,jj) = sw_lat + (jj-1)*dlat
   END DO

 END IF                   !readclm

 ! Define the 4 corners of the variables on the grid; 

 DO ii= 1, nlon
 DO jj= 1, nlat
 lclon(ii,jj,1)    = lglon(ii,jj) + dlon*0.5
 lclon(ii,jj,2)    = lglon(ii,jj) + dlon*0.5
 lclon(ii,jj,3)    = lglon(ii,jj) - dlon*0.5
 lclon(ii,jj,4)    = lglon(ii,jj) - dlon*0.5

 lclat(ii,jj,1)    = lglat(ii,jj) - dlat*0.5
 lclat(ii,jj,2)    = lglat(ii,jj) + dlat*0.5
 lclat(ii,jj,3)    = lglat(ii,jj) + dlat*0.5
 lclat(ii,jj,4)    = lglat(ii,jj) - dlat*0.5
 END DO
 END DO
!
 clgrd = 'gpfl'

!!Create the mask
 mask_land = ABS(mask_land - 1)
!CPS read from clm fracdata.nc file  mask_land = 0          !opposite convention to OASIS4

  !
 CALL prism_start_grids_writing(il_flag)
 CALL prism_write_grid(clgrd, nlon, nlat, lglon, lglat)
 CALL prism_write_corner(clgrd, nlon, nlat, 4, lclon, lclat)
 CALL prism_write_mask(clgrd, nlon, nlat, mask_land)
 CALL prism_terminate_grids_writing()

 ENDIF                       !if rank ==0

! CALL MPI_Barrier(localcomm, ierror)
 !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 !Define the partition 
 !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
 !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 ! Define the partition, need to DEBUGGGG THIS 
 !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 dim_paral = 5           !Use Box Partition always

 ALLOCATE(il_paral(dim_paral), stat=ierror )
 IF ( ierror /= 0 ) CALL prism_abort_proto(comp_id, 'oas3_pfl_define', 'Error allocating il_paral')

 il_paral(1) = 2                                    ! Box partition
 il_paral(5) = nlon                                 ! Global extent in X
 il_paral(2) = ix + iy*il_paral(5)                  ! Upper left corner global offset
 il_paral(3) = nx                                   ! local extent in X
 il_paral(4) = ny                                   ! local extent in Y 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Define the shape of valid region w/o any halo between cpus
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 vshape(1)    = 1
 vshape(2)    = nx
 vshape(3)    = 1
 vshape(4)    = ny

 CALL prism_def_partition_proto ( part_id, il_paral, ierror )
 IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'model1', 'Failure in prism_def_partition_proto')


 !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 !Variable definition
 !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 var_nodims(1) = 2    ! Rank of the field array is 2
 var_nodims(2) = 1    ! Bundles always 1 for OASIS3
! 
 psnd(:)%laction=.FALSE.  ; wsnd(:)%laction=.FALSE.  ; trcv(:)%laction=.FALSE.
 psnd(:)%clgrid=clgrd     ; wsnd(:)%clgrid=clgrd     ; trcv(:)%clgrid=clgrd

! saturation 10 level
 wsnd(1)%clpname  = 'PFLSAT01' 
 wsnd(2)%clpname  = 'PFLSAT02'
 wsnd(3)%clpname  = 'PFLSAT03'
 wsnd(4)%clpname  = 'PFLSAT04'
 wsnd(5)%clpname  = 'PFLSAT05'
 wsnd(6)%clpname  = 'PFLSAT06'
 wsnd(7)%clpname  = 'PFLSAT07'
 wsnd(8)%clpname  = 'PFLSAT08'
 wsnd(9)%clpname  = 'PFLSAT09'
 wsnd(10)%clpname = 'PFLSAT10'

! pressure 10 level
 psnd(1)%clpname  = 'PFLPSI01' 
 psnd(2)%clpname  = 'PFLPSI02'
 psnd(3)%clpname  = 'PFLPSI03'
 psnd(4)%clpname  = 'PFLPSI04'
 psnd(5)%clpname  = 'PFLPSI05'
 psnd(6)%clpname  = 'PFLPSI06'
 psnd(7)%clpname  = 'PFLPSI07'
 psnd(8)%clpname  = 'PFLPSI08'
 psnd(9)%clpname  = 'PFLPSI09'
 psnd(10)%clpname = 'PFLPSI10'

! evapotranspiration flux 10 level
 trcv(1)%clpname  = 'PFLFLX01'
 trcv(2)%clpname  = 'PFLFLX02'
 trcv(3)%clpname  = 'PFLFLX03'
 trcv(4)%clpname  = 'PFLFLX04'
 trcv(5)%clpname  = 'PFLFLX05' 
 trcv(6)%clpname  = 'PFLFLX06'
 trcv(7)%clpname  = 'PFLFLX07'
 trcv(8)%clpname  = 'PFLFLX08'
 trcv(9)%clpname  = 'PFLFLX09'
 trcv(10)%clpname = 'PFLFLX10' 

! Variable Selection
 wsnd(1:nlevsoil)%laction=.TRUE.
 psnd(1:nlevsoil)%laction=.TRUE.
 trcv(1:nlevsoil)%laction=.TRUE.

! Define send variables
 DO nn = 1, nmaxlev
   IF ( wsnd(nn)%laction ) THEN
     CALL prism_def_var_proto ( wsnd(nn)%vid, wsnd(nn)%clpname, part_id,  &
                                var_nodims, PRISM_Out, vshape, PRISM_Real, ierror )
     IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in prism_def_var_proto')
   ENDIF
   IF ( psnd(nn)%laction ) THEN
     CALL prism_def_var_proto ( psnd(nn)%vid, psnd(nn)%clpname, part_id,  &
                                var_nodims, PRISM_Out, vshape, PRISM_Real, ierror )
     IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in prism_def_var_proto')
   ENDIF 
 ENDDO

! Define receive variables
 DO nn = 1, nmaxlev
   IF ( trcv(nn)%laction ) THEN
     CALL prism_def_var_proto ( trcv(nn)%vid, trcv(nn)%clpname, part_id,  &
                                var_nodims, PRISM_In, vshape, PRISM_Real, ierror )
     IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in prism_def_var_proto')
   ENDIF
 ENDDO

 !Allocate memory for data exchange and initilize it
 !
 ALLOCATE( bufz(vshape(1):vshape(2), vshape(3):vshape(4)), stat = ierror )
 IF (ierror > 0) CALL prism_abort_proto(comp_id, 'oas_pfl_define', 'Failure in allocating bufz' )

 ! Allocate array to store received fields between two coupling steps
   ALLOCATE( frcv(nx, ny, nlevsoil), stat = ierror )
   IF ( ierror > 0 ) THEN
     CALL prism_abort_proto( comp_id, 'oas_pfl_define', 'Failure in allocating frcv' )
     RETURN
   ENDIF

 !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 !  Termination of definition phase 
 !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 CALL prism_enddef_proto ( ierror )
 IF (ierror /= 0) CALL prism_abort_proto (comp_id, 'oas_pfl_define', 'Failure in prism_enddef')
 WRITE(6,*) 'oaspfl: - oas_pfl_define : variable definition complete'
 
 IF (rank == 0) DEALLOCATE( lglat, lglon, lclat, lclon)

! CALL MPI_Barrier(localcomm, ierror)
!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE oas_pfl_define
