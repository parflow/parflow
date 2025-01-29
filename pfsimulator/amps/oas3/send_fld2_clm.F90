SUBROUTINE send_fld2_clm(pressure,saturation,topo,ix,iy,nx,ny,nz,nx_f,ny_f,pstep,porosity,dz)

!----------------------------------------------------------------------------
!
! Description
! This routine sends pressure head, soil saturation and porosity from ParFlow3.1 to CLM3.5
! presuure is converted from [m] to [mm]
! saturation is sent as a fraction [-]
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
! Usage of prism libraries
! prism_abort_proto
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!----------------------------------------------------------------------------- 

! Declarations:

! Modules Used:

USE oas_pfl_vardef
USE netcdf
!==============================================================================

IMPLICIT NONE

!==============================================================================

!
INTEGER, INTENT(IN)                :: nx, ny, nz,                       &! Subgrid
                                      nx_f, ny_f,                       &!
                                      ix,iy                              ! Need to write debug netcdf files
REAL(KIND=8), INTENT(IN)           :: pstep                              ! Parflow model time-step in hours
REAL(KIND=8), INTENT(IN)           :: pressure((nx+2)*(ny+2)*(nz+2)),   &! pressure head (m)
                                      saturation((nx+2)*(ny+2)*(nz+2)), &! saturation    (-)
                                      topo((nx+2)*(ny+2)*(nz+2)),       &! mask    (0 for inactive, 1 for active)
                                      porosity((nx+2)*(ny+2)*(nz+2)),   &! porosity [m^3/m^3]
                                      dz((nx+2)*(ny+2)*(nz+2))           ! subsurface layer thickness [m]

                                                                         ! All vectors from parflow on grid w/ ghost nodes for current proc
!Local Variables 

INTEGER                            :: i, j, k, l
INTEGER                            :: isecs                              ! Parflow model time in seconds
INTEGER                            :: j_incr, k_incr                     ! convert 1D vector to 3D i,j,k array
INTEGER, ALLOCATABLE               :: counter(:,:),                     &!
                                      topo_mask(:,:)                     ! Mask for active parflow cells
REAL(KIND=8), ALLOCATABLE          :: sat_snd(:,:,:) , psi_snd(:,:,:)    ! temporary array
!
INTEGER                            :: status, ib, pflncid, dimids(4),   &!
                                      pflvarid(2), cplfreq, cplstop,npes ! Debug netcdf output
CHARACTER(len=19)                  :: foupname
!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!- Begin Subroutine send_fld2_clm 
!------------------------------------------------------------------------------
 isecs= NINT(pstep*3600.d0)
 j_incr = nx_f
 k_incr = nx_f*ny_f

! WRITE(6,*) "CPS DEBUG", isecs

 ALLOCATE ( sat_snd(nx,ny,nlevsoil), stat=ierror)
 IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'send_fld_2clm', 'Failure in allocating sat_snd' )
 ALLOCATE ( psi_snd(nx,ny,nlevsoil), stat=ierror)
 IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'send_fld_2clm', 'Failure in allocating psi_snd' )
 ALLOCATE( topo_mask(nx,ny), stat=ierror)
 IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'send_fld_2clm', 'Failure in allocating topo_mask' )
 ALLOCATE( counter(nx,ny), stat=ierror)
 IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'send_fld_2clm', 'Failure in allocating counter' )


! Create the masking vector
 topo_mask = 0 
 DO i = 1, nx
   DO j = 1, ny
     counter(i,j) = 0
     DO k = nz, 1, -1                                                  ! PF loop over z
         l = 1+i + (nx+2)*(j) + (nx+2)*(ny+2)*(k)
         IF (topo(l) .gt. 0) THEN
           counter(i,j)=counter(i,j)+1
           IF (counter(i,j) .eq. 1) topo_mask(i,j) = k
         ENDIF
     ENDDO
    ENDDO
 ENDDO

 sat_snd = 0
 psi_snd = 0
 DO i = 1, nx
   DO j = 1, ny
     DO k = 1, nlevsoil
     IF (topo_mask(i,j) .gt. 0) THEN
       l = 1+i + j_incr*(j) + k_incr*(topo_mask(i,j)-(k-1))  !
       sat_snd(i,j,k) = saturation(l)
       psi_snd(i,j,k) = pressure(l)*1000.0                            ! convert from [m] to [mm]
     ENDIF
     ENDDO
   ENDDO
 ENDDO

! Debug output file
 IF ( IOASISDEBUGLVL == 1 ) THEN

   CALL MPI_Comm_size(localComm, npes, ierror)
   
   cplfreq = NINT(isecs/(3600.*pfl_timestep)) + 1
   cplstop = NINT(pfl_stoptime/pfl_timestep)

   IF (rank==0 .and. isecs ==0) THEN          !CPS open debug file at beginning only
     foupname = "debugsnd_pfl_clm.nc"
     status = nf90_create(foupname, NF90_CLOBBER, pflncid)
     status = nf90_def_dim(pflncid, "X", nx_tot, dimids(1))
     status = nf90_def_dim(pflncid, "Y", ny_tot, dimids(2))
     status = nf90_def_dim(pflncid, "Z", nlevsoil, dimids(3))
     status = nf90_def_dim(pflncid, "time", cplstop, dimids(4))  !CPS Added time dimension
     status = nf90_def_var(pflncid, "SAT", NF90_DOUBLE, dimids, pflvarid(1))
     status = nf90_def_var(pflncid, "PSI", NF90_DOUBLE, dimids, pflvarid(2))
     status = nf90_enddef(pflncid)
     status = nf90_close(pflncid)
   ENDIF

   DO ib = 0,npes-1
     IF (rank == ib ) THEN
!       CALL MPI_Barrier(localcomm, ierror)
       status = nf90_open(foupname, NF90_WRITE, pflncid)
       status = nf90_inq_varid(pflncid, "SAT" , pflvarid(1))
       status = nf90_inq_varid(pflncid, "PSI" , pflvarid(2))
       status = nf90_put_var(pflncid, pflvarid(1), sat_snd(:,:,:)  , & 
                                        start = (/ ix+1, iy+1, 1, cplfreq/), &
                                        count = (/ nx, ny, nlevsoil, 1 /) )
       status = nf90_put_var(pflncid, pflvarid(2), psi_snd(:,:,:)  , &     
                                        start = (/ ix+1, iy+1, 1, cplfreq/), &
                                        count = (/ nx, ny, nlevsoil, 1 /) )
       status = nf90_close(pflncid)
     ENDIF
   ENDDO
 ENDIF 

! CALL MPI_Barrier(localcomm, ierror)
!
!Send the saturation and pressure sequentially
 DO k = 1, nmaxlev
   IF( wsnd(k)%laction )  CALL oas_pfl_snd( k, isecs, sat_snd(:,:,k), nx, ny, info ,0)
 ENDDO
!
 DO k = 1, nmaxlev
   IF( psnd(k)%laction )  CALL oas_pfl_snd( k, isecs, psi_snd(:,:,k), nx, ny, info ,1)
 ENDDO

!
 DEALLOCATE(sat_snd)
 DEALLOCATE(psi_snd)
 DEALLOCATE(counter)
 DEALLOCATE(topo_mask)

! CALL MPI_Barrier(localcomm, ierror)

!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE send_fld2_clm 
