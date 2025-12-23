SUBROUTINE receive_fld2_clm(evap_trans,topo,ix,iy,nx,ny,nz,nx_f,ny_f,pstep)

!----------------------------------------------------------------------------
!
! Description
! This routine receives evapotranspiration flux from OASIS3 coupler via CLM3.5 
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
! 1.00       2011/11/17 P. Shrestha 
! Bug fix for masked coupling
!
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
INTEGER, INTENT(IN)                :: ix, iy,                           &!
                                      nx, ny, nz,                       &! Subgrid
                                      nx_f, ny_f
REAL(KIND=8), INTENT(IN)           :: pstep                              ! Parflow model time-step in hours
REAL(KIND=8), INTENT(IN)           :: topo((nx+2)*(ny+2)*(nz+2))         ! mask    (0 for inactive, 1 for active)
REAL(KIND=8), INTENT(INOUT)        :: evap_trans((nx+2)*(ny+2)*(nz+2))   ! source/sink (1/T)

                                                                         ! All vectors from parflow on grid w/ ghost nodes for current proc
!Local Variables 
INTEGER                            :: i, j, k, l
INTEGER                            :: isecs                              ! Parflow model time in seconds
INTEGER                            :: j_incr, k_incr                     ! convert 1D vector to 3D i,j,k array
INTEGER, ALLOCATABLE               :: counter(:,:),                     &!
                                      topo_mask(:,:)                     ! Mask for active parflow cells
!CPS now allocated in oas_pfl_define
!REAL(KIND=8), ALLOCATABLE          :: frcv(:,:,:)                        ! temporary array

INTEGER                            :: status, ib, pflncid, dimids(4),   &!
                                      pflvarid, cplfreq, cplstop,npes    ! Debug netcdf output
CHARACTER(len=19)                  :: foupname

!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!- Begin Subroutine receive_fld2_clm 
!------------------------------------------------------------------------------
 isecs= NINT(pstep*3600.d0)
 j_incr = nx_f
 k_incr = nx_f*ny_f

!CPS
! ALLOCATE ( frcv(nx,ny,nlevsoil), stat=ierror)
! IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'receive_fld_2clm', 'Failure in allocating fsnd' )
 ALLOCATE( topo_mask(nx,ny), stat=ierror)
 IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'receive_fld_2clm', 'Failure in allocating topo_mask' )
 ALLOCATE( counter(nx,ny), stat=ierror)
 IF (ierror /= 0)  CALL prism_abort_proto( comp_id, 'receive_fld_2clm', 'Failure in allocating counter' )

 topo_mask = 0                  !CPS initialize 
! Create the masking vector
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

 DO k = 1, nlevsoil
   IF( trcv(k)%laction )  CALL oas_pfl_rcv( k, isecs, frcv(:,:,k),nx, ny, info )
 ENDDO
!
DO i = 1, nx
  DO j = 1, ny
    DO k = 1, nlevsoil 
      IF ((topo_mask(i,j) .gt. 0) .and. (mask_land_sub(i,j) .gt. 0)) THEN                    !CPS mask bug fix
        l = 1+i + j_incr*(j) + k_incr*(topo_mask(i,j)-(k-1))  !
        evap_trans(l) = evap_trans(l) + frcv(i,j,k)
      END IF
    ENDDO
  ENDDO
ENDDO

! Debug output file
 IF ( IOASISDEBUGLVL == 1 ) THEN

   CALL MPI_Comm_size(localComm, npes, ierror)

   cplfreq = NINT(isecs/(3600.*pfl_timestep)) + 1
   cplstop = NINT(pfl_stoptime/pfl_timestep)

   IF (rank==0 .and. isecs ==0) THEN          !CPS open debug file at beginning only
     foupname = "debugrcv_pfl_clm.nc"
     status = nf90_create(foupname, NF90_CLOBBER, pflncid)
     status = nf90_def_dim(pflncid, "X", nx_tot, dimids(1))
     status = nf90_def_dim(pflncid, "Y", ny_tot, dimids(2))
     status = nf90_def_dim(pflncid, "Z", nlevsoil, dimids(3))
     status = nf90_def_dim(pflncid, "time", cplstop, dimids(4))  !CPS Added time dimension
     status = nf90_def_var(pflncid, "ETFLUX", NF90_DOUBLE, dimids, pflvarid)
     status = nf90_enddef(pflncid)
     status = nf90_close(pflncid)
   ENDIF

   DO ib = 0,npes-1
     IF (rank == ib ) THEN
!       CALL MPI_Barrier(localcomm, ierror)
       status = nf90_open(foupname, NF90_WRITE, pflncid)
       status = nf90_inq_varid(pflncid, "ETFLUX" , pflvarid)
       status = nf90_put_var(pflncid, pflvarid, frcv(:,:,:)  , &
                                        start = (/ ix+1, iy+1, 1, cplfreq/), &
                                        count = (/ nx, ny, nlevsoil, 1 /) )
       status = nf90_close(pflncid)
     ENDIF
   ENDDO
 ENDIF

! CALL MPI_Barrier(localcomm, ierror)


!CPS  DEALLOCATE(frcv)
 DEALLOCATE(counter)
 DEALLOCATE(topo_mask)
!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE receive_fld2_clm 
