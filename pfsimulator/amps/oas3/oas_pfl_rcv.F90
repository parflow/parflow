SUBROUTINE oas_pfl_rcv(kid, kstep, kdata, nx, ny, kinfo)

!----------------------------------------------------------------------------
!
! Description
! This routine receives fields from OASIS3 coupler at each coupling time step
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
! Usage of prism libraries
! prism_put_proto, prism_abort_proto
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!----------------------------------------------------------------------------- 

! Declarations:

! Modules Used:

USE oas_pfl_vardef


!==============================================================================

IMPLICIT NONE

!==============================================================================

! * Arguments
!
INTEGER, INTENT(IN)                               :: kid    ! variable intex in the array
INTEGER, INTENT(OUT)                              :: kinfo  ! OASIS info argument
INTEGER, INTENT(IN)                               :: kstep  ! Parflow model time-step in seconds 
INTEGER, INTENT(IN)                               :: nx, ny
REAL(KIND=8), DIMENSION(nx,ny), INTENT(OUT)       :: kdata

! Local Variable
LOGICAL                                           :: llaction

!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!- Begin Subroutine oas_pfl_rcv 
!------------------------------------------------------------------------------

!set buffer to 0 before calling oasis
 bufz = 0. 

 CALL prism_get_proto(trcv(kid)%vid, kstep, bufz, kinfo )
 IF ( ierror .NE. PRISM_Ok .AND. ierror .LT. PRISM_Recvd ) &
 CALL prism_abort_proto(comp_id, 'oas_pfl_rcv', 'Failure in prism_get_proto')

 llaction = .false.
 IF( kinfo == PRISM_Recvd   .OR. kinfo == PRISM_FromRest .OR.   &
     kinfo == PRISM_RecvOut .OR. kinfo == PRISM_FromRestOut )   llaction = .TRUE.

 IF ( llaction ) THEN
  ! Declare to calling routine that OASIS provided coupling field
  kinfo = OASIS_Rcv
 ! Update array which contains coupling field (only on valid shape)
  kdata = bufz
 ELSE
  ! Declare to calling routine that OASIS did not provide coupling field
  kinfo = OASIS_idle
 ENDIF
 IF (IOASISDEBUGLVL == 1) WRITE(6,*) "oaspfl: oas_pfl_rcv", kstep, kinfo, llaction 
!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE oas_pfl_rcv
