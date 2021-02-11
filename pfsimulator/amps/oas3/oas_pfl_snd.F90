SUBROUTINE oas_pfl_snd(kid, kstep, kdata, nx, ny, kinfo, kindex)

!----------------------------------------------------------------------------
!
! Description
! This routine sends fields to OASIS3 coupler at each coupling time step
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
INTEGER, INTENT(IN)                               :: kstep  ! Parflow model time-step in secs 
INTEGER, INTENT(IN)                               :: kindex ! 0 for saturation, 1 for pressure
INTEGER, INTENT(IN)                               :: nx,ny
REAL(KIND=8), DIMENSION(nx,ny), INTENT(IN)        :: kdata

!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!- Begin Subroutine oas_pfl_snd 
!------------------------------------------------------------------------------

! prepare array (only valid shape, without halos) for OASIS
 bufz = kdata
 IF (kindex .eq. 0) THEN
   CALL prism_put_proto(wsnd(kid)%vid, kstep, bufz, kinfo )        !saturation
   IF ( ierror .NE. PRISM_Ok .AND. ierror .LT. PRISM_Sent ) &
   CALL prism_abort_proto(comp_id, 'oas_pfl_snd', 'Failure in prism_put_proto')
 ELSEIF (kindex .eq. 1) THEN
   CALL prism_put_proto(psnd(kid)%vid, kstep, bufz, kinfo )        !pressure 
   IF ( ierror .NE. PRISM_Ok .AND. ierror .LT. PRISM_Sent ) &
   CALL prism_abort_proto(comp_id, 'oas_pfl_snd', 'Failure in prism_put_proto')
 ENDIF
 IF (IOASISDEBUGLVL == 1) WRITE(6,*) "oaspfl: oas_pfl_snd", kstep, kid
!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE oas_pfl_snd
