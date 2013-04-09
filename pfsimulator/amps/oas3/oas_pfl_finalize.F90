SUBROUTINE oas_pfl_finalize(argc)

!----------------------------------------------------------------------------
!
! Description
! Terminate the OASIS3 coupler and close the MPI communication
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
! prism_terminate_proto, prism_abort_proto
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!----------------------------------------------------------------------------- 

! Declarations:

! Modules used:

USE oas_pfl_vardef 

IMPLICIT NONE
INTEGER                 :: argc                          !Dummy Variable

!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
! Begin Subroutine oas_pfl_finalize
!------------------------------------------------------------------------------
 argc=0

! Terminate PSMILe
 CALL prism_terminate_proto ( ierror )
 IF (ierror /= 0) CALL prism_abort_proto (comp_id, 'oas_pfl_finalize', 'Failure in prism_terminate_proto')

!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE oas_pfl_finalize
