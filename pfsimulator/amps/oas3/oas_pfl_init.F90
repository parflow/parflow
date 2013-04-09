SUBROUTINE oas_pfl_init(argc)

!----------------------------------------------------------------------------
!
! Description
! Initialize the OASIS3 coupler 
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
! prism_init_comp_proto, prism_get_localcomm_proto, prism_abort_proto
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!----------------------------------------------------------------------------- 

! Declarations:

! Modules Used 

USE oas_pfl_vardef

!==============================================================================

IMPLICIT NONE

!==============================================================================

! Variables 
INTEGER               ::  argc                                ! dummy variable
CHARACTER(len=6)      ::  comp_name = 'oaspfl'                ! Component Name

!------------------------------------------------------------------------------
!- End of header
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!- Begin Subroutine oas_pfl_init 
!------------------------------------------------------------------------------

 argc = 0

!  Step 1: Initialize the PRISM system for the application name 
 CALL prism_init_comp_proto (comp_id, comp_name, ierror)
 IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'oas_pfl_init', 'Failure in prism_init_comp_proto')

!  Step 2: Get MPI communicator for model1 local communication 
 CALL prism_get_localcomm_proto (localComm, ierror )
 IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'oas_pfl_init', 'Failure in prism_get_localcomm_proto')

 CALL MPI_Comm_Rank(localComm, rank, ierror)
 IF (ierror /= 0) CALL prism_abort_proto(comp_id, 'oas_pfl_init', 'Failure in MPI_Comm_Rank') 

!------------------------------------------------------------------------------
!- End of the Subroutine
!------------------------------------------------------------------------------

END SUBROUTINE oas_pfl_init
