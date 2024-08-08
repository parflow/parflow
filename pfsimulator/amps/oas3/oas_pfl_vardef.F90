MODULE oas_pfl_vardef

!----------------------------------------------------------------------------
!
! Description
! Define variables for OASIS3 coupler 
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
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!----------------------------------------------------------------------------- 

! Declarations:

!Modules Used 
USE mpi
USE mod_prism

!==============================================================================

IMPLICIT NONE

!==============================================================================
SAVE
!
! Debug level of OASIS
!     0 : Minimum debugging
!     1 : Debugging

! Variables
INTEGER                                   :: IOASISDEBUGLVL = 0 
INTEGER                                   :: ierror                ! Local Variables
INTEGER                                   :: localComm             ! local MPI communicator and Initialized 
INTEGER                                   :: comp_id               ! component identification
INTEGER                                   :: rank                  ! Rank of the processor, intialized(oas_pfl_init)
INTEGER                                   :: info
INTEGER, PUBLIC                           :: OASIS_Rcv  = 1        ! return code if received field
INTEGER, PUBLIC                           :: OASIS_idle = 0        ! return code if nothing done by oasis

INTEGER, DIMENSION(:,:), ALLOCATABLE      :: mask_land             ! Mask land
INTEGER, DIMENSION(:,:), ALLOCATABLE      :: mask_land_sub         ! Mask land
!
REAL(KIND=8), DIMENSION(:,:), ALLOCATABLE :: bufz                  ! Temp buffer for field transfer
REAL(KIND=8)                              :: pfl_timestep,        &! parflow time step in hrs
                                             pfl_stoptime          ! parflow stop time in hrs
INTEGER, PARAMETER                        :: nlevsoil=10           ! Number of soil layer in CLM
INTEGER, PARAMETER                        :: nmaxlev=100           ! Maximum number of levels of coupling fields
INTEGER                                   :: nx_tot, ny_tot        ! Total Parflow grid points for debug output
! Define type of Coupling variable
TYPE, PUBLIC                              ::   FLD_CPL             ! Type for coupling field information
  LOGICAL                                 ::   laction
  CHARACTER(len = 8)                      ::   clpname             ! Name of the coupling field, max 8 for oasis3
  CHARACTER(len = 3)                      ::   clpref              ! variable reference in parflow
  CHARACTER(len = 1)                      ::   clgrid
  INTEGER                                 ::   vid                 ! Id of the field
END TYPE FLD_CPL

 ! Define coupling fields
TYPE(FLD_CPL), DIMENSION(nmaxlev)         :: trcv, psnd, wsnd      ! Coupling fields (HERE WE DEFINE §3 FIELDS)

REAL(KIND=8),  ALLOCATABLE                ::   frcv(:,:,:)         ! all oaisis receive fields


END MODULE oas_pfl_vardef
