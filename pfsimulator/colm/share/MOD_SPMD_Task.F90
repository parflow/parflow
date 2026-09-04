#include <define.h>

MODULE MOD_SPMD_Task

!-----------------------------------------------------------------------------------------
! DESCRIPTION:
!
!    SPMD refers to "Single PROGRAM/Multiple Data" parallelization.
! 
!    In CoLM, processes do three types of tasks,
!    1. master : There is only one master process, usually rank 0 in global communicator. 
!                It reads or writes global data, prints informations.
!    2. io     : IO processes read data from files and scatter to workers, gather data from 
!                workers and write to files.
!    3. worker : Worker processes do model calculations.
!   
!    Notice that,
!    1. There are mainly two types of data in CoLM: gridded data and vector data. 
!       Gridded data takes longitude and latitude   as its last two dimensions. 
!       Vector  data takes ELEMENT/PATCH/HRU/PFT/PC as its last dimension.
!       Usually gridded data is allocated on IO processes and vector data is allocated on
!       worker processes.
!    2. One IO process and multiple worker processes form a group. The Input/Output 
!       in CoLM is mainly between IO and workers in the same group. However, all processes
!       can communicate with each other.
!    3. Number of IO is less or equal than the number of blocks with non-zero elements.
!
! Created by Shupeng Zhang, May 2023
!-----------------------------------------------------------------------------------------

   USE MOD_Precision
   IMPLICIT NONE
 
   include 'mpif.h'

#ifndef USEMPI
   
   integer, parameter :: p_root       = 0

   logical, parameter :: p_is_master = .true.
   logical, parameter :: p_is_io     = .true.
   logical, parameter :: p_is_worker = .true.

   integer, parameter :: p_np_glb    = 1    
   integer, parameter :: p_np_worker = 1
   integer, parameter :: p_np_io     = 1

   integer, parameter :: p_iam_glb    = 0
   integer, parameter :: p_iam_io     = 0
   integer, parameter :: p_iam_worker = 0
   
   integer, parameter :: p_np_group   = 1

#else
   integer, parameter :: p_root = 0

   logical :: p_is_master    
   logical :: p_is_io
   logical :: p_is_worker
   logical :: p_is_writeback
   
   integer :: p_comm_glb_plus
   integer :: p_iam_glb_plus

   ! Global communicator
   integer :: p_comm_glb
   integer :: p_iam_glb    
   integer :: p_np_glb     

   ! Processes in the same working group
   integer :: p_comm_group
   integer :: p_iam_group
   integer :: p_np_group

   integer :: p_my_group

   ! Input/output processes 
   integer :: p_comm_io
   integer :: p_iam_io
   integer :: p_np_io     

   integer, allocatable :: p_itis_io (:)
   integer, allocatable :: p_address_io (:)
   
   ! Processes carrying out computing work
   integer :: p_comm_worker
   integer :: p_iam_worker
   integer :: p_np_worker     
   
   integer, allocatable :: p_itis_worker (:)
   integer, allocatable :: p_address_worker (:)

   integer :: p_stat (MPI_STATUS_SIZE)
   integer :: p_err

   ! tags
   integer, PUBLIC, parameter :: mpi_tag_size = 1
   integer, PUBLIC, parameter :: mpi_tag_mesg = 2
   integer, PUBLIC, parameter :: mpi_tag_data = 3 

   integer  :: MPI_INULL_P(1)
   real(r8) :: MPI_RNULL_P(1)

   integer, parameter :: MesgMaxSize = 4194304 ! 4MB 

   ! subroutines
   PUBLIC :: spmd_init
   PUBLIC :: spmd_exit
   PUBLIC :: divide_processes_into_groups

#endif

CONTAINS

#ifdef USEMPI
   !-----------------------------------------
   SUBROUTINE spmd_init (MyComm_r)

   IMPLICIT NONE
   integer, intent(in), optional :: MyComm_r
   logical mpi_inited

      CALL MPI_INITIALIZED (mpi_inited, p_err)

      IF ( .not. mpi_inited ) THEN
         CALL mpi_init (p_err) 
      ENDIF

      IF (present(MyComm_r)) THEN
         CALL MPI_Comm_dup (MyComm_r, p_comm_glb, p_err)
      ELSE
         CALL MPI_Comm_dup (MPI_COMM_WORLD, p_comm_glb, p_err)
      ENDIF

      ! 1. Constructing global communicator.
      CALL mpi_comm_rank (p_comm_glb, p_iam_glb, p_err)  
      CALL mpi_comm_size (p_comm_glb, p_np_glb,  p_err) 

      p_is_master = (p_iam_glb == p_root)
      p_is_writeback = .false.

   END SUBROUTINE spmd_init

   ! ----- -----
   SUBROUTINE spmd_assign_writeback ()

      CALL MPI_Comm_dup  (p_comm_glb, p_comm_glb_plus, p_err)

      CALL MPI_Comm_free (p_comm_glb, p_err)
      
      CALL mpi_comm_rank (p_comm_glb_plus, p_iam_glb_plus,  p_err)  

      p_is_writeback = (p_iam_glb_plus == 0)

      IF (.not. p_is_writeback) THEN

         ! Reconstruct global communicator.
         CALL mpi_comm_split (p_comm_glb_plus, 0, p_iam_glb_plus, p_comm_glb, p_err)
         CALL mpi_comm_rank (p_comm_glb, p_iam_glb, p_err)  
         CALL mpi_comm_size (p_comm_glb, p_np_glb,  p_err) 
         p_is_master = (p_iam_glb == p_root)

      ELSE
         CALL mpi_comm_split (p_comm_glb_plus, MPI_UNDEFINED, p_iam_glb_plus, p_comm_glb, p_err)
         p_is_master = .false.
      ENDIF 

   END SUBROUTINE spmd_assign_writeback

   !-----------------------------------------
   SUBROUTINE divide_processes_into_groups (ngrp)

   IMPLICIT NONE
   
   integer, intent(in) :: ngrp

   ! Local variables
   integer :: iproc
   integer, allocatable :: p_igroup_all (:)

   integer :: nave, nres, igrp, key, nwrt
   character(len=512) :: info
   character(len=5)   :: cnum

      ! 1. Determine number of groups
      IF (ngrp <= 0) THEN
         CALL mpi_abort (p_comm_glb, p_err)
      ENDIF

      ! 2. What task will I take? Which group I am in?
      nave = (p_np_glb-1) / ngrp
      nres = mod(p_np_glb-1, ngrp)

      IF (.not. p_is_master) THEN
         IF (p_iam_glb <= (nave+1)*nres) THEN
            p_is_io = mod(p_iam_glb, nave+1) == 1
            p_my_group = (p_iam_glb-1) / (nave+1)
         ELSE
            p_is_io = mod(p_iam_glb-(nave+1)*nres, nave) == 1
            p_my_group = (p_iam_glb-(nave+1)*nres-1) / nave + nres
         ENDIF

         p_is_worker = .not. p_is_io      
      ELSE
         p_is_io     = .false.
         p_is_worker = .false.
         p_my_group  = -1
      ENDIF

      ! 3. Construct IO communicator and address book.
      IF (p_is_io) THEN
         key = 1
         CALL mpi_comm_split (p_comm_glb, key, p_iam_glb, p_comm_io, p_err)
         CALL mpi_comm_rank  (p_comm_io, p_iam_io, p_err)  
      ELSE
         CALL mpi_comm_split (p_comm_glb, MPI_UNDEFINED, p_iam_glb, p_comm_io, p_err)
      ENDIF
         
      IF (.not. p_is_io) p_iam_io = -1
      allocate (p_itis_io (0:p_np_glb-1))
      CALL mpi_allgather (p_iam_io, 1, MPI_INTEGER, p_itis_io, 1, MPI_INTEGER, p_comm_glb, p_err)
      
      p_np_io = count(p_itis_io >= 0)
      allocate (p_address_io (0:p_np_io-1))

      DO iproc = 0, p_np_glb-1
         IF (p_itis_io(iproc) >= 0) THEN
            p_address_io(p_itis_io(iproc)) = iproc
         ENDIF
      ENDDO

      ! 4. Construct worker communicator and address book.
      IF (p_is_worker) THEN
         key = 1
         CALL mpi_comm_split (p_comm_glb, key, p_iam_glb, p_comm_worker, p_err)
         CALL mpi_comm_rank  (p_comm_worker, p_iam_worker, p_err)  
      ELSE
         CALL mpi_comm_split (p_comm_glb, MPI_UNDEFINED, p_iam_glb, p_comm_worker, p_err)
      ENDIF

      IF (.not. p_is_worker) p_iam_worker = -1
      allocate (p_itis_worker (0:p_np_glb-1))
      CALL mpi_allgather (p_iam_worker, 1, MPI_INTEGER, p_itis_worker, 1, MPI_INTEGER, p_comm_glb, p_err)
      
      p_np_worker = count(p_itis_worker >= 0)
      allocate (p_address_worker (0:p_np_worker-1))

      DO iproc = 0, p_np_glb-1
         IF (p_itis_worker(iproc) >= 0) THEN
            p_address_worker(p_itis_worker(iproc)) = iproc
         ENDIF
      ENDDO

      ! 5. Construct group communicator.
      CALL mpi_comm_split (p_comm_glb, p_my_group, p_iam_glb, p_comm_group, p_err)
      CALL mpi_comm_rank  (p_comm_group, p_iam_group, p_err)  
      CALL mpi_comm_size  (p_comm_group, p_np_group,  p_err) 

      ! 6. Print global task informations.
      allocate (p_igroup_all (0:p_np_glb-1))
      CALL mpi_allgather (p_my_group, 1, MPI_INTEGER, p_igroup_all, 1, MPI_INTEGER, p_comm_glb, p_err)

      IF (p_is_master) THEN

         write (*,'(A)')     '----- MPI information -----'
         write (*,'(A,I0,A)') ' Master is <', p_root, '>'

         DO igrp = 0, p_np_io-1
            write (*,'(A,I0,A,I0,A)') &
               ' Group ', igrp, ' includes IO <', p_address_io(igrp), '> and workers:'
            info = '        '
            nwrt = 0
            DO iproc = 0, p_np_glb-1
               IF ((p_igroup_all(iproc) == igrp) .and. (iproc /= p_address_io(igrp))) THEN
                  nwrt = nwrt + 1
                  write (cnum,'(I5)') iproc
                  info = trim(info) // cnum
                  IF (nwrt == 16) THEN
                     write(*,'(A)') trim(info)
                     info = '        '
                     nwrt = 0
                  ENDIF
               ENDIF
            ENDDO
            IF (nwrt /= 0) THEN
               write(*,'(A)') trim(info)
            ENDIF
         ENDDO
            
         write (*,*) 
      ENDIF

      deallocate (p_igroup_all  )
      
   END SUBROUTINE divide_processes_into_groups

   !-----------------------------------------
   SUBROUTINE spmd_exit

      IF (allocated(p_itis_io       )) deallocate (p_itis_io       )
      IF (allocated(p_address_io    )) deallocate (p_address_io    )
      IF (allocated(p_itis_worker   )) deallocate (p_itis_worker   )
      IF (allocated(p_address_worker)) deallocate (p_address_worker)

      IF (.not. p_is_writeback) THEN
         CALL mpi_barrier (p_comm_glb, p_err)
      ENDIF

      CALL mpi_finalize(p_err)

   END SUBROUTINE spmd_exit

#endif

   ! -- STOP all processes --
   SUBROUTINE CoLM_stop (mesg)

   IMPLICIT NONE
   character(len=*), optional :: mesg

      IF (present(mesg)) write(*,*) trim(mesg)

#ifdef USEMPI
      CALL mpi_abort (p_comm_glb, p_err)
#else
      STOP
#endif

   END SUBROUTINE CoLM_stop

END MODULE MOD_SPMD_Task
