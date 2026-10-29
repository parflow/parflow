!#include <misc.h>

subroutine drv_readvegpf (drv,grid,tile,clm,veg_water_stress_typepf,per_pft_water_stresspf, &
                          wilting_pointpf,field_capacitypf)

  !=========================================================================
  !
  !  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely  
  !  L                        M  available land surface process model.  
  !  M --COMMON LAND MODEL--  C  
  !  C                        L  CLM WEB INFO: http://www.clm.org?
  !  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
  !
  !=========================================================================
  ! DESCRIPTION:	  
  !   Read in vegetation class paramters from input file and assign to
  !   CLM variables.
  !
  ! INPUT DATA FORMAT:
  !  FORTRAN PARAMETER NAME, description (not read in)
  !  values (number of types in vegetation classification)
  !  
  !  This is free format, in any order.  drv_readvegp.f skips any comment lines
  !
  ! REVISION HISTORY:
  !  6 May 1999: Paul Houser; initial code
  !  15 Jan 2000: Paul Houser; revised for F90
  !=========================================================================
  ! $Id: drv_readvegpf.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_gridmodule      ! Grid space module
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use clm_varcon, only : istwet, istice, istdlak , istslak
  implicit none

  !=== Arguments ===========================================================

  type (drvdec)  :: drv              
  type (griddec) :: grid(drv%nc,drv%nr)
  type (tiledec) :: tile(drv%nch)
  type (clm1d)   :: clm (drv%nch)
  integer        :: veg_water_stress_typepf   ! CLM veg water stress mode (1=pressure, 2=saturation)
  integer        :: per_pft_water_stresspf    ! 1 = apply per-PFT wilting point / field capacity from this file
  real(r8)       :: wilting_pointpf           ! scalar Solver.CLM.WiltingPoint default
  real(r8)       :: field_capacitypf          ! scalar Solver.CLM.FieldCapacity default

  !=== Local Variables =====================================================

  character(15) :: vname   ! variable name read from clm_in.dat
  integer :: ioval,t       ! Read error code; tile space counter
  ! Per-PFT wilting point / field capacity, one pair per VegWaterStress mode,
  ! with a presence flag per row (a row absent from drv_vegp.dat is never read).
  real(r8) :: wp_press_v(drv%nch), fc_press_v(drv%nch)
  real(r8) :: wp_sat_v(drv%nch),   fc_sat_v(drv%nch)
  logical  :: wp_press_present, fc_press_present, wp_sat_present, fc_sat_present

  !=== End Variable List ===================================================

  ! Open and read 1-D  CLM input file
  open(9, file=drv%vegpf, form='formatted', status = 'old',action='read')


  ! Setup defaults; this prevents use of unitialized state
  do t=1,drv%nch
     clm(t)%irrig = 0  !default - no irrigation
  end do

  ! No per-PFT row seen yet; each flag flips true when its row is read below.
  wp_press_present = .false.
  fc_press_present = .false.
  wp_sat_present   = .false.
  fc_sat_present   = .false.

  ioval=0
  do while (ioval == 0)

     vname='!'
     read(9,'(a15)',iostat=ioval)vname
     if (vname == 'itypwat'  ) call drv_vpi(drv,tile,clm%itypwat)
     if (vname == 'lai0')      call drv_vpr(drv,tile,clm%minlai) 
     if (vname == 'lai')       call drv_vpr(drv,tile,clm%maxlai) 
     clm%tlai=clm%maxlai
     if (vname == 'sai')       call drv_vpr(drv,tile,clm%tsai  )
     if (vname == 'z0m')       call drv_vpr(drv,tile,clm%z0m   )
     if (vname == 'displa')    call drv_vpr(drv,tile,clm%displa)
     if (vname == 'dleaf')     call drv_vpr(drv,tile,clm%dleaf )
     if (vname == 'roota')     call drv_vpr(drv,tile,tile%roota)
     if (vname == 'rootb')     call drv_vpr(drv,tile,tile%rootb)
     if (vname == 'rhol_vis')  call drv_vpr(drv,tile,clm%rhol(1))
     if (vname == 'rhol_nir')  call drv_vpr(drv,tile,clm%rhol(2))
     if (vname == 'rhos_vis')  call drv_vpr(drv,tile,clm%rhos(1))
     if (vname == 'rhos_nir')  call drv_vpr(drv,tile,clm%rhos(2))
     if (vname == 'taul_vis')  call drv_vpr(drv,tile,clm%taul(1))
     if (vname == 'taul_nir')  call drv_vpr(drv,tile,clm%taul(2))
     if (vname == 'taus_vis')  call drv_vpr(drv,tile,clm%taus(1))
     if (vname == 'taus_nir')  call drv_vpr(drv,tile,clm%taus(2))
     if (vname == 'xl')        call drv_vpr(drv,tile,clm%xl)
     if (vname == 'vw')        call drv_vpr(drv,tile,clm%vw)
     if (vname == 'irrig')     call drv_vpi(drv,tile,clm%irrig)    ! @IMF
     if (vname == 'bkmult')    call drv_vpr(drv,tile,clm%bkmult)   ! @CAP 2014-02-24
     if (vname == 'vcmx25') then                                     ! @RMM 2026 PFT photosyn
        call drv_vpr(drv,tile,clm%vcmx25)
        do t=1,drv%nch
           clm(t)%photosyn_custom = .true.
        end do
     endif
     if (vname == 'c3psn')     call drv_vpr(drv,tile,clm%c3psn)    ! @RMM 2026 PFT photosyn
     if (vname == 'mp')        call drv_vpr(drv,tile,clm%mp)       ! @RMM 2026 PFT photosyn
     if (vname == 'bp')        call drv_vpr(drv,tile,clm%bp)       ! @RMM 2026 PFT photosyn
     if (vname == 'qe25')      call drv_vpr(drv,tile,clm%qe25)     ! @RMM 2026 PFT photosyn
     if (vname == 'folnmx')    call drv_vpr(drv,tile,clm%folnmx)   ! @RMM 2026 PFT photosyn
     if (vname == 'g1_medlyn')  call drv_vpr(drv,tile,clm%g1_medlyn) ! @RMM 2026 Medlyn stomata
     if (vname == 'clump')      call drv_vpr(drv,tile,clm%clump_index) ! @RMM 2026 canopy clumping
     if (vname == 'omega_max')  call drv_vpr(drv,tile,clm%omega_max)   ! @RMM 2026 compensatory RWU
     ! Per-PFT wilting point / field capacity, one pair per VegWaterStress mode.
     ! Read into locals; the pair matching the active mode is applied after the
     ! loop, only when Solver.CLM.PerPFTWaterStress is on.  @RMM 2026
     if (vname == 'wp_press') then
        call drv_vpr(drv,tile,wp_press_v) ; wp_press_present = .true.
     end if
     if (vname == 'fc_press') then
        call drv_vpr(drv,tile,fc_press_v) ; fc_press_present = .true.
     end if
     if (vname == 'wp_sat') then
        call drv_vpr(drv,tile,wp_sat_v) ; wp_sat_present = .true.
     end if
     if (vname == 'fc_sat') then
        call drv_vpr(drv,tile,fc_sat_v) ; fc_sat_present = .true.
     end if
     ! initialize lakpoi from itypwat variable

     do t=1,drv%nch 

        if (clm(t)%itypwat == istdlak .or. clm(t)%itypwat == istslak) then
           clm(t)%lakpoi = .true.
        else
           clm(t)%lakpoi = .false.
        endif

        if (tile(t)%vegt == 18) then  !bare soil index
           clm(t)%baresoil = .true.
        else
           clm(t)%baresoil = .false.
        endif

        ! IMF: Irrigation flag for each veg type added to drv_vegp.dat,  
        !      read from file in above routine)
        !      (irrig=0 -> no irrigation, irrig=1 -> irrigate)
        ! clm(t)%irrig = .false.  !for now - no irrigation 

     end do

  enddo
  close(9)

  !=== Wilting point / field capacity onto tile space ======================
  ! Every tile takes the scalar Solver.CLM.WiltingPoint / FieldCapacity by
  ! default.  This is the sole owner of these fields (clm.F90 no longer sets
  ! them), so behavior is independent of any NaN initialization.
  do t = 1, drv%nch
     clm(t)%wilting_point  = wilting_pointpf
     clm(t)%field_capacity = field_capacitypf
  end do

  ! Per-PFT override (Solver.CLM.PerPFTWaterStress): where the row for the active
  ! VegWaterStress mode was present in drv_vegp.dat, use it instead of the
  ! scalar.  Rows absent from the file are never read, so their tiles keep the
  ! scalar.  When the switch is off the rows are ignored entirely.
  if (per_pft_water_stresspf == 1) then
     do t = 1, drv%nch
        if (veg_water_stress_typepf == 1) then        ! pressure formulation
           if (wp_press_present) clm(t)%wilting_point  = wp_press_v(t)
           if (fc_press_present) clm(t)%field_capacity = fc_press_v(t)
        else if (veg_water_stress_typepf == 2) then   ! saturation formulation
           if (wp_sat_present) clm(t)%wilting_point  = wp_sat_v(t)
           if (fc_sat_present) clm(t)%field_capacity = fc_sat_v(t)
        end if
     end do
  end if

end subroutine drv_readvegpf

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely  
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://www.clm.org?
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! drv_vp.f:
!
! DESCRIPTION:
! The following subroutine simply reads and distributes spatially-constant
!  data from drv_vegp.dat into clm arrays.
!
! REVISION HISTORY:
!  6 May 1999: Paul Houser; initial code
!=========================================================================

subroutine drv_vpi(drv,tile,clmvar)  

  ! Declare Modules and data structures
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  implicit none
  type (drvdec)           :: drv              
  type (tiledec)          :: tile(drv%nch)

  integer t
  integer clmvar(drv%nch)
  integer ivar(drv%nt)

  read(9,*)ivar
  do t=1,drv%nch
     clmvar(t)=ivar(tile(t)%vegt)
  enddo

end subroutine drv_vpi


subroutine drv_vpr(drv,tile,clmvar)  

  ! Declare Modules and data structures
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  implicit none
  type (drvdec)           :: drv              
  type (tiledec)          :: tile(drv%nch)

  integer t
  real(r8) clmvar(drv%nch)
  real(r8) rvar(drv%nt)

  read(9,*)rvar
  do t=1,drv%nch
     clmvar(t)=rvar(tile(t)%vegt)
  enddo

end subroutine drv_vpr
