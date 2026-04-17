!#include <misc.h>

subroutine clm_snowalb (clm, coszen, nband, ind, alb)

!-----------------------------------------------------------------------
!
! Purpose:
! Determine snow albedos with configurable parameterization schemes
!
! Schemes available:
!   0 = CLM (default) - age-based decay with configurable parameters
!   1 = VIC - dual cold/warm decay rates based on ground temperature
!   2 = Tarboton - Arrhenius temperature-dependent aging
!
!-----------------------------------------------------------------------
! $Id: clm_snowalb.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
! Modified @RMM 2025 - configurable albedo parameters and alternative schemes
!-----------------------------------------------------------------------

  use precision
  use clmtype
  use clm_varcon, only : tfrz
  implicit none

! ------------------------ arguments ------------------------------
  type (clm1d), intent(inout) :: clm       !CLM 1-D Module
  real(r8)    , intent(in) :: coszen       !cosine solar zenith angle for next time step
  integer     , intent(in) :: nband        !number of solar radiation waveband classes
  integer     , intent(in) :: ind          !0=direct beam, 1=diffuse radiation
  real(r8)    , intent(out):: alb(numrad)  !snow albedo by waveband
! -----------------------------------------------------------------

! ------------------------ local variables ------------------------

  integer  :: ib           !waveband class

  real(r8) :: sl    = 2.0  !factor that helps control alb zenith dependence [-]

  real(r8) :: snal0        !vis albedo of new snow (from clm state)
  real(r8) :: snal1        !nir albedo of new snow (from clm state)
  real(r8) :: conn         !constant for visible snow alb calculation (from clm state)
  real(r8) :: cons         !constant for nir snow albedo calculation (from clm state)

  real(r8) :: age_vis      !factor to reduce VIS snow alb due to VIS snow age [-]
  real(r8) :: age_nir      !factor to reduce NIR snow alb due to NIR snow age [-]
  real(r8) :: age_days_vis !VIS snow age in days for VIC/Tarboton schemes
  real(r8) :: age_days_nir !NIR snow age in days for VIC/Tarboton schemes
  real(r8) :: decay_factor_vis !VIS temperature-dependent decay factor for Tarboton scheme
  real(r8) :: decay_factor_nir !NIR temperature-dependent decay factor for Tarboton scheme
  real(r8) :: albs         !temporary vis snow albedo
  real(r8) :: albl         !temporary nir snow albedo
  real(r8) :: cff          !snow alb correction factor for zenith angle > 60 [-]
  real(r8) :: czf          !solar zenith correction for new snow albedo [-]

! -----------------------------------------------------------------

! Use configurable parameters from clm state
  snal0 = clm%albedo_vis_new     ! default 0.95
  snal1 = clm%albedo_nir_new     ! default 0.65
  conn  = clm%albedo_decay_vis   ! default 0.5
  cons  = clm%albedo_decay_nir   ! default 0.2

! zero albedos

  do ib = 1, nband
     alb(ib) = 0._r8
  end do

! =========================================================================
! Snow albedo calculation based on selected scheme
! =========================================================================

  select case (clm%albedo_scheme)

  case (0)  ! CLM default - age-based decay
    ! =====================================================================
    ! CLM Albedo for snow cover.
    ! Snow albedo depends on snow-age, zenith angle, and thickness of snow age
    ! gives reduction of visible radiation
    ! Modified @RMM 2025: Use band-specific snow ages (VIS/NIR separation)
    ! =====================================================================

    ! Correction for snow age - use band-specific ages
    age_vis = 1._r8 - 1._r8/(1._r8 + clm%snowage_vis)
    age_nir = 1._r8 - 1._r8/(1._r8 + clm%snowage_nir)
    albs = snal0*(1._r8 - cons*age_vis)  ! VIS uses VIS age
    albl = snal1*(1._r8 - conn*age_nir)  ! NIR uses NIR age

  case (1)  ! VIC - dual cold/warm decay rates
    ! =====================================================================
    ! VIC-style albedo decay with different rates for cold vs warm conditions
    ! Cold (accumulating): slower decay, albedo_accum_a base
    ! Warm (melting): faster decay, albedo_thaw_a base
    ! Reference: Andreadis et al. (2009), VIC snow model documentation
    ! Modified @RMM 2025: Snow age is now in days (from clm_snowage day counter)
    ! =====================================================================

    ! Snow age already in days from clm_snowage (day counter mode)
    age_days_vis = max(clm%snowage_vis, 0.001_r8)  ! Prevent zero/negative values
    age_days_nir = max(clm%snowage_nir, 0.001_r8)

    if (clm%t_grnd < tfrz) then
      ! Cold/accumulating conditions - slow decay
      albs = snal0 * (clm%albedo_accum_a ** (age_days_vis ** 0.58_r8))
      albl = snal1 * (clm%albedo_accum_a ** (age_days_nir ** 0.58_r8))
    else
      ! Warm/melting conditions - fast decay
      albs = snal0 * (clm%albedo_thaw_a ** (age_days_vis ** 0.46_r8))
      albl = snal1 * (clm%albedo_thaw_a ** (age_days_nir ** 0.46_r8))
    endif

  case (2)  ! Tarboton - Arrhenius temperature dependence
    ! =====================================================================
    ! Tarboton-style temperature-dependent albedo decay
    ! Aging rate increases exponentially as temperature approaches freezing
    ! Reference: Tarboton & Luce (1996), Utah Energy Balance Snow Model
    ! Modified @RMM 2025: Snow age is now in days (from clm_snowage day counter)
    ! =====================================================================

    ! Temperature-dependent aging rate (faster near melting point)
    ! Uses Arrhenius-type formulation - same decay factor for both bands
    decay_factor_vis = exp(5000.0_r8 * (1.0_r8/tfrz - 1.0_r8/max(clm%t_grnd, 200.0_r8)))
    decay_factor_vis = min(decay_factor_vis, 10.0_r8)  ! Cap to prevent extreme values
    decay_factor_nir = decay_factor_vis  ! Same temperature dependence for both bands

    ! Snow age already in days from clm_snowage (day counter mode)
    age_days_vis = clm%snowage_vis
    age_days_nir = clm%snowage_nir

    ! Modified age factor with temperature dependence
    age_vis = (age_days_vis * decay_factor_vis) / (1.0_r8 + age_days_vis * decay_factor_vis)
    age_nir = (age_days_nir * decay_factor_nir) / (1.0_r8 + age_days_nir * decay_factor_nir)

    albs = snal0 * (1.0_r8 - cons * age_vis)
    albl = snal1 * (1.0_r8 - conn * age_nir)

  case default
    ! Fallback to CLM default with band-specific ages
    age_vis = 1._r8 - 1._r8/(1._r8 + clm%snowage_vis)
    age_nir = 1._r8 - 1._r8/(1._r8 + clm%snowage_nir)
    albs = snal0*(1._r8 - cons*age_vis)
    albl = snal1*(1._r8 - conn*age_nir)

  end select

! =========================================================================
! Apply minimum albedo floor (prevents unrealistically low values)
! =========================================================================
  albs = max(clm%albedo_min, albs)
  albl = max(clm%albedo_min, albl)

! =========================================================================
! Zenith angle correction (applied to all schemes)
! =========================================================================

  if (ind == 0) then

    ! Czf corrects albedo of new snow for solar zenith
    cff    = ((1._r8 + 1._r8/sl)/(1._r8 + max(0.001_r8, coszen)*2._r8*sl) - 1._r8/sl)
    cff    = max(cff, 0.0_r8)
    czf    = 0.4_r8*cff*(1._r8 - albs)
    albs = albs + czf
    czf    = 0.4_r8*cff*(1._r8 - albl)
    albl = albl + czf

  endif

  alb(1) = albs
  alb(2) = albl

  return
end subroutine clm_snowalb


