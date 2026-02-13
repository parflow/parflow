!#include <misc.h>

subroutine clm_snowage (clm)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely
!  L                        M  available land surface process model.
!  M --COMMON LAND MODEL--  C
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List:
!
!=========================================================================
! DESCRIPTION:
!  Updates snow cover and snow age, based on BATS code.
!  Modified @RMM 2025 for separate VIS/NIR band aging per Abolafia-Rosenzweig
!  et al. (2022) to allow independent calibration of band-specific albedo decay.
!
!  Aging method depends on albedo scheme:
!    Scheme 0 (CLM): BATS-style dimensionless aging with configurable parameters
!    Schemes 1,2 (VIC/Tarboton): True day counter (days since last snowfall)
!
! REVISION HISTORY:
!  Original Code:  Robert Dickinson
!  15 September 1999: Yongjiu Dai; Integration of code into CLM
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision
!  February 2025: Reed Maxwell; VIS/NIR separation with configurable parameters
!  February 2025: Reed Maxwell; Scheme-dependent aging (day counter for VIC/Tarboton)
!=========================================================================
! $Id: clm_snowage.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

! Declare Modules and data structures

  use precision
  use clmtype
  use clm_varcon, only : tfrz
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm  !CLM 1-D Module

!=== Local Variables =====================================================

  ! VIS band aging factors
  real(r8)                 &
       age1_vis,           & ! snow aging factor due to crystal growth [-]
       age2_vis,           & ! snow aging factor due to surface growth [-]
       age3_vis,           & ! snow aging factor due to accum of other particles [-]
       arg_vis,            & ! temporary variable used in snow age calculation [-]
       arg2_vis,           & ! temporary variable used in snow age calculation [-]
       dela_vis,           & ! temporary variable used in snow age calculation [-]
       sge_vis               ! temporary variable used in snow age calculation [-]

  ! NIR band aging factors
  real(r8)                 &
       age1_nir,           & ! snow aging factor due to crystal growth [-]
       age2_nir,           & ! snow aging factor due to surface growth [-]
       age3_nir,           & ! snow aging factor due to accum of other particles [-]
       arg_nir,            & ! temporary variable used in snow age calculation [-]
       arg2_nir,           & ! temporary variable used in snow age calculation [-]
       dela_nir,           & ! temporary variable used in snow age calculation [-]
       sge_nir               ! temporary variable used in snow age calculation [-]

  ! Shared
  real(r8)                 &
       dels,               & ! fresh snow reset factor [-]
       new_snow_mm           ! new snow this timestep [mm SWE]

  ! Fresh snow threshold for day counter reset [mm SWE]
  real(r8), parameter :: fresh_snow_threshold = 1.0_r8

!=== End Variable List ===================================================

  if (clm%h2osno <= 0.) then

     ! No snow - reset all ages to zero
     clm%snowage = 0.
     clm%snowage_vis = 0.
     clm%snowage_nir = 0.

  else

     ! ===================================================================
     ! Aging method depends on albedo scheme
     ! ===================================================================

     select case (clm%albedo_scheme)

     case (0)  ! CLM scheme: BATS-style dimensionless aging
        ! ================================================================
        ! VIS band aging (using VIS-specific parameters)
        ! ================================================================

        ! Dirt/soot factor (configurable, default 0.3)
        age3_vis = clm%snowage_dirt_soot_vis

        ! Grain growth aging (temperature-dependent)
        ! Uses configurable grain growth factor (default 5000 K)
        arg_vis  = clm%snowage_grain_growth_vis*(1./tfrz-1./clm%t_grnd)
        arg2_vis = min(dble(0.),dble(10.)*arg_vis)
        age2_vis = exp(arg2_vis)
        age1_vis = exp(arg_vis)

        ! Aging increment (uses configurable e-folding time, default 1e6 s)
        dela_vis = (1.0d0/clm%snowage_tau0_vis)*clm%dtime*(age1_vis+age2_vis+age3_vis)

        ! ================================================================
        ! NIR band aging (using NIR-specific parameters)
        ! ================================================================

        ! Dirt/soot factor (configurable, default 0.3)
        age3_nir = clm%snowage_dirt_soot_nir

        ! Grain growth aging (temperature-dependent)
        ! Uses configurable grain growth factor (default 5000 K)
        arg_nir  = clm%snowage_grain_growth_nir*(1./tfrz-1./clm%t_grnd)
        arg2_nir = min(dble(0.),dble(10.)*arg_nir)
        age2_nir = exp(arg2_nir)
        age1_nir = exp(arg_nir)

        ! Aging increment (uses configurable e-folding time, default 1e6 s)
        dela_nir = (1.0d0/clm%snowage_tau0_nir)*clm%dtime*(age1_nir+age2_nir+age3_nir)

        ! ================================================================
        ! Fresh snow reset (shared between bands)
        ! Uses configurable reset factor (default 0.1)
        ! ================================================================
        dels = clm%snowage_reset_factor*max(dble(0.0), clm%h2osno-clm%h2osno_old)

        ! ================================================================
        ! Update ages
        ! ================================================================

        ! VIS band
        sge_vis = (clm%snowage_vis + dela_vis)*(1.0-dels)
        clm%snowage_vis = max(dble(0.0), sge_vis)

        ! NIR band
        sge_nir = (clm%snowage_nir + dela_nir)*(1.0-dels)
        clm%snowage_nir = max(dble(0.0), sge_nir)

        ! Legacy: average of VIS/NIR for backward compatibility
        clm%snowage = 0.5d0*(clm%snowage_vis + clm%snowage_nir)

     case default  ! VIC/Tarboton schemes: true day counter
        ! ================================================================
        ! Simple day counter for VIC and Tarboton albedo schemes
        ! These schemes expect snow age in actual days since last snowfall
        ! ================================================================

        ! Calculate new snow this timestep [mm SWE]
        new_snow_mm = max(0.0_r8, clm%h2osno - clm%h2osno_old)

        if (new_snow_mm > fresh_snow_threshold) then
           ! Significant fresh snow - reset day counter
           clm%snowage_vis = 0.0_r8
           clm%snowage_nir = 0.0_r8
        else
           ! Increment by timestep in days
           clm%snowage_vis = clm%snowage_vis + clm%dtime / 86400.0_r8
           clm%snowage_nir = clm%snowage_nir + clm%dtime / 86400.0_r8
        endif

        ! Ensure non-negative
        clm%snowage_vis = max(0.0_r8, clm%snowage_vis)
        clm%snowage_nir = max(0.0_r8, clm%snowage_nir)

        ! Same age for both bands in day-counter mode
        clm%snowage = clm%snowage_vis

     end select

  endif

end subroutine clm_snowage
