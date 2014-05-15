!#include <misc.h>

subroutine clm_soilalb (clm, coszen, nband, albsnd, albsni)

!----------------------------------------------------------------------- 
! 
! Purpose: 
! Determine ground surface albedo, accounting for snow
! 
! Method: 
! 
! Author: Gordon Bonan
! 
!-----------------------------------------------------------------------
! $Id: clm_soilalb.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!-----------------------------------------------------------------------

  use precision
  use clmtype
  use clm_varpar, only : numrad
  use clm_varcon, only : albsat, albdry, alblak, albice, tfrz, istice, istsoil
  implicit none

! ------------------------- arguments ----------------------------
  type (clm1d), intent(inout) :: clm            !CLM 1-D Module
  real(r8)    , intent(in)    :: coszen         !cosine solar zenith angle for next time step
  integer     , intent(in)    :: nband          !number of solar radiation waveband classes
  real(r8)    , intent(in)    :: albsnd(numrad) !snow albedo (direct)
  real(r8)    , intent(in)    :: albsni(numrad) !snow albedo (diffuse)
! -----------------------------------------------------------------

! ------------------------- local variables -----------------------
  integer  ib      !waveband number (1=vis, 2=nir)
  real(r8) inc     !soil water correction factor for soil albedo
  real(r8) albsod  !soil albedo (direct)
  real(r8) albsoi  !soil albedo (diffuse)
! -----------------------------------------------------------------

  do ib = 1, nband
     if (clm%itypwat == istsoil)  then               !soil
        inc    = max(0.11-0.40*clm%h2osoi_vol(1), 0._r8)
        albsod = min(albsat(clm%isoicol,ib)+inc, albdry(clm%isoicol,ib))
        albsoi = albsod
     else if (clm%itypwat == istice)  then           !land ice
        albsod = albice(ib)
        albsoi = albsod
     else if (clm%t_grnd > tfrz) then                !unfrozen lake, wetland
        albsod = 0.05/(max(dble(0.001),coszen) + 0.15)
        albsoi = albsod
     else                                            !frozen lake, wetland
        albsod = alblak(ib)
        albsoi = albsod
     end if

     clm%albgrd(ib) = albsod*(1.-clm%frac_sno) + albsnd(ib)*clm%frac_sno
     clm%albgri(ib) = albsoi*(1.-clm%frac_sno) + albsni(ib)*clm%frac_sno

  end do

  return
end subroutine clm_soilalb


