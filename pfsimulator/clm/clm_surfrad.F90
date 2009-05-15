!#include <misc.h>

subroutine clm_surfrad (clm)

!----------------------------------------------------------------------- 
! 
! Purpose: 
! Solar fluxes absorbed by vegetation and ground surface
! 
! Method: 
! Note possible problem when land is on different grid than atmosphere.
!
! Land may have sun above the horizon (coszen > 0) but atmosphere may
! have sun below the horizon (forc_solad = 0 and forc_solai = 0). This is okay
! because all fluxes (absorbed, reflected, transmitted) are multiplied
! by the incoming flux and all will equal zero.
!
! Atmosphere may have sun above horizon (forc_solad > 0 and forc_solai > 0) but
! land may have sun below horizon. This is okay because fabd, fabi,
! ftdd, ftid, and ftii all equal zero so that sabv=sabg=fsa=0. Also,
! albd and albi equal one so that fsr=forc_solad+forc_solai. In other words, all
! the radiation is reflected. NDVI should equal zero in this case.
! However, the way the code is currently implemented this is only true
! if (forc_solad+forc_solai)|vis = (forc_solad+forc_solai)|nir.
!
! Output variables are parsun,parsha,sabv,sabg,fsa,fsr,ndvi
! 
! Author: Gordon Bonan
! 
!-----------------------------------------------------------------------
! $Id: clm_surfrad.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!-----------------------------------------------------------------------

  use precision
  use clmtype
  implicit none

! ------------------------ arguments-------------------------------
  type (clm1d), intent(inout) :: clm    !CLM 1-D Module
! -----------------------------------------------------------------

! ------------------------ local variables ------------------------
  integer ib              !waveband number (1=vis, 2=nir)
  real(r8) abs            !absorbed solar radiation (W/m**2) 
  real(r8) rnir           !reflected solar radiation [nir] (W/m**2)
  real(r8) rvis           !reflected solar radiation [vis] (W/m**2)
  real(r8) laifra         !leaf area fraction of canopy
  real(r8) trd            !transmitted solar radiation: direct (W/m**2)
  real(r8) tri            !transmitted solar radiation: diffuse (W/m**2)
  real(r8) cad(numrad)    !direct beam absorbed by canopy (W/m**2)
  real(r8) cai(numrad)    !diffuse radiation absorbed by canopy (W/m**2)
  integer  nband          !number of solar radiation waveband classes              
  real(r8) fsha           !shaded fraction of canopy
  real(r8) vai            !total leaf area index + stem area index, one sided
  real(r8) mpe            !prevents overflow for division by zero                  
! -----------------------------------------------------------------

  mpe   = 1.e-06
  nband = numrad

  fsha = 1.-clm%fsun
  clm%laisun = clm%elai*clm%fsun
  clm%laisha = clm%elai*fsha
  vai = clm%elai+ clm%esai

! zero summed solar fluxes

  clm%sabg = 0.
  clm%sabv = 0.
  clm%fsa  = 0.

! loop over nband wavebands

  do ib = 1, nband

! absorbed by canopy

     cad(ib)  = clm%forc_solad(ib)*clm%fabd(ib)
     cai(ib)  = clm%forc_solai(ib)*clm%fabi(ib)
     clm%sabv = clm%sabv + cad(ib) + cai(ib)
     clm%fsa  = clm%fsa  + cad(ib) + cai(ib)

! transmitted = solar fluxes incident on ground

     trd = clm%forc_solad(ib)*clm%ftdd(ib)
     tri = clm%forc_solad(ib)*clm%ftid(ib) + clm%forc_solai(ib)*clm%ftii(ib)

! solar radiation absorbed by ground surface

     abs = trd*(1.-clm%albgrd(ib)) + tri*(1.-clm%albgri(ib)) 
     clm%sabg = clm%sabg + abs
     clm%fsa  = clm%fsa  + abs

  end do

! partion visible canopy absorption to sunlit and shaded fractions
! to get average absorbed par for sunlit and shaded leaves

  laifra = clm%elai / max(vai,mpe)
  if (clm%fsun > 0.) then
     clm%parsun = (cad(1)+clm%fsun*cai(1)) * laifra / max(clm%laisun,mpe)
     clm%parsha = (fsha*cai(1))*laifra / max(clm%laisha,mpe)
  else
     clm%parsun = 0._r8 
     clm%parsha = (cad(1)+cai(1))*laifra / max(clm%laisha,mpe)
  endif

! NDVI and reflected solar radiation

  rvis = clm%albd(1)*clm%forc_solad(1) + clm%albi(1)*clm%forc_solai(1) 
  rnir = clm%albd(2)*clm%forc_solad(2) + clm%albi(2)*clm%forc_solai(2)
  clm%fsr = rvis + rnir
  clm%ndvi = (rnir-rvis) / max(rnir+rvis,mpe)

  return
end subroutine clm_surfrad

