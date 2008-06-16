subroutine clm_surfalb (clm, coszen)

!----------------------------------------------------------------------- 
! 
! Purpose: 
! Surface albedo and two-stream fluxes
! 
! Method: 
! Surface albedos. Also fluxes (per unit incoming direct and diffuse
! radiation) reflected, transmitted, and absorbed by vegetation. 
! Also sunlit fraction of the canopy. 
!
! The calling sequence is:
!   -> clm_surfalb:          albedos for next time step
!        -> clm_snowalb:     snow albedos: direct beam
!        -> clm_snowalb:     snow albedos: diffuse
!        -> clm_soilalb:     soil/lake albedos
!        -> clm_twostream:   absorbed, reflected, transmitted solar fluxes (vis dir)
!        -> clm_twostream:   absorbed, reflected, transmitted solar fluxes (vis dif)
!        -> clm_twostream:   absorbed, reflected, transmitted solar fluxes (nir dir)
!        -> clm_twostream:   absorbed, reflected, transmitted solar fluxes (nir dif)
! 
! Author: Gordon Bonan
! 
! ------------------------------------------------------------------------
! $Id: clm_surfalb.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!-----------------------------------------------------------------------

  use precision
  use clmtype
  implicit none

! ------------------------ arguments -------------------------------------
  type (clm1d), intent(inout) :: clm    !CLM 1-D Module
  real(r8), intent(in) :: coszen        !cosine solar zenith angle for next time step
! ------------------------------------------------------------------------

! ------------------------ local variables -------------------------------
  integer  :: ib              !band index
  integer  :: ic              !direct beam: ic=0; diffuse: ic=1
  integer  :: nband = numrad  !number of solar radiation wave bands
  real(r8) :: wl              !fraction of LAI+SAI that is LAI
  real(r8) :: ws              !fraction of LAI+SAI that is SAI
  real(r8) :: mpe = 1.e-06    !prevents overflow for division by zero 
  real(r8) :: vai             !elai+esai
  real(r8) :: rho(numrad)     !leaf/stem refl weighted by fraction LAI and SAI
  real(r8) :: tau(numrad)     !leaf/stem tran weighted by fraction LAI and SAI
  real(r8) :: ftdi(numrad)    !down direct flux below veg per unit dif flux = 0
  real(r8) :: albsnd(numrad)  !snow albedo (direct)
  real(r8) :: albsni(numrad)  !snow albedo (diffuse)
  real(r8) :: gdir            !aver projected leaf/stem area in solar direction
  real(r8) :: ext             !optical depth direct beam per unit LAI+SAI
! ------------------------------------------------------------------------

! initialize output because solar radiation only done if coszen > 0

  do ib = 1, nband
     clm%albd(ib)   = 1.
     clm%albi(ib)   = 1.
     clm%albgrd(ib) = 0._r8
     clm%albgri(ib) = 0._r8
     clm%fabd(ib)   = 0._r8
     clm%fabi(ib)   = 0._r8
     clm%ftdd(ib)   = 0._r8
     clm%ftid(ib)   = 0._r8
     clm%ftii(ib)   = 0._r8
     if (ib==1) clm%fsun = 0.
  end do

! IF COSZEN IS NOT POSITIVE - NEW for CLM_OFFLINE
! NOTE: All NEW changes for CLM offline assumes that the incoming solar 
! radiation is 70% direct, 30% diffuse, and 50% visible, 50% near-infrared 
! (as is assumed in drv_getforce.F90).

  do ib = 1, nband
     albsnd(ib)     = 0._r8
     albsni(ib)     = 0._r8
  end do

  if (coszen <= 0._r8) then
    clm%surfalb = 35./100.*(clm%albd(1)+clm%albd(2)) &
                 +15./100.*(clm%albi(1)+clm%albi(2))
    clm%snoalb =  35./100.*(albsnd(1)+albsnd(2)) &
                 +15./100.*(albsni(1)+albsni(2))
    RETURN 
  endif

! weight reflectance/transmittance by lai and sai

  do ib = 1, nband
     vai = clm%elai + clm%esai
     wl = clm%elai / max( vai,mpe )
     ws = clm%esai / max( vai,mpe )
     rho(ib) = max( clm%rhol(ib)*wl + clm%rhos(ib)*ws, mpe )
     tau(ib) = max( clm%taul(ib)*wl + clm%taus(ib)*ws, mpe )
  end do

! snow albedos: only if coszen > 0 and h2osno > 0

  if ( clm%h2osno > 0._r8 ) then
     ic=0; call clm_snowalb (clm, coszen, nband, ic, albsnd)
     ic=1; call clm_snowalb (clm, coszen, nband, ic, albsni)  
  else
     albsnd(:) = 0._r8
     albsni(:) = 0._r8
  endif

! NEW for CLM offline     

  clm%snoalb = 35./100.*(albsnd(1)+albsnd(2)) + 15./100.*(albsni(1)+albsni(2))

! ground surface albedos: only if coszen > 0

  call clm_soilalb (clm, coszen, nband, albsnd, albsni)      

  if (vai /= 0.) then  ! vegetated patch

! loop over nband wavebands to calculate surface albedos and solar 
! fluxes for vegetated patch for unit incoming direct 
! (ic=0) and diffuse flux (ic=1) only if coszen > 0

     do ib = 1, nband
        ic = 0
        call clm_twostream (clm     , ib , ic      , coszen  , vai     , &
                            rho     , tau, clm%fabd, clm%albd, clm%ftdd, &
                            clm%ftid, gdir)
        ic = 1
        call clm_twostream (clm     , ib , ic      , coszen  , vai     , &
                            rho     , tau, clm%fabi, clm%albi, ftdi    , &
                            clm%ftii, gdir)
     end do
     
! sunlit fraction of canopy. set fsun = 0 if fsun < 0.01.
     
     ext = gdir/coszen * sqrt(1.-rho(1)-tau(1))
     clm%fsun = (1.-exp(-ext*vai)) / max(ext*vai,mpe)
     ext = clm%fsun                                       !temporary fsun
     if (ext < 0.01) then 
        wl = 0._r8                                        !temporary fsun
     else
        wl = ext                                          !temporary fsun
     end if
     clm%fsun = wl

  else     ! non-vegetated patch

     do ib = 1,numrad
        clm%fabd(ib) = 0.
        clm%fabi(ib) = 0.
        clm%ftdd(ib) = 1.
        clm%ftid(ib) = 0.
        clm%ftii(ib) = 1.
        clm%albd(ib) = clm%albgrd(ib)
        clm%albi(ib) = clm%albgri(ib)
        clm%fsun     = 0.
     end do

  endif

! NEW for CLM offline:

  clm%surfalb = 35./100.*(clm%albd(1)+clm%albd(2)) +15./100.*(clm%albi(1)+clm%albi(2))

  return
end subroutine clm_surfalb
