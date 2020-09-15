!#include <misc.h>

subroutine clm_coszen (clm, day, coszen)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! clm_coszen.F90: 
!
! DESCRIPTION:
!  Cosine solar zenith angle from:
!    o day (1.x to 365.x), where x=0 (e.g. 213.0) denotes 00:00 at greenwich
!    o latitude,  where SH = - and NH = + 
!    o longitude, where WH = - and EH = +
!
!  The solar declination must match that used in the atmospheric model.
!  For ccm2, this code matches the ccm2 cosz to within +/- 0.0001.
!
!  This discrepancy between clm cosz and atm cosz causes a problem.
!  clm cosz may be <= zero (sun below horizon), in which case albedos
!  equal zero, but atm cosz may be > zero (sun above horizon), in which
!  case atm model needs albedos. There is no problem if the atm model has sun   
!  below horizon, but the CLM has sun above horizon if the atm solar fluxes 
!  are equal zero.  A possible solution then is to reset points with sun 
!  slightly below horizon to slightly above horizon. 
!
!  In practice this error is not very large. e.g., if albedo error is 
!  0.0001 (atm cosz = 0.0001, clm cosz = 0) absorbed solar radiation 
!  error is incident flux * 0.0001.  Since incident flux is itself 
!  multiplied by atm cosz, incident flux is small.  Hence, error is small.
!  In fact the error is smaller than the difference between atm net solar 
!  radiation at the surface and CLM net solar radiation at the surface, which
!  arises due to the different surface radiation parameterizations.
!
!  The reset points are discussed above just in case the atm model 
!  blows up when the albedos are equal zero if atm cosz > 0.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!   3 March 2000:     Jon Radakovich; Revision for diagnostic output
!=========================================================================
! $Id: clm_coszen.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype             ! CLM tile variables
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout)  :: clm    !CLM 1-D Module
  real(r8), intent(in)         :: day
  real(r8), intent(out)        :: coszen

!=== Local Variables =====================================================

  real(r8) slope          !terrain departure from the horizontal (m/m)
  real(r8) aspect         !terrain aspect (radians)
  real(r8) Sx             !corrected slope x from terrain
  real(r8) Sy             !corrected slope y from terrain
  real(r8) theta          !earth orbit seasonal angle in radians
  real(r8) delta          !solar declination angle  in radians
  real(r8) sind           !sine   of declination
  real(r8) cosd           !cosine of declination
  real(r8) phi            !greenwich calendar day + longitude offset
  real(r8) si             !latitude 
  real(r8) loctim         !local time (hour)
  real(r8) hrang          !solar hour angle, 24 hour periodicity (radians)
  real(r8) mcsec          !current seconds in day (0, ..., 86400)
  real(r8) pie             !calculated value of numerical constant Pi

!=== End Variable List ===================================================



  pie = 4.*atan(1.)  ! Value of pie to system and types maximum precision


! Slope aspect from ParFlow terrain
  slope = atan( sqrt( clm%slope_x**2 + clm%slope_y**2 ) )
  Sy = abs(clm%slope_y)
  Sx = abs(clm%slope_x)


  if (clm%slope_y>0 .and. clm%slope_x>0) then
    aspect = atan(Sx/Sy)

  else if(clm%slope_y>0 .and. clm%slope_x<0) then
    aspect = -atan(Sx/Sy)

  else if(clm%slope_y<0 .and. clm%slope_x>0) then
    aspect = pie - atan(Sx/Sy)

  else if(clm%slope_y<0 .and. clm%slope_x<0) then
    aspect = -pie + atan(Sx/Sy)
    
  else if (clm%slope_y>0 .and. clm%slope_x==0) then
    aspect = 0

  else if(clm%slope_y<0 .and. clm%slope_x==0) then
    aspect = pie

  else if(clm%slope_y==0 .and. clm%slope_x>0) then
    aspect = pie/2

  else if(clm%slope_y==0 .and. clm%slope_x<0) then
    aspect = -pie/2
    
  else if(clm%slope_y==0.0 .and. clm%slope_x==0.0) then
    aspect = 0
 
  else
    aspect = -99
    
  end if
  

! Solar declination: match CCM2

  theta = (2.*pie*day)/365.0 
  delta = .006918 - .399912*cos(   theta) + .070257*sin(   theta) &
       - .006758*cos(2.*theta) + .000907*sin(2.*theta) &
       - .002697*cos(3.*theta) + .001480*sin(3.*theta)
  sind = sin(delta)
  cosd = cos(delta)

! Local time

  mcsec = (day - int(day)) * 86400.
  phi = day + (clm%lon)/(2.*pie)
  loctim = (mcsec + (phi-day)*86400.) / 3600.
  if (loctim>=24.) loctim = loctim - 24.

! Hour angle

  hrang = 15. * (loctim-12.) * pie/180.     ! 360/24 = 15

  
  si = clm%lat


! Cosine solar zenith angle.  Reset points with sun slightly below horizon 
! to slightly above horizon, as discussed in description.

  coszen = sin(delta) * sin(si)  * cos(slope) &
    - sin(delta) * cos(si) * sin(slope) * cos(aspect) &
    + cos(delta) * cos(si) * cos(slope) * cos(hrang) &
    + cos(delta) * sin(si) * sin(slope) * cos(aspect) * cos(hrang) &
    + cos(delta) * sin(aspect) * sin(slope) * sin(hrang)


  if (coszen >= -0.001 .and. coszen <= 0.) coszen=0.001


end subroutine clm_coszen