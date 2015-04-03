!#include <misc.h>

module clm_varcon

!----------------------------------------------------------------------- 
! 
! Purpose: 
! module for Lsm constants 

! Method: 
! 
! Author: Mariana Vertenstein
! 
!-----------------------------------------------------------------------
! $Id: clm_varcon.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!-----------------------------------------------------------------------

  use precision
  use clm_varpar, only : numcol, numrad
  implicit none

!------------------------------------------------------------------
! Initialize physical constants
!------------------------------------------------------------------

  real(r8) :: grav   = 9.80616   !gravity constant [m/s2]
  real(r8) :: sb     = 5.67e-8   !stefans constant  [W/m2/K4]
!@ real(r8) :: vkc    = 0.4       !von Karman constant [-]
  real(r8) :: vkc    = 0.378       !von Karman constant [-]
  real(r8) :: rwat   = 461.296   !gas constant for water vapor [J/(kg K)]
  real(r8) :: rair   = 287.04    !gas constant for dry air [J/kg/K]
  real(r8) :: roverg = 4.71047e4 !Rw/g constant = (8.3144/0.018)/(9.80616)*1000. mm/K
  real(r8) :: cpliq  = 4188.     !Specific heat of water [J/kg-K]
  real(r8) :: cpice  = 2117.27   !Specific heat of ice [J/kg-K]
  real(r8) :: cpair  = 1004.67   !specific heat of dry air [J/kg/K]
  real(r8) :: hvap   = 2.5104e06 !Latent heat of evap for water [J/kg]
  real(r8) :: hsub   = 2.8440e06 !Latent heat of sublimation    [J/kg]
  real(r8) :: hfus   = 0.3336e06 !Latent heat of fusion for ice [J/kg]
  real(r8) :: denh2o = 1000.0d0     !density of liquid water [kg/m3]
  real(r8) :: denice = 917.      !density of ice [kg/m3]
  real(r8) :: tkair  = 0.023     !thermal conductivity of air   [W/m/k]
  real(r8) :: tkice  = 2.290     !thermal conductivity of ice   [W/m/k]
  real(r8) :: tkwat  = 0.6       !thermal conductivity of water [W/m/k]
  real(r8) :: cwat   = 4.188e06  !LSM: specific heat capacity of water (J/m**3/Kelvin)
  real(r8) :: cice   = 2.094e06  !LSM: specific heat capacity of ice   (J/m**3/Kelvin)
  real(r8) :: tcrit  = 2.5       !critical temperature to determine rain or snow
  real(r8) :: tfrz   = 273.16    !freezing temperature [K]
  real(r8) :: po2    = 0.209     !constant atmospheric partial pressure  O2 (mol/mol)
  real(r8) :: pco2   = 355.e-06  !constant atmospheric partial pressure CO2 (mol/mol)

  real(r8) :: bdsno = 250.       !bulk density snow (kg/m**3)

!------------------------------------------------------------------
! Initialize water type constants
!------------------------------------------------------------------

! "water" types 
!   1     soil
!   2     land ice (glacier)
!   3     deep lake
!   4     shallow lake
!   5     wetland: swamp, marsh, etc

  integer :: istsoil = 1  !soil         "water" type
  integer :: istice  = 2  !land ice     "water" type
  integer :: istdlak = 3  !deep lake    "water" type
  integer :: istslak = 4  !shallow lake "water" type
  integer :: istwet  = 5  !wetland      "water" type

!------------------------------------------------------------------
! Initialize miscellaneous radiation constants
!------------------------------------------------------------------

  integer, private :: i  ! loop index

! saturated soil albedos for 8 color classes: 1=vis, 2=nir

  real(r8) :: albsat(numcol,numrad) !wet soil albedo by color class and waveband
  data(albsat(i,1),i=1,8)/0.12,0.11,0.10,0.09,0.08,0.07,0.06,0.05/
  data(albsat(i,2),i=1,8)/0.24,0.22,0.20,0.18,0.16,0.14,0.12,0.10/

! dry soil albedos for 8 color classes: 1=vis, 2=nir 

  real(r8) :: albdry(numcol,numrad) !dry soil albedo by color class and waveband
  data(albdry(i,1),i=1,8)/0.24,0.22,0.20,0.18,0.16,0.14,0.12,0.10/
  data(albdry(i,2),i=1,8)/0.48,0.44,0.40,0.36,0.32,0.28,0.24,0.20/

! albedo land ice: 1=vis, 2=nir

  real(r8) :: albice(numrad)        !albedo land ice by waveband
  data (albice(i),i=1,numrad) /0.80, 0.55/

! albedo frozen lakes: 1=vis, 2=nir 

  real(r8) :: alblak(numrad)        !albedo frozen lakes by waveband
  data (alblak(i),i=1,numrad) /0.60, 0.40/

! omega,betad,betai for snow 

  real(r8) :: betads  = 0.5       !two-stream parameter betad for snow
  real(r8) :: betais  = 0.5       !two-stream parameter betai for snow
  real(r8) :: omegas(numrad)      !two-stream parameter omega for snow by band
  data (omegas(i),i=1,numrad) /0.8, 0.4/

end module clm_varcon
