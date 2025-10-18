#include <define.h>

SUBROUTINE soil_thermal_parameters(wf_gravels_s,wf_sand_s,wf_clay_s,&
           vf_gravels_s,vf_om_s,vf_sand_s,vf_clay_s,vf_silt_s,vf_pores_s,&
           vf_quartz_mineral_s,BD_mineral_s,k_solids,&
           csol,kdry,ksat_u,ksat_f)

!------------------------------------------------------------------------------------------
! DESCRIPTION:
! Calculate volumetric soil heat capacity and soil thermal conductivity with 8 optional schemes by using the rawdata soil properties.
! The default soil thermal conductivity scheme is the fourth one (Balland V. and P. A. Arp, 2005)
!
! REFERENCE:
! Dai et al.,2019: Evaluation of Soil Thermal Conductivity Schemes for Use in Land Surface Modeling.
! J. of Advances in Modeling Earth Systems, DOI: 10.1029/2019MS001723
!
! Original author: Yongjiu Dai, 02/2018/
!
! Revisions:
! Nan Wei, 06/2018: add to CoLM/mksrfdata
! Nan Wei, 01/2020: update thermal conductivity of gravels
! Nan Wei, 09/2022: add soil thermal conductivity of Hailong He (Yan & He et al., 2019)
! -----------------------------------------------------------------------------------------

use MOD_Precision
USE MOD_Namelist

IMPLICIT NONE
      real(r8), intent(in) :: BD_mineral_s ! bulk density of mineral soil (g/cm^3)
      real(r8), intent(in) :: wf_gravels_s ! gravimetric fraction of gravels
      real(r8), intent(in) :: wf_sand_s    ! gravimetric fraction of sand
      real(r8), intent(in) :: wf_clay_s    ! gravimetric fraction of clay

      real(r8), intent(in) :: vf_quartz_mineral_s ! volumetric fraction of quartz within mineral soil

      real(r8), intent(in) :: vf_pores_s ! volumetric pore space of the soil
      real(r8), intent(in) :: vf_gravels_s ! volumetric fraction of gravels
      real(r8), intent(in) :: vf_om_s      ! volumetric fraction of organic matter
      real(r8), intent(in) :: vf_sand_s    ! volumetric fraction of sand
      real(r8), intent(in) :: vf_clay_s    ! volumetric fraction of clay
      real(r8), intent(in) :: vf_silt_s    ! volumetric fraction of silt

      real(r8), intent(out) :: csol        ! heat capacity of dry soil [J/(m3 K)]
      real(r8), intent(out) :: kdry        ! thermal conductivity for dry soil [W/m/K]
      real(r8), intent(out) :: ksat_u      ! thermal conductivity of unfrozen saturated soil [W/m/K]
      real(r8), intent(out) :: ksat_f      ! thermal conductivity of frozen saturated soil [W/m/K]
      real(r8), intent(out) :: k_solids    ! thermal conductivity of soil solids [W/m/K]

! -----------------------------------------------------------------------------------------
      real(r8) csol_om_s      ! heat capacity of peat soil [J/m3/K]
      real(r8) csol_gravel_s  ! heat capacity of gravels [J/m3/K]
      real(r8) csol_mineral_s ! volumetric heat capacity of fine earth [J/m3/K]

      real(r8) k_om_wet      ! thermal conductivity of wet organic soil [W/m/K]
      real(r8) k_om_dry      ! thermal conductivity of dry organic soil [W/m/K]
      real(r8) k_quartz      ! thermal conductivity of quartz [W/m/K]
      real(r8) k_minerals_o  ! thermal conductivity of non-quartz minerals [W/m/K]
      real(r8) k_gravels_dry ! thermal conductivity of dry gravel soils [W/m/K]
      real(r8) k_gravels_wet ! thermal conductivity of wet gravels [W/m/K]
      real(r8) k_water       ! thermal conductivity of liquid water [W/m/K]
      real(r8) k_ice         ! thermal conductivity of ice [W/m/K]
      real(r8) k_air         ! thermal conductivity of air [W/m/K]
      real(r8) k_minerals    ! thermal conductivity of mineral soil [W/m/K]

      real(r8) a, aa, nwm, n, f1, f2, f3

! -----------------------------------------------------------------------------------------
! The volumetric heat capacity of soil solids (J/m3/K)
! -----------------------------------------------------------------------------------------
! Douglas W. Waples and Jacob S. Waples (2004):
! A Review and Evaluation of Specific Heat Capacities of Rocks, Minerals, and Subsurface Fluids.
! Part 1: Minerals and Nonporous Rocks. Natural Resources Research, 13(2), 97-122.
! Mean value of Table 2
      csol_gravel_s = 2.35e6 ! (J/m3/K) volumetric heat capacity of gravels

! Daniel Hillel, 1998, Environmental soil physics, Table 12.1 (pp.315)
      csol_om_s = 2.51e6   ! Hillel (1982)

! [1] Weighted with mass fractions (CoLM all versions)
! Campbell and Norman (1998, page118, Table 8.2)
! 2.128 = 2.66(density, g/cm3) x 0.80 (quartz specific heat, J /g/K);
! 2.385 = 2.65(density, g/cm3) x 0.89 (other minerals specific heat, J /g/K)
      csol_mineral_s = 1.0e6*(2.128*wf_sand_s+2.385*wf_clay_s) &
                                 / (wf_sand_s+wf_clay_s)

!*[2] Farouki (1981), Table 2
!*    csol_mineral_s = 0.46*4.185*1.0e6  ! = 1.9251*1.0e6
!*
!*[3] Daniel Hillel, 1998, Environmental soil physics, Table 12.1 (pp.315)
!*    csol_mineral_s = 2.0*1.0e6
!*
!*[4] Waples and Waples (2004)
!*    For all low- and medium-density inorganic minerals gives a predictive
!*    relationship between mineral density and thermal capacity at 20C.
!*    real(r8) BD_minerals ! bulk density of soil minerals (g/cm3)
!*    BD_minerals = 2.65   ! (g/cm3)
!*    csol_mineral_s = 1.0263*exp(0.2697*BD_minerals)*1.0e6  ! = 2.0973*1.0e6

      a = vf_sand_s + vf_clay_s + vf_silt_s ! Fraction of minerals of soil

! The volumetric heat capacity of soil solids (J/m3/K)
      csol = vf_gravels_s*csol_gravel_s &
           +      vf_om_s*csol_om_s &
           +            a*csol_mineral_s

! -----------------------------------------------------------------------------------------
! The constants of thermal conductivity (W/m/K)

      k_air = 0.024    ! (W/m/K)
      k_water = 0.57   ! (W/m/K)
      k_ice = 2.29     ! (W/m/K)

      k_om_wet = 0.25  ! (W/m/K)
      k_om_dry = 0.05  ! (W/m/K)
      k_quartz = 7.7   ! (W/m/K)  ! Johansen suggested 7.7

! Thermal conductivity of gravels and crushed rocks
      n = max(vf_pores_s, 0.1) ! =/ porosity of crushed rocks
!*(Johnasen 1975)
      k_gravels_dry = 0.039*n**(-2.2)
      k_gravels_wet = 2.875    ! Cote and Konrad(2005), Thermal conductivity of base-course materials,
                               ! mean value of Table 3

! The thermal conductivty of non-quartz soil minerals
      if(vf_quartz_mineral_s > 0.2)then ! non-quartz soil minerals
         k_minerals_o = 2.0
      else ! coarse-grained soil with low quartz contents
         k_minerals_o = 3.0
      endif

      k_minerals = k_quartz**vf_quartz_mineral_s * k_minerals_o**(1.0-vf_quartz_mineral_s)
      f1 = vf_gravels_s/(vf_gravels_s+vf_om_s+a)
      f2 = vf_om_s     /(vf_gravels_s+vf_om_s+a)
      f3 = a           /(vf_gravels_s+vf_om_s+a)

      select case (DEF_THERMAL_CONDUCTIVITY_SCHEME)
      case (1)
! -----------------------------------------------------------------------------------------
! [1] Oleson K.W. et al., 2013: Technical Description of version 4.5 of the Community
!     Land Model (CLM). NCAR/TN-503+STR (Section 6.3: Soil and snow thermal properties)
! -----------------------------------------------------------------------------------------
!*    real(r8), intent(in) :: soildepth ! (cm)
!*    real(r8) om_watsat ! porosity of organic soil
!*    real(r8) zsapric ! depth (m) that organic matter takes on characteristics of sapric peat
!*    zsapric = 0.5 ! (m)
!*    om_watsat = max(0.93 - 0.1*(0.01*soildepth/zsapric), 0.83)
!*    vf_pores_s = (1.0-vf_om_s)*porsl + vf_om_s*om_watsat
!*    kdry = (1.0-vf_om_s)*(0.137*BD_mineral_s+0.0647)/(2.7-0.947*BD_mineral_s) + vf_om_s*k_om_dry
!*    k_solids_wet = (1.0-vf_om_s)*(8.80*sand+2.92*clay)/(sand+clay) + vf_om_s*k_om_wet

      kdry = f1*k_gravels_dry + f2*k_om_dry + f3*(0.137*BD_mineral_s+0.0647)/(2.7-0.947*BD_mineral_s)

      k_solids = f1*k_gravels_wet + f2*k_om_wet + f3*(8.80*wf_sand_s+2.92*wf_clay_s)/(wf_sand_s+wf_clay_s)

      ksat_u = k_solids**(1.0-vf_pores_s) * k_water**vf_pores_s
      ksat_f = k_solids**(1.0-vf_pores_s) * k_ice**vf_pores_s


      case (2)
! -----------------------------------------------------------------------------------------
! [2] Johansen O (1975): Thermal conductivity of soils. PhD Thesis. Trondheim, Norway:
!     University of Trondheim. US army Crops of Engineerings,
!     CRREL English Translation 637.
! -----------------------------------------------------------------------------------------
      kdry = f1*k_gravels_dry + f2*k_om_dry + f3*(0.137*BD_mineral_s+0.0647)/(2.7-0.947*BD_mineral_s)

      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

      ksat_u = k_solids * k_water**vf_pores_s
      ksat_f = k_solids * k_ice**vf_pores_s


      case (3)
! -----------------------------------------------------------------------------------------
! [3] Cote, J., and J.-M. Konrad (2005), A generalized thermal conductivity model for soils
!     and construction materials. Canadian Geotechnical Journal, 42(2): 443-458.
! -----------------------------------------------------------------------------------------
! Empirical parameters
!*       /rocks and gravels/ /organic fibrous soil (peat)/ /natural minerals soils/
!* chi = /1.7                 0.30                          0.75                  /
!* eta = /1.80                0.87                          1.20                  /
!* kdry = chi*10.0**(-eta*vf_pores_s)

      kdry = (1.70*10.0**(-1.80*vf_pores_s)) * f1 &
           + (0.30*10.0**(-0.87*vf_pores_s)) * f2 &
           + (0.75*10.0**(-1.20*vf_pores_s)) * f3

      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

      ksat_u = k_solids * k_water**vf_pores_s
      ksat_f = k_solids * k_ice**vf_pores_s


      case (4)
! -----------------------------------------------------------------------------------------
! [4] Balland V. and P. A. Arp, 2005: Modeling soil thermal conductivities over a wide
! range of conditions. J. Environ. Eng. Sci. 4: 549-558.
! be careful in specifying all k affecting fractions as Volumetric Fraction,
! whether these fractions are part of the bulk volume, the pore space, or the solid space.
! -----------------------------------------------------------------------------------------
      kdry = f1*k_gravels_dry + f2*k_om_dry + f3*(0.137*BD_mineral_s+0.0647)/(2.7-0.947*BD_mineral_s)

      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

      ksat_u = k_solids * k_water**vf_pores_s
      ksat_f = k_solids * k_ice**vf_pores_s


      case (5)
! -----------------------------------------------------------------------------------------
! [5] Lu et al., 2007: An improved model for predicting soil thermal conductivity from
!     water content at room temperature. Soil Sci. Soc. Am. J. 71:8-14
! -----------------------------------------------------------------------------------------
      kdry = f1*k_gravels_dry &
           + f2*k_om_dry &
           + f3*(-0.56*vf_pores_s+0.51)

      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

      ksat_u = k_solids * k_water**vf_pores_s
      ksat_f = k_solids * k_ice**vf_pores_s


      case (6)
! -----------------------------------------------------------------------------------------
! [6] Series-Parallel Models (Woodside and Messmer, 1961; Kasubuchi et al., 2007;
!                         Tarnawski and Leong, 2012)
! -----------------------------------------------------------------------------------------
      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

! a fitting parameter of the soil solid uniform passage
      aa = 0.0237 - 0.0175*(wf_gravels_s+wf_sand_s)**3

! a fitting parameter of a minuscule portion of soil water (nw) plus a minuscule portion of soil air (na)
      nwm = 0.088 - 0.037*(wf_gravels_s+wf_sand_s)**3

      kdry = k_solids*aa &
           + (1.0-vf_pores_s-aa+nwm)**2/((1.0-vf_pores_s-aa)/k_solids+nwm/k_air) &
           + k_air*(vf_pores_s-nwm)

      ksat_u = k_solids*aa &
             + (1.0-vf_pores_s-aa+nwm)**2/((1.0-vf_pores_s-aa)/k_solids+nwm/k_water) &
             + k_water*(vf_pores_s-nwm)

      ksat_f = k_solids*aa &
                + (1.0-vf_pores_s-aa+nwm)**2/((1.0-vf_pores_s-aa)/k_solids+nwm/k_ice) &
                + k_ice*(vf_pores_s-nwm)


      case (7)
!*-----------------------------------------------------------------------------------------
!*[7] de Vries, Thermal properties of soils, in Physics of Plant Environment,
!*    ed. by W.R. van Wijk (North-Holland, Amsterdam, 1963), pp. 210-235
!*-----------------------------------------------------------------------------------------
      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

      aa = (2.0/(1.0+(k_solids/k_air-1.0)*0.125) &    ! the shape factor
         +  1.0/(1.0+(k_solids/k_air-1.0)*(1.0-2.0*0.125)))/3.0
      kdry = k_air*(vf_pores_s+(1.0-vf_pores_s)*aa*k_solids/k_air) &
           / (vf_pores_s+(1.0-vf_pores_s)*aa)

      aa = (2.0/(1.0+(k_solids/k_water-1.0)*0.125) &  ! the shape factor
         +  1.0/(1.0+(k_solids/k_water-1.0)*(1.0-2.0*0.125)))/3.0
      ksat_u = k_water*(1.0+(1.0-vf_pores_s)*(aa*k_solids/k_water-1.0)) &
                / (1.0+(1.0-vf_pores_s)*(aa-1.0))

      aa = (2.0/(1.0+(k_solids/k_ice-1.0)*0.125) &  ! the shape factor
         +  1.0/(1.0+(k_solids/k_ice-1.0)*(1.0-2.0*0.125)))/3.0
      ksat_f = k_ice*(1.0+(1.0-vf_pores_s)*(aa*k_solids/k_ice-1.0)) &
                / (1.0+(1.0-vf_pores_s)*(aa-1.0))


      case (8)
! -----------------------------------------------------------------------------------------
! [8] Yan & He et al., 2019: A generalized model for estimating effective soil thermal conductivity
!     based on the Kasubuchi algorithm, Geoderma, Vol 353, 227-242
! -----------------------------------------------------------------------------------------
      kdry = -0.5815*vf_pores_s + 0.4999

      k_solids = k_gravels_wet**vf_gravels_s &
               * k_om_wet**vf_om_s &
               * k_minerals**a

      ksat_u = k_solids * k_water**vf_pores_s
      ksat_f = k_solids * k_ice**vf_pores_s


      case (9)
!*-----------------------------------------------------------------------------------------
!*[9] Tarnawski et al (2018) Canadian field soils IV: Modeling thermal
!*    conductivity at dryness and saturation. Int J Thermophys (2018) 39:35
!*    Equation(30-32)
!*-----------------------------------------------------------------------------------------
!*    kdry = 0.55*(1.0-vf_pores_s)**1.4
!*    if((wf_gravels_s+wf_sand_s)>0.4)then
!*       ksat_u = 1.147 + 0.007*vf_pores_s**(-5.31)
!*       ksat_f = ?
!*    else
!*       ksat_u = 1.284 + 13.36e-6*vf_pores_s**(-17.484)
!*       ksat_f = ?
!*    endif


      case (10)
!*-----------------------------------------------------------------------------------------
!*[10] Tarnawski et al (2018) Canadian field soils IV: Modeling thermal
!*    conductivity at dryness and saturation. Int J Thermophys (2018) 39:35
!*    Equation(34-37), de Vries' average.
!*-----------------------------------------------------------------------------------------
!*    if((wf_gravels_s+wf_sand_s)>0.4)then
!*       kdry = (0.3965-0.395*vf_pores_s)/(0.837+1.86*vf_pores_s)
!*       ksat_u = (5.0126-3.369*vf_pores_s)/(1.147+1.55*vf_pores_s)
!*       ksat_f = ?
!*    else
!*       kdry = (0.4234-0.424*vf_pores_s)/(0.238+2.46*vf_pores_s)
!*       ksat_u = (5.0831-3.437*vf_pores_s)/(1.517+1.18*vf_pores_s)
!*       ksat_f = ?
!*    endif
      end select
!*-----------------------------------------------------------------------------------------


END SUBROUTINE soil_thermal_parameters

