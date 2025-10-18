!-------------------------------------------------------------------------------
SUBROUTINE soil_solids_fractions(BD,gravels,SOC,SAND,CLAY,&
           wf_gravels_s,wf_om_s,wf_sand_s,wf_clay_s,&
           vf_gravels_s,vf_om_s,vf_sand_s,vf_clay_s,vf_silt_s,vf_pores_s,&
           vf_quartz_mineral_s,BD_mineral_s,OM_density,BD_ave)

!-------------------------------------------------------------------------------
! DESCRIPTION:
! Calculate soil porocity and The volumetric fractions of soil solids needed for soil parameter estimations.
! Theta = 1 - BD/PD
!
! REFERENCE:
! Dai et al.,2019: A Global High-Resolution Data Set of Soil Hydraulic and Thermal Properties
! for Land Surface Modeling. J. of Advances in Modeling Earth Systems, DOI: 10.1029/2019MS001784
!
! Original author: Yongjiu Dai, 01/2018
!
! Revisions:
! Nan Wei, 06/2018: add to CoLM/mksrfdata
! Nan Wei, 01/2020: update paticle size of soil solids and gravel porosity
!-------------------------------------------------------------------------------
use MOD_Precision

IMPLICIT NONE
      real(r8), intent(in) :: BD ! bulk density of fine earth (mineral+organic)(g/cm^3)
      real(r8), intent(in) :: gravels ! gravels percentage of soil solids (% of volume)
      real(r8), intent(in) :: SOC     ! organic carbon of fine earth      (% of weight)
      real(r8), intent(in) :: SAND    ! sand percentage of mineral soil   (% of weight)
      real(r8), intent(in) :: CLAY    ! clay percentage of mineral soil   (% of weight)

      real(r8), intent(out) :: vf_gravels_s ! volumetric fraction of gravels
      real(r8), intent(out) :: vf_om_s      ! volumetric fraction of organic matter
      real(r8), intent(out) :: vf_sand_s    ! volumetric fraction of sand
      real(r8), intent(out) :: vf_clay_s    ! volumetric fraction of clay
      real(r8), intent(out) :: vf_silt_s    ! volumetric fraction of silt
      real(r8), intent(out) :: vf_quartz_mineral_s  ! volumetric fraction of quartz within mineral soil

      real(r8), intent(out) :: vf_pores_s   ! volumetric pore space of the soil

      real(r8), intent(out) :: wf_gravels_s ! weight fraction of gravel
      real(r8), intent(out) :: wf_om_s      ! weight fraction of organic matter
      real(r8), intent(out) :: wf_sand_s    ! weight fraction of sand
      real(r8), intent(out) :: wf_clay_s    ! weight fraction of clay

      real(r8), intent(out) :: BD_mineral_s ! bulk density of mineral soil (g/cm^3)
      real(r8), intent(out) :: OM_density   ! OC density (kg/m^3)
      real(r8), intent(out) :: BD_ave       ! bulk density of soil (GRAVELS + MINERALS + ORGANIC MATTER) (g/cm^3)

      real(r8) wf_om_fine_earth ! weight fraction of organic matter within fine earth
      real(r8) wf_silt_s    ! weight fraction of silt within soil soilds

      real(r8) BD_om        ! particle density of organic matter (g/cm3)
      real(r8) BD_minerals  ! particle density of soil minerals (g/cm3)
      real(r8) BD_gravels   ! particle density of gravels (g/cm3)
      real(r8) BD_particle  ! particle density of the soil (g/cm3)
      real(r8) BD_particle_inverse  ! 1/BD_particle
      real(r8) vf_pores_gravels     ! volumetric pore space within gravels
      real(r8) a, SILT
      logical Peters_Lidard_Scheme

      real(r8)   wf_sand_fine_earth
      real(r8)     vf_om_fine_earth
      real(r8) vf_quartz_fine_earth
      integer i
!-----------------------------------------------------------------------
      BD_om = 1.3
      BD_minerals = 2.71                 !average for pd whose om < 0.02 in Table 4, Thermal conductivity of 40 Canadian soils
                                         !Tarnawski et al. 2015, Canadian Field Soils III. Thermal-Conductivity Data and Modeling
      BD_gravels = 2.80                  !Cote and Konrad(2005), Thermal conductivity of base-course materials,
                                         !mean value of Table 5
      SILT = 100.0-SAND-CLAY
      vf_pores_gravels = 0.24            !Cote and Konrad(2005), Thermal conductivity of base-course materials,
                                         !mean value of Table 2

! The volumetric fraction of coarse fragments within the soil
      vf_gravels_s = gravels*(1.0-vf_pores_gravels)/100.0

! The weight fraction of the soil organic matter within fine earth
      wf_om_fine_earth = 1.724*SOC/100.0

! The volumetric fraction of soil organic matter within fine earth
      vf_om_fine_earth = min(wf_om_fine_earth*BD/BD_om, 1.0)

! Bulk density of soil (GRAVELS + MINERALS + ORGANIC MATTER)
      ! BD is the BULK DENSITY of FINE EARTH (MINERALS + ORGANIC MATTER)
      BD_ave = (1.0-vf_gravels_s/(1.0-vf_pores_gravels))*BD + vf_gravels_s*BD_gravels

! Mass fraction of gravels
      wf_gravels_s = vf_gravels_s * BD_gravels/BD_ave
      wf_sand_s = SAND/100.0 * (1.0 - wf_om_fine_earth) * (1.0 - wf_gravels_s)
      wf_clay_s = CLAY/100.0 * (1.0 - wf_om_fine_earth) * (1.0 - wf_gravels_s)
      wf_silt_s = SILT/100.0 * (1.0 - wf_om_fine_earth) * (1.0 - wf_gravels_s)
      wf_om_s = wf_om_fine_earth * (1.0 - wf_gravels_s)

! Volumetric fraction of soil constituents
      vf_sand_s = wf_sand_s * BD_ave/BD_minerals
      vf_clay_s = wf_clay_s * BD_ave/BD_minerals
      vf_silt_s = wf_silt_s * BD_ave/BD_minerals
      vf_om_s = wf_om_s * BD_ave/BD_om

! Particle density of soil (minerals + organic matter + gravels)
      BD_particle_inverse = wf_gravels_s/BD_gravels + (1.0-wf_gravels_s) &
                          * ((1.0-wf_om_fine_earth)/BD_minerals+wf_om_fine_earth/BD_om)

      BD_particle = 1.0/BD_particle_inverse

! POROSITY OF SOIL
      vf_pores_s = 1.0 - BD_ave*BD_particle_inverse
      if(vf_pores_s <= 0.0) then
         write(6,*)"Error: negative soil porosity. BD, PD = ",BD_ave,BD_particle
         stop
      end if

! Bulk density of mineral soil
      BD_mineral_s = (BD_ave - vf_om_s*BD_om - vf_gravels_s*BD_gravels) &
                   / (1.0 - vf_om_s - vf_gravels_s/(1.0-vf_pores_gravels))

      OM_density = BD_ave*wf_om_s*1000.

! Check
      a = vf_gravels_s + vf_om_s + vf_sand_s + vf_clay_s + vf_silt_s
      a = a - (1.0-vf_pores_s)
      if(abs(a) > 1.0e-3)then
         print*, 'Error in soil volumetric calculation 1', a,vf_gravels_s,vf_om_s,vf_sand_s,vf_clay_s,vf_silt_s,vf_pores_s
         call abort
      endif

! The volumetric fraction of quartz (vf_quartz_mineral_s) within mineral soil

      Peters_Lidard_Scheme = .true.

      if(Peters_Lidard_Scheme)then ! (1) Peters-Lidard et al. (1998)
         a = SAND+CLAY
         if(a >= 1.0)then
            CALL vf_quartz(SAND,CLAY,vf_quartz_mineral_s)
         else
            vf_quartz_mineral_s = 0.0
         endif
      else                         ! (2) Calvet et al. (2016)
         wf_sand_fine_earth = SAND/100.0*(1.0-wf_om_fine_earth)
         a = wf_sand_fine_earth / max(wf_om_fine_earth, 1.0e-6)
         if(a < 40.0)then
          ! vf_quartz_fine_earth = 0.12 + 0.0134*a
            vf_quartz_fine_earth = 0.15 + 0.572*wf_sand_fine_earth
         else
            vf_quartz_fine_earth = 0.04 + 0.386*wf_sand_fine_earth
         endif

         a = 1.0 - vf_pores_s - vf_om_s
         vf_quartz_fine_earth = min(vf_quartz_fine_earth,a)
         vf_quartz_mineral_s = min(vf_quartz_fine_earth,a)/(1.0-vf_om_fine_earth)
      endif

END SUBROUTINE soil_solids_fractions



! =========================================
SUBROUTINE vf_quartz(sand,clay,vf_quartz_s)
! =========================================
! Table 2 (page 1212) of  Peters-Lidard, et al., 1998, The effect of soil thermal conductivity
! parameterization on surface energy fluxes and temperatures. J Atmos. Sci., Vol.55, 1209-1224.
!
! Yongjiu Dai, 02/2014
! --------------------------------------------------------------------------------------------
use MOD_Precision

IMPLICIT NONE
      real(r8), intent(in) :: sand
      real(r8), intent(in) :: clay
      real(r8), intent(out) :: vf_quartz_s  ! volumetric fraction of quartz within the soil solids

      real(r8) silt
      integer, parameter :: PNUM=12  ! number of polygons(texture classes)
      logical c(PNUM)   ! indicate wheather a soil is in an class
      integer i

      vf_quartz_s = 0.0
      silt = 100.0-sand-clay

      if(sand<0. .or. silt<0. .or. clay<0.)then
         print*,'Each of the 3 variables should be >= 0: check the data'
         call abort
      end if

      CALL USDA_soil_classes(silt,clay,c)

! Quartz content
      if(c(1))  vf_quartz_s = 0.25   ! clay
      if(c(2))  vf_quartz_s = 0.1    ! silty clay
      if(c(3))  vf_quartz_s = 0.52   ! sandy clay
      if(c(4))  vf_quartz_s = 0.35   ! clay loam
      if(c(5))  vf_quartz_s = 0.1    ! silty clay loam
      if(c(6))  vf_quartz_s = 0.6    ! sandy clay loam
      if(c(7))  vf_quartz_s = 0.4    ! loam
      if(c(8))  vf_quartz_s = 0.25   ! silty loam
      if(c(9))  vf_quartz_s = 0.6    ! sandy loam
      if(c(10)) vf_quartz_s = 0.1    ! silt
      if(c(11)) vf_quartz_s = 0.82   ! loamy sand
      if(c(12)) vf_quartz_s = 0.92   ! sand

END SUBROUTINE vf_quartz



SUBROUTINE USDA_soil_classes(x,y,c)
! --------------------------------------------------------------------------------------------
! USDA major soil textural classes based on the relative percentage of sand, silt and clay
! Initial Author : Wei Shangguan and Yongjiu Dai, 02/2014
! --------------------------------------------------------------------------------------------
use MOD_Precision

IMPLICIT NONE
   integer, parameter :: TNUM=26   ! number of points in the triangle
   integer, parameter :: PNUM=12   ! number of polygons(texture class) in the triangle
   integer, parameter :: PONUM(PNUM)=(/5,3,4,6,4,5,5,8,7,4,4,3/) ! number of points in a polygon (texture class)
   real(r8), intent(in) :: x       ! x(silt) of a soil
   real(r8), intent(in) :: y       ! y(clay) of a soil
   logical, intent(out) :: c(PNUM) ! indicate wheather a soil is in an class

   integer i,j
   real(r8) :: xpos(TNUM)          ! x(silt) coordinates of the  points in the triangle
   real(r8) :: ypos(TNUM)          ! y(clay) coordinates of the  points in the triangle
   integer :: points(PNUM,8)       ! sequence number of the points in a poygon (texture class)
                                   ! 8 is the maximun number of the points
   character(len=15) :: tnames(PNUM)  ! name of a texture class
   integer :: tcodes(PNUM)         ! code of a texture class, may be change accordingly
   real(r8) :: xpol(8)             ! x(silt) coordinates of the  points in a poygon
   real(r8) :: ypol(8)             ! y(clay) coordinates of the  points in a poygon

   xpos = (/ 0.0,  40.0,   0.0,  20.0,  15.0,  40.0,  60.0,   0.0,  27.5,  27.5,  50.0,  52.5,&
            72.5,   0.0,   0.0,  40.0,  50.0,  80.0,  87.5,  15.0,  30.0,  50.0,  80.0,   0.0,&
             0.0, 100.0/)
   ypos = (/55.0,  60.0,  35.0,  35.0,  40.0,  40.0,  40.0,  20.0,  20.0,  27.5,  27.5,  27.5,&
            27.5,  15.0,  10.0,   7.5,   7.5,  12.5,  12.5,   0.0,   0.0,   0.0,   0.0, 100.0,&
             0.0,   0.0/)

   points(1,1:PONUM(1))   = (/24,  1,  5,  6,  2/)
   points(2,1:PONUM(2))   = (/2, 6, 7/)
   points(3,1:PONUM(3))   = (/1, 3, 4, 5/)
   points(4,1:PONUM(4))   = (/5,  4, 10, 11, 12,  6/)
   points(5,1:PONUM(5))   = (/6, 12, 13,  7/)
   points(6,1:PONUM(6))   = (/3,  8,  9, 10,  4/)
   points(7,1:PONUM(7))   = (/10,  9, 16, 17, 11/)
   points(8,1:PONUM(8))   = (/11, 17, 22, 23, 18, 19, 13, 12/)
   points(9,1:PONUM(9))   = (/8, 14, 21, 22, 17, 16,  9/)
   points(10,1:PONUM(10)) = (/18, 23, 26, 19/)
   points(11,1:PONUM(11)) = (/14, 15, 20, 21/)
   points(12,1:PONUM(12)) = (/15, 25, 20/)

   tnames( 1) = 'clay           '
   tnames( 2) = 'silty clay     '
   tnames( 3) = 'sandy clay     '
   tnames( 4) = 'clay loam      '
   tnames( 5) = 'silty clay loam'
   tnames( 6) = 'sandy clay loam'
   tnames( 7) = 'loam           '
   tnames( 8) = 'silty loam     '
   tnames( 9) = 'sandy loam     '
   tnames(10) = 'silt           '
   tnames(11) = 'loamy sand     '
   tnames(12) = 'sand           '

   tcodes=(/1,2,3,4,5,6,7,8,9,10,11,12/)
!  -------------------------------------
   do i = 1, PNUM
      xpol(:) = 0
      do j = 1, PONUM(i)
         xpol(j) = xpos(points(i,j))
         ypol(j) = ypos(points(i,j))
      end do

      call pointinpolygon(x,y,xpol(1:PONUM(i)),ypol(1:PONUM(i)),PONUM(i),c(i))
   end do

END SUBROUTINE USDA_soil_classes



SUBROUTINE pointinpolygon(xp,yp,xpol,ypol,ponum,c)
! --------------------------------------------------------
! For each query point q, InPoly returns one of four char's:
!    i : q is strictly interior to P
!    o : q is strictly exterior to P
!    v : q is a vertex of P
!    e : q lies on the relative interior of an edge of P
!
! Initial Author :  Wei Shangguan, 02/2014
! --------------------------------------------------------
use MOD_Precision

IMPLICIT NONE

   integer, intent(in) :: ponum ! number of points in a polygon
   real(r8), intent(in) :: xp, yp   ! x, y of a point
   real(r8), intent(in) :: xpol(ponum), ypol(ponum)
   logical, intent(out) :: c    ! indicate wheather a soil is in an class

   integer i, i1   ! point index; i1 = i-1 mod n
   real(r8) x      ! x intersection of e with ray
   integer Rcross  ! number of right edge/ray crossings
   integer Lcross  ! number of left edge/ray crossings
   character c2

   Rcross = 0
   Lcross = 0
   c2 = ''

! For each edge e=(i-1,i), see if crosses ray.
   do i = 1, ponum
! First see if q=(0,0) is a vertex.
      if(( xpol(i) - xp )==0 .AND. ( ypol(i) - yp )==0 )then
           c2 = 'v'
           exit
      end if
      i1 = mod(( i-2 + ponum ), ponum) + 1

! if e "straddles" the x-axis...
      if( (( ypol(i) - yp ) > 0 ) .NEQV. (( ypol(i1) - yp ) > 0 ) )then
          ! e straddles ray, so compute intersection with ray.
          x = ( (xpol(i)-xp)*(ypol(i1)-yp) - (xpol(i1)-xp )*(ypol(i)-yp) ) &
              / (ypol(i1)-ypol(i))
          ! crosses ray if strictly positive intersection.
          if(x > 0)then
             Rcross=Rcross+1
          end if
      end if

! if e straddles the x-axis when reversed...
      if( (( ypol(i) - yp ) < 0 ) .NEQV. (( ypol(i1) - yp ) < 0 ) )then
    ! e straddles ray, so compute intersection with ray.
          x = ( (xpol(i)-xp)*(ypol(i1)-yp) - (xpol(i1)-xp)*(ypol(i)-yp) ) &
              / (ypol(i1)-ypol(i))
    ! crosses ray if strictly positive intersection.
          if(x < 0)then
             Lcross=Lcross+1
          end if
      end if

   end do

    ! q on the edge if left and right cross are not the same parity /
    if(c2=='v')then
       c = .true.
    else if( mod(Rcross,2) .NE. mod(Lcross, 2) )then
       c = .true.
       c2 = 'e'
    ! q inside iff an odd number of crossings.
    else if( mod(Rcross,2) == 1 )then
       c = .true.
       c2 = 'i'
    else
       c = .false.
       c2 = 'o'
    end if

END SUBROUTINE pointinpolygon
