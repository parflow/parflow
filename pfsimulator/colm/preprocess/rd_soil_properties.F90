#include <define.h>
SUBROUTINE rd_soil_properties(grid,nx,ny,planar_mask,numpatch)

!-----------------------------------------------------------------------
! DESCRIPTION:
! Read in soil characteristic dataset GSDE with 30 arc seconds resolution,
! fill the missing data, and estimate soil porosity and
! soil hydraulic and thermal parameters at the resolution of 30 arc seconds.
! The data format are binary.
!
! The Global Soil Characteristics dataset GSDE
!    (http://globalchange.bnu.edu.cn/research/soilw)
! 1 percentage of gravel (fine earth and rock fragments) (% volume)
! 2 percentage of sand   (mineral soil)                  (% weight)
! 3 percentage of clay   (mineral soil)                  (% weight)
! 4 organic Carbon (SOC) (fine earth)                    (% weight)
! 5 bulk density (BD)    (fine earth)                    (g/cm3)
! 6 ...

! The calling sequence is:
! -> soil_solids_fractions:     soil porosity and soil fractions which are needed to estimate
!                               soil hydraulic and thermal parameters
! -> soil_thermal_parameters:   soil solid heat capacity and (dry and saturated) soil thermal conductivity
! -> soil_hydraulic_parameters: soil water retension curves and saturated hydraulic conductivity
!
! Reference:
! (1) Shangguan et al., 2014: A global soil data set for earth system modeling.
!     J. of Advances in Modeling Earth Systems, DOI: 10.1002/2013MS000293
! (2) Dai et al.,2019: A Global High-Resolution Data Set of Soil Hydraulic and Thermal Properties
!     for Land Surface Modeling. J. of Advances in Modeling Earth Systems, DOI: 10.1029/2019MS001784
!
! Original author: Yongjiu Dai, 12/2013/
!
! Rivisions:
! Hua Yuan, 06/2016: add OPENMP parallel function.
! Yongjiu Dai and Nan Wei,
!           06/2018: update a new version of soil hydraulic and thermal parameters
! Nan Wei,  12/2022: output more parameters for BGC parts
! ----------------------------------------------------------------------
use MOD_Precision
use drv_gridmodule 
use MOD_Vars_TimeInvariants
use MOD_Vars_Global, only : zi_soi
IMPLICIT NONE

! arguments:
      ! character(len=256), intent(in) :: dir_rawdata

! local variables:
      ! integer, parameter :: nlat=21600    ! 180*(60*2)
      ! integer, parameter :: nlon=43200    ! 360*(60*2)

      ! character(len=256) lndname

      ! character(len=1) land_chr1(nlon)
      ! character(len=2) land_chr2(nlon)
      ! integer(kind=1)  land_int1(nlon)
      ! integer(kind=2)  land_int2(nlon)

      ! (1) global land cover characteristics
      ! ---------------------------------
      ! integer, allocatable :: landtypes(:,:)  ! GLCC USGS/MODIS IGBP land cover types

      ! (2) global soil characteristcs
      ! --------------------------

      ! integer(kind=1), allocatable :: int_soil_grav_l (:,:) ! Coarse fragments volumetric in %
      ! integer(kind=1), allocatable :: int_soil_sand_l (:,:) ! Sand content (50-2000 micro meter) mass fraction in %
      ! integer(kind=1), allocatable :: int_soil_clay_l (:,:) ! Clay content (0-2 micro meter) mass fraction in %
      ! integer(kind=2), allocatable :: int_soil_oc_l   (:,:) ! Soil organic carbon content (% of weight)
      ! integer(kind=2), allocatable :: int_soil_bd_l   (:,:) ! Bulk density of fine earth (g/cm3)

!   ---------------------------------------------------------------
      integer i, j, t
      ! integer nrow, ncol
      ! integer iunit
      ! integer(i8) length
      integer nsl, MODEL_SOIL_LAYER

      integer  :: nx, ny 
      type (griddec) :: grid(nx,ny) 
      integer  :: numpatch
      integer  :: planar_mask(3,nx*ny)

! soil hydraulic parameters
      ! real(r8), allocatable :: vf_quartz_mineral_s_l (:,:) ! volumetric fraction of quartz within mineral soil
      ! real(r8), allocatable :: vf_gravels_s_l        (:,:) ! volumetric fraction of gravels within soil solids
      ! real(r8), allocatable :: vf_om_s_l             (:,:) ! volumetric fraction of organic matter within soil solids
      ! real(r8), allocatable :: vf_sand_s_l           (:,:) ! volumetric fraction of sand within soil solids
      ! real(r8), allocatable :: wf_gravels_s_l        (:,:) ! gravimetric fraction of gravels
      ! real(r8), allocatable :: wf_sand_s_l           (:,:) ! gravimetric fraction of sand

      ! real(r8), allocatable :: theta_s_l    (:,:) ! volumetric pore space of the soil(cm3/cm3)
      ! real(r8), allocatable :: psi_s_l      (:,:) ! matric potential at saturation (cm)
      ! real(r8), allocatable :: lambda_l     (:,:) ! pore size distribution index (dimensionless)
      ! real(r8), allocatable :: k_s_l        (:,:) ! saturated hydraulic conductivity (cm/day)

      ! real(r8), allocatable :: VGM_theta_r_l(:,:) ! residual moisture content
      ! real(r8), allocatable :: VGM_alpha_l  (:,:) ! a parameter corresponding approximately to the inverse of the air-entry value
      ! real(r8), allocatable :: VGM_n_l      (:,:) ! a shape parameter
      ! real(r8), allocatable :: VGM_L_l      (:,:) ! pore-connectivity parameter

! soil hydraulic parameters from Rosetta3-H3w
      ! real(r8), allocatable :: VGM_theta_r_Rose (:,:) ! residual moisture content
      ! real(r8), allocatable :: VGM_alpha_Rose   (:,:) ! a parameter corresponding approximately to the inverse of the air-entry value
      ! real(r8), allocatable :: VGM_n_Rose       (:,:) ! a shape parameter
      ! real(r8), allocatable :: k_s_Rose         (:,:) ! saturated hydraulic conductivity (cm/day)

! soil thermal parameters
      ! real(r8), allocatable :: csol_l      (:,:) ! heat capacity of soil solids [J/(m3 K)]
      ! real(r8), allocatable :: k_solids_l  (:,:) ! thermal conductivity of soil solids [W/m/K]
      ! real(r8), allocatable :: tksatu_l    (:,:) ! thermal conductivity of unfrozen saturated soil [W/m-K]
      ! real(r8), allocatable :: tksatf_l    (:,:) ! thermal conductivity of frozen saturated soil [W/m-K]
      ! real(r8), allocatable :: tkdry_l     (:,:) ! thermal conductivity for dry soil  [W/(m-K)]

      ! real(r8), allocatable :: OM_density_l(:,:) ! OM_density(kg/m3)
      ! REAL(r8), allocatable :: BD_all_l(:,:)     ! Bulk density of soil (GRAVELS + MINERALS + ORGANIC MATTER)(kg/m3)

! CoLM soil layer thickiness and depths
      ! integer nl_soil
      ! real(r8), allocatable ::  zsoi(:)  ! soil layer depth [m]
      ! real(r8), allocatable ::  dzsoi(:) ! soil node thickness [m]
      ! real(r8), allocatable ::  zsoih(:) ! interface level below a zsoi level [m]

! intermediate variables for soil hydraulic and thermal parameters
      real(r8) soil_grav_l  ! gravel content   (% of volume)
      real(r8) soil_sand_l  ! sand percentage  (% of weight)
      real(r8) soil_clay_l  ! clay percentage  (% of weight)
      real(r8) soil_oc_l    ! organic carbon percentage (% of weight)
      real(r8) soil_bd_l    ! bulk density     (g/cm3)

      real(r8) vf_quartz_mineral_s ! volumetric fraction of quartz within mineral soil
      real(r8) wf_gravels_s ! weight fraction of gravel
      real(r8) wf_om_s      ! weight fraction of organic matter
      real(r8) wf_sand_s    ! weight fraction of sand
      real(r8) wf_clay_s    ! weight fraction of clay

      real(r8) vf_gravels_s ! volumetric fraction of gravels
      real(r8) vf_om_s      ! volumetric fraction of organic matter
      real(r8) vf_sand_s    ! volumetric fraction of sand
      real(r8) vf_clay_s    ! volumetric fraction of clay
      real(r8) vf_silt_s    ! volumetric fraction of silt
      real(r8) vf_pores_s   ! volumetric pore space of the soil

      real(r8) BD_mineral_s ! bulk density of mineral soil (g/cm^3)

      real(r8) theta_s      ! saturated water content (cm3/cm3)
      real(r8) psi_s        ! matric potential at saturation (cm)
      real(r8) lambda       ! pore size distribution index (dimensionless)
      real(r8) k_s          ! saturated hydraulic conductivity (cm/day)

      real(r8) VGM_theta_r  ! residual moisture content
      real(r8) VGM_alpha    ! a parameter corresponding approximately to the inverse of the air-entry value
      real(r8) VGM_n        ! a shape parameter
      real(r8) VGM_L        ! pore-connectivity parameter

      real(r8) csol_s         ! heat capacity of soil solids [J/(m3 K)]
      real(r8) k_solids_s     ! thermal conductivity of soil solids [W/m/K]
      real(r8) tksatu_s       ! thermal conductivity of unfrozen saturated soil [W/m-K]
      real(r8) tksatf_s       ! thermal conductivity of frozen saturated soil [W/m-K]
      real(r8) tkdry_s        ! thermal conductivity for dry soil  [W/(m-K)]
      REAL(r8) OM_density_s   ! soil organic carbon density
      REAL(r8) BD_all_s       ! bulk density of soil

      ! character c
      real(r8) a,SOM
      real(r8) soildepth
      integer ii, iii, iiii, jj, jjj, jjjj

! ........................................
! ... (1) gloabl land cover characteristics
! ........................................
      ! iunit = 100
      ! inquire(iolength=length) land_chr1
      ! allocate ( landtypes(nlon,nlat) )

#if(defined LULC_USGS)
     !! GLCC USGS classification
     !! -------------------
     ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/landtypes_usgs_update.bin'
     ! print*,lndname

     ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
     ! do nrow = 1, nlat
     !    read(iunit,rec=nrow,err=100) land_chr1
     !    landtypes(:,nrow) = ichar(land_chr1(:))
     ! enddo
     ! close (iunit)
#endif

#if(defined LULC_IGBP)
     !! MODIS IGBP classification
     !! -------------------
     ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/landtypes_igbp_update.bin'
     ! print*,lndname

     ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
     ! do nrow = 1, nlat
     !    read(iunit,rec=nrow,err=100) land_chr1
     !    landtypes(:,nrow) = ichar(land_chr1(:))
     ! enddo
     ! close (iunit)
#endif

! .................................
! ... (2) global soil charateristics
! .................................
      ! nl_soil = 10
      ! allocate ( zsoi(1:nl_soil), dzsoi(1:nl_soil), zsoih(0:nl_soil) )

     ! ! ----------------------------------
     ! ! soil layer thickness, depths (m)
     ! ! ----------------------------------
     ! do nsl = 1, nl_soil
     !   zsoi(nsl) = 0.025*(exp(0.5*(nsl-0.5))-1.)  ! node depths
     ! end do

     ! dzsoi(1) = 0.5*(zsoi(1)+zsoi(2))         ! =zsoih(1)
     ! dzsoi(nl_soil) = zsoi(nl_soil)-zsoi(nl_soil-1)
     ! do nsl = 2, nl_soil-1
     !    dzsoi(nsl) = 0.5*(zsoi(nsl+1)-zsoi(nsl-1))  ! thickness b/n two interfaces
     ! end do

     ! zsoih(0) = 0.
     ! zsoih(nl_soil) = zsoi(nl_soil) + 0.5*dzsoi(nl_soil)
     ! do nsl = 1, nl_soil-1
     !    zsoih(nsl) = 0.5*(zsoi(nsl)+zsoi(nsl+1))    ! interface depths
     ! enddo

      ! allocate ( int_soil_grav_l       (nlon,nlat) ,&
      !            int_soil_sand_l       (nlon,nlat) ,&
      !            int_soil_clay_l       (nlon,nlat) ,&
      !            int_soil_oc_l         (nlon,nlat) ,&
      !            int_soil_bd_l         (nlon,nlat)  )

      ! allocate ( vf_quartz_mineral_s_l (nlon,nlat) ,&
      !            vf_gravels_s_l        (nlon,nlat) ,&
      !            vf_om_s_l             (nlon,nlat) ,&
      !            vf_sand_s_l           (nlon,nlat) ,&
      !            wf_gravels_s_l        (nlon,nlat) ,&
      !            wf_sand_s_l           (nlon,nlat) ,&
      !            theta_s_l             (nlon,nlat) ,&
      !            psi_s_l               (nlon,nlat) ,&
      !            lambda_l              (nlon,nlat) ,&
      !            k_s_l                 (nlon,nlat)  )

      ! allocate ( VGM_theta_r_l         (nlon,nlat) ,&
      !            VGM_alpha_l           (nlon,nlat) ,&
      !            VGM_n_l               (nlon,nlat) ,&
      !            VGM_L_l               (nlon,nlat)   )

      ! allocate ( VGM_theta_r_Rose      (nlon,nlat) ,&
      !            VGM_alpha_Rose        (nlon,nlat) ,&
      !            VGM_n_Rose            (nlon,nlat) ,&
      !            k_s_Rose              (nlon,nlat)  )
 
      ! allocate ( csol_l                (nlon,nlat) ,&
      !            k_solids_l            (nlon,nlat) ,&
      !            tksatu_l              (nlon,nlat) ,&
      !            tksatf_l              (nlon,nlat) ,&
      !            tkdry_l               (nlon,nlat)  )
      ! allocate ( OM_density_l          (nlon,nlat) ,&
      !            BD_all_l              (nlon,nlat)  )

! -----------------------------------------------------------------
! Soil physical properties at the model soil vertical layers
! The parameters of the top NINTH soil layers were given by datasets
! [0-0.0175, 0.0175-0.045(the top two layers share the same data), 0.045-0.091, 0.091-0.166, 0.166-0.289,
!  0.289-0.493, 0.493-0.829, 0.829-1.383 and 1.383-2.296 m].
! The NINTH layer's soil parameters will assigned to the bottom soil layer (2.296 - 3.8019m).
! -----------------------------------------------------------------
      ! iunit = 100
      DO nsl = 1, 8
         MODEL_SOIL_LAYER = nsl
         ! write(c,'(i1)') MODEL_SOIL_LAYER

         ! ------------------------------------
         ! (6.1) precentage of gravel (% volume)
         ! ------------------------------------
         ! inquire(iolength=length) land_int1
         ! lndname = trim(dir_rawdata)//'soil/GRAV_L'//trim(c)
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) land_int1
         !    int_soil_grav_l(:,nrow) = land_int1(:)
         ! enddo
         ! close (iunit)

         ! ----------------------------------
         ! (6.2) percentage of sand (% weight)
         ! ----------------------------------
         ! inquire(iolength=length) land_int1
         ! lndname = trim(dir_rawdata)//'soil/SAND_L'//trim(c)
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) land_int1
         !    int_soil_sand_l(:,nrow) = land_int1(:)
         ! enddo
         ! close (iunit)

         ! ----------------------------------
         ! (6.3) percentage of clay (% weight)
         ! ----------------------------------
         ! inquire(iolength=length) land_int1
         ! lndname = trim(dir_rawdata)//'soil/CLAY_L'//trim(c)
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) land_int1
         !    int_soil_clay_l(:,nrow) = land_int1(:)
         ! enddo
         ! close (iunit)

         ! -------------------------------------
         ! (6.4) percentage of organic carbon (%)
         ! -------------------------------------
         ! inquire(iolength=length) land_int2
         ! lndname = trim(dir_rawdata)//'soil/OC_L'//trim(c)
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) land_int2
         !    int_soil_oc_l(:,nrow) = land_int2(:)
         ! enddo
         ! close (iunit)

         ! -------------------------
         ! (6.5) bulk density (g/cm3)
         ! -------------------------
         ! inquire(iolength=length) land_int2
         ! lndname = trim(dir_rawdata)//'soil/BD_L'//trim(c)
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) land_int2
         !    int_soil_bd_l(:,nrow) = land_int2(:)
         ! enddo
         ! close (iunit)

         ! -------------------------
         ! (6.6) Rosetta parameters generated by yonggen Zhang
         ! -------------------------
         ! inquire(iolength=length) VGM_theta_r_Rose(:,1)
         ! lndname = '/work/ygzhang/data/CLMrawdata_2021/Rosetta_VGM/SSCBD_L'//trim(c)//'_VGM_merged_thr_binary'
         ! print*,lndname
 
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) VGM_theta_r_Rose(:,nrow)
         ! end do
         ! close(iunit)
 
         ! inquire(iolength=length) VGM_alpha_Rose(:,1)
         ! lndname = '/work/ygzhang/data/CLMrawdata_2021/Rosetta_VGM/SSCBD_L'//trim(c)//'_VGM_merged_alpha_binary'
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) VGM_alpha_Rose(:,nrow)
         ! end do
         ! close(iunit)

         ! inquire(iolength=length) VGM_n_Rose(:,1)
         ! lndname = '/work/ygzhang/data/CLMrawdata_2021/Rosetta_VGM/SSCBD_L'//trim(c)//'_VGM_merged_n_binary'
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) VGM_n_Rose(:,nrow)
         ! end do
         ! close(iunit)

         ! inquire(iolength=length) k_s_Rose(:,1)
         ! lndname = '/work/ygzhang/data/CLMrawdata_2021/Rosetta_VGM/SSCBD_L'//trim(c)//'_VGM_merged_Ks_binary'
         ! print*,lndname

         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='old')
         ! do nrow = 1, nlat
         !    read(iunit,rec=nrow,err=100) k_s_Rose(:,nrow)
         ! end do
         ! close(iunit)


         ! ---------------------------------------
         ! calculate soil parameters
         ! ---------------------------------------

#ifdef OPENMP
print *, 'OPENMP enabled, threads num = ', OPENMP, "soil parameters..."
!$OMP PARALLEL DO NUM_THREADS(OPENMP) SCHEDULE(DYNAMIC,1) &
!$OMP PRIVATE(soil_grav_l,soil_sand_l,soil_clay_l,soil_oc_l,soil_bd_l) &
!$OMP PRIVATE(vf_quartz_mineral_s,wf_gravels_s,wf_om_s,wf_sand_s,wf_clay_s) &
!$OMP PRIVATE(vf_gravels_s,vf_om_s,vf_sand_s,vf_clay_s,vf_silt_s,vf_pores_s) &
!$OMP PRIVATE(BD_mineral_s,theta_s,psi_s,lambda,k_s,csol_s,k_solids_s,tksatu_s,tksatf_s,tkdry_s,OM_density_s,BD_all_s) &
!$OMP PRIVATE(VGM_theta_r,VGM_alpha,VGM_n,VGM_L) &
!$OMP PRIVATE(soildepth,a,SOM,i)
#endif

         ! do j = 1, nlat
         !    do i = 1, nlon
         do t = 1, numpatch

               i = planar_mask(1,t)
               j = planar_mask(2,t)

               patchclass(t)  = grid(i,j)%patchclass
               patchlatr(t)   = grid(i,j)%patchlatr*4.*atan(1.)/180.  
               patchlonr(t)   = grid(i,j)%patchlonr*4.*atan(1.)/180.

               soil_grav_l    =  grid(i,j)%int_soil_grav_l(nsl)         ! int_soil_grav_l(i,j) ! Volumetric in %
               soil_sand_l    =  grid(i,j)%int_soil_sand_l(nsl)         ! int_soil_sand_l(i,j) ! Gravimetric fraction in %
               soil_clay_l    =  grid(i,j)%int_soil_clay_l(nsl)         ! int_soil_clay_l(i,j) ! Gravimetric fraction in %
               soil_oc_l      =  grid(i,j)%int_soil_oc_l(nsl)*0.01      ! int_soil_oc_l  (i,j)*0.01 ! Gravimetric fraction (fine earth) in %
               soil_bd_l      =  grid(i,j)%int_soil_bd_l(nsl)*0.01      ! int_soil_bd_l  (i,j)*0.01 ! (fine earth) in g/cm3

               if(soil_grav_l < 0.0) soil_grav_l = 0.0  ! missing value = -1

#if(defined LULC_USGS)
               !if(landtypes(i,j)==16)then   !WATER BODIES(16)
               if(patchclass(t)==16)then
#endif
#if(defined LULC_IGBP)
               if(patchclass(t)==17)then   !WATER BODIES(17)
#endif
                  soil_grav_l = 0.
                  soil_sand_l = 10.
                  soil_clay_l = 45.
                  soil_oc_l   = 3.0
                  soil_bd_l   = 1.2
               endif

#if(defined LULC_USGS)
               if(patchclass(t)==24)then   !GLACIER and ICESHEET(24)
#endif
#if(defined LULC_IGBP)
               if(patchclass(t)==15)then   !GLACIER and ICE SHEET(15)
#endif
                  soil_grav_l = 90.
                  soil_sand_l = 89.
                  soil_clay_l = 1.
                  soil_oc_l   = 0.
                  soil_bd_l   = 2.0
               endif

#if(defined Gravels0)
                  soil_grav_l = 0.0
#endif

               if(patchclass(t)/=0)then    !NOT OCEAN(0)
                  ! checking the soil physical properties
                  ! ------------------------------------
                  if( soil_sand_l < 0.0 ) soil_sand_l = 43.   ! missing value = -100
                  if( soil_clay_l < 0.0 ) soil_clay_l = 18.   ! missing value = -100
                  if( soil_oc_l   < 0.0   ) soil_oc_l = 1.0     ! missing value = -999
                  if( soil_bd_l   < 0.0   ) soil_bd_l = 1.2     ! missing value = -999

                  if( soil_sand_l < 1.0   ) soil_sand_l = 1.
                  if( soil_clay_l < 1.0   ) soil_clay_l = 1.

                  a = soil_sand_l + soil_clay_l
                  if( a >= 96. ) then
                      soil_sand_l = soil_sand_l * 96. / a
                      soil_clay_l = soil_clay_l * 96. / a
                  endif
                  if( soil_oc_l < 0.01 ) soil_oc_l = 0.01
                  if( soil_oc_l > 58.0 ) soil_oc_l = 58.0
                  if( soil_bd_l < 0.1  ) soil_bd_l = 0.1

                  soildepth = zi_soi(MODEL_SOIL_LAYER + 1)*100.0
                  if (soil_bd_l < 0.111 .or. soil_bd_l > 2.0 .or. soil_oc_l > 10.0) then
                     SOM=1.724*soil_oc_l
                     soil_bd_l = 0.111*2.0/(2.0*SOM/100.+0.111*(100.-SOM)/100.)
                  end if

                 ! --------------------------------------------------
                 ! The weight and volumetric fractions of soil solids
                 ! --------------------------------------------------
                  CALL soil_solids_fractions(&
                       soil_bd_l,soil_grav_l,soil_oc_l,soil_sand_l,soil_clay_l,&
                       wf_gravels_s,wf_om_s,wf_sand_s,wf_clay_s,&
                       vf_gravels_s,vf_om_s,vf_sand_s,vf_clay_s,vf_silt_s,vf_pores_s,&
                       vf_quartz_mineral_s,BD_mineral_s,OM_density_s,BD_all_s)

                       theta_s = vf_pores_s

                 ! ---------------------------------------------------------------------
                 ! The volumetric heat capacity and thermal conductivity of soil solids
                 ! ---------------------------------------------------------------------
                  CALL soil_thermal_parameters(&
                       wf_gravels_s,wf_sand_s,wf_clay_s,&
                       vf_gravels_s,vf_om_s,vf_sand_s,vf_clay_s,vf_silt_s,vf_pores_s,&
                       vf_quartz_mineral_s,BD_mineral_s,k_solids_s,&
                       csol_s,tkdry_s,tksatu_s,tksatf_s)
                 ! -----------------------------
                 ! The soil hydraulic properties
                 ! -----------------------------
                 ! CALL soil_hydraulic_parameters(soil_bd_l,soil_sand_l,soil_clay_l,soil_oc_l,soildepth,&
                 !       soil_grav_l,theta_s,psi_s,lambda,k_s,&
                 !       VGM_theta_r,VGM_alpha,VGM_n,VGM_L,&
                 !       VGM_theta_r_Rose(i,j),VGM_alpha_Rose(i,j),VGM_n_Rose(i,j),k_s_Rose(i,j))

                 vf_gravels_s = vf_gravels_s/(1 - vf_pores_s)
                 vf_om_s      = vf_om_s     /(1 - vf_pores_s)
                 vf_sand_s    = vf_sand_s   /(1 - vf_pores_s)
                 wf_sand_s    = soil_sand_l / 100.0
                 BD_all_s     = BD_all_s * 1000.0

               else                         !OCEAN
                  vf_quartz_mineral_s = -1.0e36
                  vf_gravels_s        = -1.0e36
                  vf_om_s             = -1.0e36
                  vf_sand_s           = -1.0e36
                  wf_gravels_s        = -1.0e36
                  wf_sand_s           = -1.0e36
                  theta_s             = -1.0e36

                  psi_s               = -1.0e36
                  lambda              = -1.0e36
                  k_s                 = -1.0e36

                  VGM_theta_r         = -1.0e36
                  VGM_alpha           = -1.0e36
                  VGM_n               = -1.0e36
                  VGM_L               = -1.0e36

                  csol_s                = -1.0e36
                  k_solids_s            = -1.0e36
                  tksatu_s              = -1.0e36
                  tksatf_s              = -1.0e36
                  tkdry_s               = -1.0e36
                  OM_density_s          = -1.0e36
                  BD_all_s              = -1.0e36
               endif

               ! vf_quartz_mineral_s_l (i,j) = vf_quartz_mineral_s
               ! vf_gravels_s_l        (i,j) = vf_gravels_s
               ! vf_om_s_l             (i,j) = vf_om_s
               ! vf_sand_s_l           (i,j) = vf_sand_s
               ! wf_gravels_s_l        (i,j) = wf_gravels_s
               ! wf_sand_s_l           (i,j) = wf_sand_s
               ! theta_s_l             (i,j) = theta_s

               ! psi_s_l               (i,j) = psi_s
               ! lambda_l              (i,j) = lambda
               ! k_s_l                 (i,j) = k_s

               ! VGM_theta_r_l         (i,j) = VGM_theta_r
               ! VGM_alpha_l           (i,j) = VGM_alpha
               ! VGM_n_l               (i,j) = VGM_n
               ! VGM_L_l               (i,j) = VGM_L

               ! csol_l                (i,j) = csol_s
               ! k_solids_l            (i,j) = k_solids_s
               ! tksatu_l              (i,j) = tksatu_s
               ! tksatf_l              (i,j) = tksatf_s
               ! tkdry_l               (i,j) = tkdry_s

               ! OM_density_l          (i,j) = OM_density_s
               ! BD_all_l              (i,j) = BD_all_s

               vf_quartz    (nsl+1,t) = vf_quartz_mineral_s
               vf_gravels   (nsl+1,t) = vf_gravels_s
               vf_om        (nsl+1,t) = vf_om_s
               vf_sand      (nsl+1,t) = vf_sand_s
               wf_gravels   (nsl+1,t) = wf_gravels_s
               wf_sand      (nsl+1,t) = wf_sand_s
  
               csol         (nsl+1,t) = csol_s
               k_solids     (nsl+1,t) = k_solids_s
               dksatu       (nsl+1,t) = tksatu_s
               dksatf       (nsl+1,t) = tksatf_s
               dkdry        (nsl+1,t) = tkdry_s

               OM_density   (nsl+1,t) = OM_density_s
               BD_all       (nsl+1,t) = BD_all_s
               ! theta_s_l  (nsl+1,t) = theta_s

               ! psi_s_l               (i,j) = psi_s
               ! lambda_l              (i,j) = lambda
               ! k_s_l                 (i,j) = k_s

               ! VGM_theta_r_l         (i,j) = VGM_theta_r
               ! VGM_alpha_l           (i,j) = VGM_alpha
               ! VGM_n_l               (i,j) = VGM_n
               ! VGM_L_l               (i,j) = VGM_L

         enddo
         !    enddo
         ! enddo
#ifdef OPENMP
!$OMP END PARALLEL DO
#endif

         ! print*,'vf_quartz_mineral_s =', minval(vf_quartz_mineral_s_l, mask = vf_quartz_mineral_s_l .gt. -1.0e30), &
         !                                 maxval(vf_quartz_mineral_s_l, mask = vf_quartz_mineral_s_l .gt. -1.0e30)
         ! print*,'vf_gravels_s        =', minval(vf_gravels_s_l,        mask = vf_gravels_s_l        .gt. -1.0e30), &
         !                                 maxval(vf_gravels_s_l,        mask = vf_gravels_s_l        .gt. -1.0e30)
         ! print*,'vf_om_s             =', minval(vf_om_s_l,             mask = vf_om_s_l             .gt. -1.0e30), &
         !                                 maxval(vf_om_s_l,             mask = vf_om_s_l             .gt. -1.0e30)
         ! print*,'vf_sand_s           =', minval(vf_sand_s_l,           mask = vf_sand_s_l           .gt. -1.0e30), &
         !                                 maxval(vf_sand_s_l,           mask = vf_sand_s_l           .gt. -1.0e30)
         ! print*,'wf_gravels_s        =', minval(wf_gravels_s_l,        mask = wf_gravels_s_l        .gt. -1.0e30), &
         !                                 maxval(wf_gravels_s_l,        mask = wf_gravels_s_l        .gt. -1.0e30)
         ! print*,'wf_sand_s           =', minval(wf_sand_s_l,           mask = wf_sand_s_l           .gt. -1.0e30), &
         !                                 maxval(wf_sand_s_l,           mask = wf_sand_s_l           .gt. -1.0e30)

         ! print*,'theta  =', minval(theta_s_l, mask = theta_s_l .gt. -1.0e30), maxval(theta_s_l, mask = theta_s_l .gt. -1.0e30)
         ! print*,'psi    =', minval(psi_s_l,   mask = psi_s_l   .gt. -1.0e30), maxval(psi_s_l,   mask = psi_s_l   .gt. -1.0e30)
         ! print*,'lambda =', minval(lambda_l,  mask = lambda_l  .gt. -1.0e30), maxval(lambda_l,  mask = lambda_l  .gt. -1.0e30)
         ! print*,'ks     =', minval(k_s_l,     mask = k_s_l     .gt. -1.0e30), maxval(k_s_l,     mask = k_s_l     .gt. -1.0e30)
         ! print*,'csol_s   =', minval(csol_l,    mask = csol_l    .gt. -1.0e30), maxval(csol_l,    mask = csol_l    .gt. -1.0e30)
         ! print*,'k_solids_s    =', minval(k_solids_l,    mask = k_solids_l    .gt. -1.0e30), &
         !                         maxval(k_solids_l,    mask = k_solids_l    .gt. -1.0e30)
         ! print*,'tksatu_s =', minval(tksatu_l,  mask = tksatu_l  .gt. -1.0e30), maxval(tksatu_l,  mask = tksatu_l  .gt. -1.0e30)
         ! print*,'tksatf_s =', minval(tksatf_l,  mask = tksatf_l  .gt. -1.0e30), maxval(tksatf_l,  mask = tksatf_l  .gt. -1.0e30)
         ! print*,'tkdry_s  =', minval(tkdry_l,   mask = tkdry_l   .gt. -1.0e30), maxval(tkdry_l,   mask = tkdry_l   .gt. -1.0e30)
         ! print*,'OM_density_s =', minval(OM_density_l,   mask = OM_density_l   .gt. -1.0e30), &
         !                        maxval(OM_density_l,   mask = OM_density_l   .gt. -1.0e30)
         ! print*,'BD_all_s =', minval(BD_all_l,  mask = BD_all_l  .gt. -1.0e30), maxval(BD_all_l,  mask = BD_all_l  .gt. -1.0e30)

         ! print*,'VGM_theta_r =', minval(VGM_theta_r_l, mask = VGM_theta_r_l .gt. -1.0e30), &
         !                         maxval(VGM_theta_r_l, mask = VGM_theta_r_l .gt. -1.0e30)
         ! print*,'VGM_alpha   =', minval(VGM_alpha_l,   mask = VGM_alpha_l   .gt. -1.0e30), &
         !                         maxval(VGM_alpha_l,   mask = VGM_alpha_l   .gt. -1.0e30)
         ! print*,'VGM_n       =', minval(VGM_n_l,       mask = VGM_n_l       .gt. -1.0e30), &
         !                         maxval(VGM_n_l,       mask = VGM_n_l       .gt. -1.0e30)
         ! print*,'VGM_L       =', minval(VGM_L_l,       mask = VGM_L_l       .gt. -1.0e30), &
         !                         maxval(VGM_L_l,       mask = VGM_L_l       .gt. -1.0e30)

! (1) Write out the volumetric fraction of quartz within mineral soil
         ! inquire(iolength=length) vf_quartz_mineral_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/vf_quartz_mineral_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) vf_quartz_mineral_s_l(:,j)
         ! enddo
         ! close(iunit)

! (2) Write out the volumetric fraction of gravels within soil solids
         ! inquire(iolength=length) vf_gravels_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/vf_gravels_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) vf_gravels_s_l(:,j)
         ! enddo
         ! close(iunit)

! (3) Write out the volumetric fraction of organic matter within soil solids
         ! inquire(iolength=length) vf_om_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/vf_om_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) vf_om_s_l(:,j)
         ! enddo
         ! close(iunit)

! (4) Write out the volumetric fraction of sand within soil solids
         ! inquire(iolength=length) vf_sand_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/vf_sand_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) vf_sand_s_l(:,j)
         ! enddo
         ! close(iunit)

! (5) Write out the gravimetric fraction of gravels
         ! inquire(iolength=length) wf_gravels_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/wf_gravels_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) wf_gravels_s_l(:,j)
         ! enddo
         ! close(iunit)

! (6) Write out the gravimetric fraction of sand
         ! inquire(iolength=length) wf_sand_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/wf_sand_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) wf_sand_s_l(:,j)
         ! enddo
         ! close(iunit)

! (7) Write out the saturated water content [cm3/cm3]
         ! inquire(iolength=length) theta_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/theta_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) theta_s_l(:,j)
         ! enddo
         ! close(iunit)

! (8) Write out the matric potential at saturation [cm]
         ! inquire(iolength=length) psi_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/psi_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) psi_s_l(:,j)
         ! enddo
         ! close(iunit)

! (9) Write out the pore size distribution index [dimensionless]
         ! inquire(iolength=length) lambda_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/lambda_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) lambda_l(:,j)
         ! enddo
         ! close(iunit)

! (10) Write out the saturated hydraulic conductivity [cm/day]
         ! inquire(iolength=length) k_s_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/k_s_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) k_s_l(:,j)
         ! enddo
         ! close(iunit)

! (11) Write out the heat capacity of soil solids [J/(m3 K)]
         ! inquire(iolength=length) csol_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/csol_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) csol_l(:,j)
         ! enddo
         ! close(iunit)

! (12) Write out the thermal conductivity of mineral soil [W/m/K]
         ! inquire(iolength=length) k_solids_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/k_solids_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) k_solids_l(:,j)
         ! enddo
         ! close(iunit)

! (13) Write out the thermal conductivity of unfrozen saturated soil [W/m-K]
         ! inquire(iolength=length) tksatu_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/tksatu_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) tksatu_l(:,j)
         ! enddo
         ! close(iunit)

! (14) Write out the thermal conductivity of frozen saturated soil [W/m-K]
         ! inquire(iolength=length) tksatf_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/tksatf_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) tksatf_l(:,j)
         ! enddo
         ! close(iunit)

! (15) Write out the thermal conductivity for dry soil [W/(m-K)]
         ! inquire(iolength=length) tkdry_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/tkdry_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) tkdry_l(:,j)
         ! enddo
         ! close(iunit)

! (16) Write out the VGM's residual moisture content
         ! inquire(iolength=length) VGM_theta_r_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/VGM_theta_r_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) VGM_theta_r_l(:,j)
         ! enddo
         ! close(iunit)

! (17) Write out the VGM's parameter corresponding approximately to the inverse of the air-entry value
         ! inquire(iolength=length) VGM_alpha_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/VGM_alpha_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) VGM_alpha_l(:,j)
         ! enddo
         ! close(iunit)

! (18) Write out the VGM's shape parameter
         ! inquire(iolength=length) VGM_n_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/VGM_n_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) VGM_n_l(:,j)
         ! enddo
         ! close(iunit)

! (19) Write out the VGM's pore-connectivity parameter
         ! inquire(iolength=length) VGM_L_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/VGM_L_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) VGM_L_l(:,j)
         ! enddo
         ! close(iunit)

! (20) Write out the soil organic matter density (kg/m3)
         ! inquire(iolength=length) OM_density_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/OM_density_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) OM_density_l(:,j)
         ! enddo
         ! close(iunit)

! (21) Write out the bulk density of soil (GRAVELS + MINERALS + ORGANIC MATTER) (kg/m3)
         ! inquire(iolength=length) BD_all_l(:,1)
         ! lndname = trim(dir_rawdata)//'RAW_DATA_updated/BD_all_l'//trim(c)
         ! print*,lndname
         ! open(iunit,file=trim(lndname),access='direct',recl=length,form='unformatted',status='unknown')
         ! do j = 1, nlat
         !    write(iunit,rec=j,err=101) BD_all_l(:,j)
         ! enddo
         ! close(iunit)

      ENDDO

      vf_quartz    (1,:) = vf_quartz    (2,:)
      vf_gravels   (1,:) = vf_gravels   (2,:)
      vf_om        (1,:) = vf_om        (2,:)
      vf_sand      (1,:) = vf_sand      (2,:)
      wf_gravels   (1,:) = wf_gravels   (2,:)
      wf_sand      (1,:) = wf_sand      (2,:)
  
      csol         (1,:) = csol         (2,:)
      k_solids     (1,:) = k_solids     (2,:)
      dksatu       (1,:) = dksatu       (2,:)
      dksatf       (1,:) = dksatf       (2,:)
      dkdry        (1,:) = dkdry        (2,:)

      OM_density   (1,:) = OM_density   (2,:)
      BD_all       (1,:) = BD_all       (2,:)

      vf_quartz    (10,:) = vf_quartz    (9,:)
      vf_gravels   (10,:) = vf_gravels   (9,:)
      vf_om        (10,:) = vf_om        (9,:)
      vf_sand      (10,:) = vf_sand      (9,:)
      wf_gravels   (10,:) = wf_gravels   (9,:)
      wf_sand      (10,:) = wf_sand      (9,:)
  
      csol         (10,:) = csol         (9,:)
      k_solids     (10,:) = k_solids     (9,:)
      dksatu       (10,:) = dksatu       (9,:)
      dksatf       (10,:) = dksatf       (9,:)
      dkdry        (10,:) = dkdry        (9,:)

      OM_density   (10,:) = OM_density   (9,:)
      BD_all       (10,:) = BD_all       (9,:)

      WHERE ((vf_gravels(:,:) + vf_sand(:,:)) > 0.4)
         BA_alpha(:,:) = 0.38
         BA_beta(:,:) = 35.0
      ELSEWHERE ((vf_gravels(:,:) + vf_sand(:,:)) > 0.25)
         BA_alpha(:,:) = 0.24
         BA_beta(:,:) = 26.0
      ELSEWHERE
         BA_alpha(:,:) = 0.2
         BA_beta(:,:) = 10.0
      END WHERE
      ! deallocate ( int_soil_grav_l       ,&
      !              int_soil_sand_l       ,&
      !              int_soil_clay_l       ,&
      !              int_soil_oc_l         ,&
      !              int_soil_bd_l          )

      ! deallocate ( vf_quartz_mineral_s_l ,&
      !              vf_gravels_s_l        ,&
      !              vf_om_s_l             ,&
      !              vf_sand_s_l           ,&
      !              wf_gravels_s_l        ,&
      !              wf_sand_s_l           ,&
      !              theta_s_l             ,&
      !              psi_s_l               ,&
      !              lambda_l              ,&
      !              k_s_l                  )

      ! deallocate ( VGM_theta_r_l         ,&
      !              VGM_alpha_l           ,&
      !              VGM_n_l               ,&
      !              VGM_L_l                 )

      ! deallocate ( VGM_theta_r_Rose      ,&
      !              VGM_alpha_Rose        ,&
      !              VGM_n_Rose            ,&
      !              k_s_Rose                )

      ! deallocate ( csol_l                ,&
      !              k_solids_l            ,&
      !              tksatu_l              ,&
      !              tksatf_l              ,&
      !              tkdry_l               ,&
      !              OM_density_l          ,&
      !              BD_all_l)


      ! deallocate ( landtypes )
      ! deallocate ( zsoi, dzsoi, zsoih )

!      go to 1000
!100   print 102,nrow,lndname
!101   print 102,j,lndname
!102   format(' record =',i8,',  error occured on file: ',a50)
!1000  continue
!
!print*,'------ END rd_soil_properties ------'

END SUBROUTINE rd_soil_properties
