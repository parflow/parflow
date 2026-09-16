#include <define.h>

MODULE MOD_Initialize

   IMPLICIT NONE
   SAVE

   ! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: initialize

CONTAINS

   SUBROUTINE initialize (casename, dir_landdata, dir_restart, &
         idate, lc_year, greenwich, numpatch, lulcc_call)

! ======================================================================
! initialization routine for land surface model.
!
! Created by Yongjiu Dai, 09/15/1999
! Revised by Yongjiu Dai, 08/30/2002
! Revised by Yongjiu Dai, 03/2014
!
! ======================================================================
   USE MOD_Precision
   USE MOD_Vars_Global
   USE MOD_Namelist
   USE MOD_SPMD_Task
   !USE MOD_Pixel
   !USE MOD_LandPatch
#ifdef URBAN_MODEL
   USE MOD_LandUrban
   USE MOD_UrbanIniTimeVariable
   USE MOD_UrbanReadin
   USE MOD_Urban_LAIReadin
   USE MOD_Urban_Albedo
#endif
   USE MOD_Const_Physical
   USE MOD_Vars_TimeInvariants
   USE MOD_Vars_TimeVariables
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_LandPFT
   USE MOD_Vars_PFTimeInvariants
   USE MOD_Vars_PFTimeVariables
#endif
   USE MOD_Const_LC
   USE MOD_Const_PFT
   USE MOD_TimeManager

   !USE MOD_Grid
   !USE MOD_DataType
   !USE MOD_NetCDFSerial
   !USE MOD_NetCDFVector
   !USE MOD_NetCDFBlock
#ifdef RangeCheck
   USE MOD_RangeCheck
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
   USE MOD_Hydro_SoilFunction
#endif
   !USE MOD_SpatialMapping
#ifdef CatchLateralFlow
   USE MOD_Mesh
   USE MOD_LandHRU
   USE MOD_LandPatch
   USE MOD_ElmVector
   USE MOD_HRUVector
   USE MOD_ElementNeighbour
   USE MOD_Catch_HillslopeNetwork
   USE MOD_Catch_RiverLakeNetwork
#endif
#ifdef CROP
   USE MOD_CropReadin
#endif
   USE MOD_LAIEmpirical
   !USE MOD_LAIReadin
   USE MOD_OrbCoszen
   !USE MOD_DBedrockReadin
   USE MOD_HtopReadin
   USE MOD_IniTimeVariable
   USE MOD_LakeDepthReadin
   !USE MOD_PercentagesPFTReadin
   !USE MOD_SoilParametersReadin
   USE MOD_SoilColorRefl

   IMPLICIT NONE

   ! ----------------------------------------------------------------------
   character(len=*), intent(in) :: casename      ! case name
   character(len=*), intent(in) :: dir_landdata
   character(len=*), intent(in) :: dir_restart
   !this three could be passed without values as we don't use them currently
   integer, intent(in)    :: idate(3)   ! year, julian day, seconds of the starting time
   integer, intent(in)    :: lc_year    ! year, land cover year
   logical, intent(in)    :: greenwich  ! true: greenwich time, false: local time
   logical, optional, intent(in) :: lulcc_call   ! whether it is a lulcc CALL

   ! ------------------------ local variables -----------------------------
   real(r8) :: rlon, rlat

   ! for SOIL INIT of water, temperature, snow depth
   logical :: use_soilini
   logical :: use_cnini
   logical :: use_snowini

   character(len=256) :: fsoildat
   character(len=256) :: fsnowdat
   character(len=256) :: fcndat
   character(len=256) :: ftopo, lndname
   !type(grid_type) :: gsoil
   !type(grid_type) :: gsnow
   !type(grid_type) :: gcn
   !type(spatial_mapping_type) :: ms2p
   !type(spatial_mapping_type) :: mc2p
   !type(spatial_mapping_type) :: mc2f
   !type(spatial_mapping_type) :: msoil2p, msnow2p

   integer  :: nl_soil_ini
   real(r8) :: missing_value

   real(r8), allocatable :: soil_z(:)

   !type(block_data_real8_2d) :: snow_d_grid
   !type(block_data_real8_3d) :: soil_t_grid
   !type(block_data_real8_3d) :: soil_w_grid
   !type(block_data_real8_2d) :: zwt_grid

   !type(block_data_real8_3d) :: litr1c_grid
   !type(block_data_real8_3d) :: litr2c_grid
   !type(block_data_real8_3d) :: litr3c_grid
   !type(block_data_real8_3d) :: cwdc_grid
   !type(block_data_real8_3d) :: soil1c_grid
   !type(block_data_real8_3d) :: soil2c_grid
   !type(block_data_real8_3d) :: soil3c_grid
   !type(block_data_real8_3d) :: litr1n_grid
   !type(block_data_real8_3d) :: litr2n_grid
   !type(block_data_real8_3d) :: litr3n_grid
   !type(block_data_real8_3d) :: cwdn_grid
   !type(block_data_real8_3d) :: soil1n_grid
   !type(block_data_real8_3d) :: soil2n_grid
   !type(block_data_real8_3d) :: soil3n_grid
   !type(block_data_real8_3d) :: sminn_grid
   !type(block_data_real8_3d) :: smin_nh4_grid
   !type(block_data_real8_3d) :: smin_no3_grid
   !type(block_data_real8_2d) :: leafc_grid
   !type(block_data_real8_2d) :: leafc_storage_grid
   !type(block_data_real8_2d) :: frootc_grid
   !type(block_data_real8_2d) :: frootc_storage_grid
   !type(block_data_real8_2d) :: livestemc_grid
   !type(block_data_real8_2d) :: deadstemc_grid
   !type(block_data_real8_2d) :: livecrootc_grid
   !type(block_data_real8_2d) :: deadcrootc_grid

   real(r8), allocatable :: snow_d(:)
   real(r8), allocatable :: soil_t(:,:)
   real(r8), allocatable :: soil_w(:,:)
   logical , allocatable :: validval(:)

   real(r8), allocatable :: litr1c_vr(:,:)
   real(r8), allocatable :: litr2c_vr(:,:)
   real(r8), allocatable :: litr3c_vr(:,:)
   real(r8), allocatable :: cwdc_vr  (:,:)
   real(r8), allocatable :: soil1c_vr(:,:)
   real(r8), allocatable :: soil2c_vr(:,:)
   real(r8), allocatable :: soil3c_vr(:,:)
   real(r8), allocatable :: litr1n_vr(:,:)
   real(r8), allocatable :: litr2n_vr(:,:)
   real(r8), allocatable :: litr3n_vr(:,:)
   real(r8), allocatable :: cwdn_vr  (:,:)
   real(r8), allocatable :: soil1n_vr(:,:)
   real(r8), allocatable :: soil2n_vr(:,:)
   real(r8), allocatable :: soil3n_vr(:,:)
   real(r8), allocatable :: min_nh4_vr(:,:)
   real(r8), allocatable :: min_no3_vr(:,:)
   real(r8), allocatable :: leafcin_p(:)
   real(r8), allocatable :: leafc_storagein_p(:)
   real(r8), allocatable :: frootcin_p(:)
   real(r8), allocatable :: frootc_storagein_p(:)
   real(r8), allocatable :: livestemcin_p(:)
   real(r8), allocatable :: deadstemcin_p(:)
   real(r8), allocatable :: livecrootcin_p(:)
   real(r8), allocatable :: deadcrootcin_p(:)
   ! for SOIL Water INIT by using water table depth
   logical  :: use_wtd

   character(len=256) :: fwtd
   !type(grid_type)    :: gwtd
   !type(block_data_real8_2d)  :: wtd_xy  ! [m]
   !type(spatial_mapping_type) :: m_wtd2p

   real(r8) :: zwtmm
   real(r8) :: zc_soimm(1:nl_soil)
   real(r8) :: zi_soimm(0:nl_soil)
   real(r8) :: vliq_r  (1:nl_soil)
#ifdef Campbell_SOIL_MODEL
   integer, parameter :: nprms = 1
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
   integer, parameter :: nprms = 5
#endif
   real(r8) :: prms(nprms, 1:nl_soil)

   ! CoLM soil layer thickiness and depths
   real(r8), allocatable :: z_soisno (:,:)
   real(r8), allocatable :: dz_soisno(:,:)

   real(r8) :: calday                    ! Julian cal day (1.xx to 365.xx)
   integer  :: year, jday                ! Julian day and seconds
   integer  :: month, mday
   character(len=4) :: cyear

   integer  :: i,j,ipatch,nsl,hs,he,ps,pe,ivt,m, u  ! indices
   real(r8) :: totalvolume

   integer :: Julian_8day
   integer :: ltyp

#ifdef BGC
   real(r8) f_s1s2 (1:nl_soil)
   real(r8) f_s1s3 (1:nl_soil)
   real(r8) rf_s1s2(1:nl_soil)
   real(r8) rf_s1s3(1:nl_soil)
   real(r8) f_s2s1
   real(r8) f_s2s3
   real(r8) t
#endif

   integer  :: txt_id
   real(r8) :: vic_b_infilt_, vic_Dsmax_, vic_Ds_, vic_Ws_, vic_c_

   integer, intent(in) :: numpatch

! --------------------------------------------------------------------
! Allocates memory for CoLM 1d [numpatch] variables
! --------------------------------------------------------------------

      !CALL allocate_TimeInvariants
      !CALL allocate_TimeVariables

! ---------------------------------------------------------------
! 1. INITIALIZE TIME INVARIANT VARIABLES
! ---------------------------------------------------------------

      IF (p_is_worker) THEN

         !IF (numpatch > 0) THEN
         !   patchclass = landpatch%settyp  !@CY
         !   patchmask  = .true.
         !ENDIF

         DO ipatch = 1, numpatch

            m = patchclass(ipatch)  !@CY: pfb readin
            patchtype(ipatch) = patchtypes(m)

            !     ***** patch mask setting *****
            ! ---------------------------------------

            IF (DEF_URBAN_ONLY .and. m.ne.URBAN) THEN
            !   patchmask(ipatch) = .false.
               CYCLE
            ENDIF

         ENDDO

         !CALL landpatch%get_lonlat_radian (patchlonr, patchlatr)

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
         IF (numpft > 0) pftclass = landpft%settyp
#endif

      ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL pct_readin (dir_landdata, lc_year)
#endif

! ------------------------------------------
! 1.1 Ponding water
! ------------------------------------------
      IF(DEF_USE_BEDROCK)THEN
         !CALL dbedrock_readin (dir_landdata)
      ENDIF

      IF (p_is_worker) THEN
         IF (numpatch > 0) THEN
            wdsrf(:) = 0._r8

            wetwat(:) = 0._r8
            WHERE (patchtype == 2) wetwat = 200._r8 ! for wetland
         ENDIF
      ENDIF
! ------------------------------------------
! 1.2 Lake depth and layers' thickness
! ------------------------------------------
      CALL lakedepth_readin (dir_landdata, lc_year, numpatch)

! ...............................................................
! 1.3 Read in the soil parameters of the patches of the gridcells
! ...............................................................

      !CALL soil_parameters_readin (dir_landdata, lc_year)
      IF (DEF_SOIL_REFL_SCHEME .eq. 1) THEN
         IF (p_is_worker) THEN
            DO ipatch = 1, numpatch
               !m = landpatch%settyp(ipatch)
               !patchclass = landpatch%settyp
               m = patchclass(ipatch)
               CALL soil_color_refl(m,soil_s_v_alb(ipatch),soil_d_v_alb(ipatch),&
                  soil_s_n_alb(ipatch),soil_d_n_alb(ipatch))
            ENDDO
         ENDIF
      ENDIF
      !@CY: albedo
      !IF (p_is_worker) THEN
      !   IF (numpatch > 0) THEN

      !      BVIC(:,:) = -1.0

      !      DO ipatch = 1, numpatch
      !         DO i = 1, nl_soil
      !            !soil_solids_fractions.F90; 
      !            !BVIC is the b parameter in Fraction of saturated soil in a grid calculated by VIC
      !            !Modified from NoahmpTable.TBL in NoahMP
      !            !SEE: a near-global, high resolution land surface parameter dataset for the variable infiltration capacity model
      !            !soil type (USDA)      1        2       3           4          5         6        7          8        9         10        11        12     |   13        14        15        16        17        18      19
      !            !BVIC          =    0.050,    0.080,    0.090,    0.250,    0.150,    0.180,    0.200,    0.220,    0.230,    0.250,    0.280,    0.300,   | 0.260,    0.000,    1.000,    1.000,    1.000,    0.350,    0.150
      !            ! this should be revised using usda soil type
      !            IF (vf_quartz(i,ipatch) >= 0.95) THEN! sand 
      !               BVIC(i,ipatch)=0.050
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.85 .and. vf_quartz(i,ipatch) < 0.95) THEN ! loamy sand; soil types 1 
      !               BVIC(i,ipatch)=0.080
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.69 .and. vf_quartz(i,ipatch) <  0.85) THEN  !Sandy loam; soil types 3
      !               BVIC(i,ipatch)=0.09
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.61 .and. vf_quartz(i,ipatch) <  0.69) THEN  !Sandy clay loam; soil types 7
      !               BVIC(i,ipatch)=0.20
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.50 .and. vf_quartz(i,ipatch) <  0.61) THEN  !Sandy clay; soil types 10
      !               BVIC(i,ipatch)=0.25
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.41 .and. vf_quartz(i,ipatch) <  0.50) THEN  !Silt; soil types 5
      !               BVIC(i,ipatch)=0.150
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.25 .and. vf_quartz(i,ipatch) <  0.41 .and. wf_sand(i,ipatch)<=0.20) THEN  !Silty clay loam ; soil types 8
      !               BVIC(i,ipatch)=0.220
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.25 .and. vf_quartz(i,ipatch) <  0.41 .and. wf_sand(i,ipatch)>0.20) THEN    !Clay; soil types 12 
      !               BVIC(i,ipatch)=0.30          
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.19 .and. vf_quartz(i,ipatch) <  0.25) THEN  !Loam; soil types 6
      !               BVIC(i,ipatch)=0.180 
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.09 .and. vf_quartz(i,ipatch) <  0.19) THEN  !Clay loam; soil types 9
      !               BVIC(i,ipatch)=0.230
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.08 .and. vf_quartz(i,ipatch) <  0.09) THEN  !Silty clay; soil types 11 
      !               BVIC(i,ipatch)=0.280
      !            ELSEIF (vf_quartz(i,ipatch) >= 0.0 .and. vf_quartz(i,ipatch) <  0.08) THEN  !Silt loam; soil types 4
      !               BVIC(i,ipatch)=0.100
      !            ELSE
      !               BVIC(i,ipatch)=1.0
      !            ENDIF
      !         ENDDO
      !      ENDDO
      !   ENDIF
      !ENDIF




#ifdef vanGenuchten_Mualem_SOIL_MODEL
      IF (p_is_worker) THEN
         IF (numpatch > 0) THEN

            psi0(:,:) = -1.0

            DO ipatch = 1, numpatch
               DO i = 1, nl_soil
                  CALL get_derived_parameters_vGM ( &
                     psi0(i,ipatch), alpha_vgm(i,ipatch), n_vgm(i,ipatch), &
                     sc_vgm(i,ipatch), fc_vgm(i,ipatch))
               ENDDO
            ENDDO
         ENDIF
      ENDIF
#endif

! ...............................................................
! 1.4 Plant time-invariant variables
! ...............................................................

      ! read global tree top height from nc file
      CALL HTOP_readin (dir_landdata, lc_year, numpatch) !! more work to pass htoplc to this subroutine

#ifdef URBAN_MODEL
      CALL Urban_readin (dir_landdata, lc_year)
#endif
! ................................
! 1.5 Initialize topography data
! ................................
#ifdef SinglePoint
      topoelv(:) = SITE_topography
      topostd(:) = SITE_topostd
#else
      !write(cyear,'(i4.4)') lc_year
      !ftopo = trim(dir_landdata)//'/topography/'//trim(cyear)//'/topography_patches.nc'
      !CALL ncio_read_vector (ftopo, 'topography_patches', landpatch, topoelv)
      !ftopo = trim(dir_landdata)//'/topography/'//trim(cyear)//'/topostd_patches.nc'
      !CALL ncio_read_vector (ftopo, 'topostd_patches', landpatch, topostd)
#endif
! ......................................
! 1.5 Initialize topography factor data
! ......................................
#ifdef SinglePoint
      slp_type_patches(:,1) = SITE_slp_type
      asp_type_patches(:,1) = SITE_asp_type
      area_type_patches(:,1) = SITE_area_type
      svf_patches(:) = SITE_svf
      cur_patches(:) = SITE_cur
      sf_lut_patches(:,:,1) = SITE_sf_lut
#else
      IF (DEF_USE_Forcing_Downscaling) THEN  !set false
         !lndname = trim(DEF_dir_landdata) // '/topography/'//trim(cyear)//'/slp_type_patches.nc'             ! slope
         !CALL ncio_read_vector (lndname, 'slp_type_patches', num_type, landpatch, slp_type_patches)
         !lndname = trim(DEF_dir_landdata) // '/topography/'//trim(cyear)//'/svf_patches.nc'               ! sky view factor
         !CALL ncio_read_vector (lndname, 'svf_patches', landpatch, svf_patches)
         !lndname = trim(DEF_dir_landdata) // '/topography/'//trim(cyear)//'/asp_type_patches.nc'            ! aspect
         !CALL ncio_read_vector (lndname, 'asp_type_patches', num_type, landpatch, asp_type_patches)
         !lndname = trim(DEF_dir_landdata) // '/topography/'//trim(cyear)//'/area_type_patches.nc'         ! area percent
         !CALL ncio_read_vector (lndname, 'area_type_patches', num_type, landpatch, area_type_patches)
         !lndname = trim(DEF_dir_landdata) // '/topography/'//trim(cyear)//'/sf_lut_patches.nc'        ! shadow mask
         !CALL ncio_read_vector (lndname, 'sf_lut_patches', num_azimuth, num_zenith, landpatch, sf_lut_patches)
         !lndname = trim(DEF_dir_landdata) // '/topography/'//trim(cyear)//'/cur_patches.nc'               ! curvature
         !CALL ncio_read_vector (lndname, 'cur_patches', landpatch, cur_patches)
       ENDIF
#endif
! ................................
! 1.6 Initialize TUNABLE constants
! ................................
      zlnd   = 0.01    !Roughness length for soil [m]
      zsno   = 0.0024  !Roughness length for snow [m]
      csoilc = 0.004   !Drag coefficient for soil under canopy [-]
      dewmx  = 0.1     !maximum dew
      wtfact = 0.38    !Maximum saturated fraction (global mean; see Niu et al., 2005)
      capr   = 0.34    !Tuning factor to turn first layer T into surface T
      cnfac  = 0.5     !Crank Nicholson factor between 0 and 1
      ssi    = 0.033   !Irreducible water saturation of snow
      wimp   = 0.05    !Water impremeable if porosity less than wimp
      pondmx = 10.0    !Ponding depth (mm)
      smpmax = -1.5e5  !Wilting point potential in mm
      smpmin = -1.e8   !Restriction for min of soil poten. (mm)
      trsmx0 = 2.e-4   !Max transpiration for moist soil+100% veg. [mm/s]
      tcrit  = 2.5     !critical temp. to determine rain or snow
      wetwatmax = 200.0 !maximum wetland water (mm)

!      IF (DEF_Runoff_SCHEME == 1) THEN
!         IF (p_is_master) THEN
!            txt_id = 111
!            open(txt_id, file=trim(DEF_file_VIC_para), status='old', form='formatted')
!            READ(txt_id, *)
!            READ(txt_id, *) vic_b_infilt_, vic_Dsmax_, vic_Ds_, vic_Ws_, vic_c_
!            close(txt_id)
!         ENDIF

!#ifdef USEMPI
!         CALL mpi_bcast (vic_b_infilt_, 1, MPI_REAL8, p_root, p_comm_glb, p_err)
!         CALL mpi_bcast (vic_Dsmax_   , 1, MPI_REAL8, p_root, p_comm_glb, p_err)
!         CALL mpi_bcast (vic_Ds_      , 1, MPI_REAL8, p_root, p_comm_glb, p_err)
!         CALL mpi_bcast (vic_Ws_      , 1, MPI_REAL8, p_root, p_comm_glb, p_err)
!         CALL mpi_bcast (vic_c_       , 1, MPI_REAL8, p_root, p_comm_glb, p_err)
!#endif

!         IF (p_is_worker) THEN
!            IF (numpatch > 0) THEN
!               vic_b_infilt = vic_b_infilt_
!               vic_Dsmax    = vic_Dsmax_
!               vic_Ds       = vic_Ds_
!               vic_Ws       = vic_Ws_
!               vic_c        = vic_c_
!            ENDIF
!         ENDIF
!      ELSE
!         IF (p_is_worker) THEN
!            IF (numpatch > 0) THEN
!               vic_b_infilt = 0
!               vic_Dsmax    = 0
!               vic_Ds       = 0
!               vic_Ws       = 0
!               vic_c        = 0
!            ENDIF
!         ENDIF
!      ENDIF

#ifdef BGC
   ! bgc constant
      i_met_lit = 1
      i_cel_lit = 2
      i_lig_lit = 3
      i_cwd     = 4
      i_soil1   = 5
      i_soil2   = 6
      i_soil3   = 7
      i_atm     = 0

      donor_pool    = (/i_met_lit, i_cel_lit, i_lig_lit, i_soil1, i_cwd    , i_cwd    , i_soil1, i_soil2, i_soil2, i_soil3/)
      receiver_pool = (/i_soil1  , i_soil1  , i_soil2  , i_soil2, i_cel_lit, i_lig_lit, i_soil3, i_soil1, i_soil3, i_soil1/)
      am = 0.02_r8
      floating_cn_ratio = (/.true., .true., .true., .true., .false. ,.false., .false./)
      initial_cn_ratio  = (/90._r8, 90._r8, 90._r8, 90._r8,    8._r8, 11._r8,  11._r8/)      ! 1:ndecomp_pools

      f_s2s1 = 0.42_r8/(0.45_r8)
      f_s2s3 = 0.03_r8/(0.45_r8)
      IF (p_is_worker) THEN
         IF(numpatch > 0)THEN
            DO j=1,nl_soil
               DO i = 1, numpatch
                  t = 0.85_r8 - 0.68_r8 * 0.01_r8 * (100._r8 - 50._r8)
                  f_s1s2 (j) = 1._r8 - .004_r8 / (1._r8 - t)
                  f_s1s3 (j) = .004_r8 / (1._r8 - t)
                  rf_s1s2(j) = t
                  rf_s1s3(j) = t
                  rf_decomp(j,:,i)  = (/0.55_r8, 0.5_r8, 0.5_r8, rf_s1s2(j), 0._r8  , 0._r8  , rf_s1s3(j), 0.55_r8, 0.55_r8, 0.55_r8/)
                  pathfrac_decomp(j,:,i) = (/1.0_r8 ,1.0_r8 , 1.0_r8, f_s1s2(j) , 0.76_r8, 0.24_r8, f_s1s3(j) , f_s2s1 , f_s2s3 , 1._r8/)
               ENDDO
            ENDDO
         ENDIF
      ENDIF

      is_cwd            = (/.false.,.false.,.false.,.true. ,.false.,.false.,.false./)
      is_litter         = (/.true. ,.true. ,.true. ,.false.,.false.,.false.,.false./)
      is_soil           = (/.false.,.false.,.false.,.false.,.true. ,.true. ,.true./)
      cmb_cmplt_fact = (/0.5_r8,0.25_r8/)

      nitrif_n2o_loss_frac = 6.e-4 !fraction of N lost as N2O in nitrification (Li et al., 2000)
      dnp    = 0.01_r8
      bdnr   = 0.5_r8
      compet_plant_no3 = 1._r8
      compet_plant_nh4 = 1._r8
      compet_decomp_no3 = 1._r8
      compet_decomp_nh4 = 1._r8
      compet_denit = 1._r8
      compet_nit = 1._r8
      surface_tension_water = 0.073
      rij_kro_a     = 1.5e-10_r8
      rij_kro_alpha = 1.26_r8
      rij_kro_beta  = 0.6_r8
      rij_kro_gamma = 0.6_r8
      rij_kro_delta = 0.85_r8
      IF(DEF_USE_NITRIF)THEN
         nfix_timeconst = 10._r8
      ELSE
         nfix_timeconst = 0._r8
      ENDIF
      organic_max        = 130
      d_con_g21          = 0.1759_r8
      d_con_g22          = 0.00117_r8
      d_con_w21          = 1.172_r8
      d_con_w22          = 0.03443_r8
      d_con_w23          = 0.0005048_r8
      denit_resp_coef    = 0.1_r8
      denit_resp_exp     = 1.3_r8
      denit_nitrate_coef = 1.15_r8
      denit_nitrate_exp  = 0.57_r8
      k_nitr_max         = 1.1574074e-06_r8
      Q10       = 1.5_r8
      froz_q10  = 1.5_r8
      tau_l1    = 1._r8/18.5_r8
      tau_l2_l3 = 1._r8/4.9_r8
      tau_s1    = 1._r8/7.3_r8
      tau_s2    = 1._r8/0.2_r8
      tau_s3    = 1._r8/.0045_r8
      tau_cwd   = 1._r8/0.3_r8
      lwtop     = 0.7_r8/31536000.0_r8

      som_adv_flux               = 0._r8
      som_diffus                 = 3.170979198376459e-12_r8
      cryoturb_diffusion_k       = 1.585489599188229e-11_r8
      max_altdepth_cryoturbation = 2._r8
      max_depth_cryoturb         = 3._r8

      br              = 2.525e-6_r8
      br_root         = 0.83e-6_r8

      fstor2tran      = 0.5
      ndays_on        = 30
      ndays_off       = 15
      crit_dayl       = 39300
      crit_onset_fdd  = 15
      crit_onset_swi  = 15
      crit_offset_fdd = 15
      crit_offset_swi = 15
      soilpsi_on      = -0.6
      soilpsi_off     = -0.8

      ! constant for fire module
      occur_hi_gdp_tree        = 0.39_r8
      lfuel                    = 75._r8
      ufuel                    = 650._r8
      cropfire_a1              = 0.3_r8
      borealat                 = 40._r8/(4.*atan(1.))
      troplat                  = 23.5_r8/(4.*atan(1.))
      non_boreal_peatfire_c    = 0.001_r8
      boreal_peatfire_c        = 4.2e-5_r8
      rh_low                   = 30.0_r8
      rh_hgh                   = 80.0_r8
      bt_min                   = 0.3_r8
      bt_max                   = 0.7_r8
      pot_hmn_ign_counts_alpha = 0.0035_r8
      g0_fire                  = 0.05_r8

      sf     = 0.1_r8
      sf_no3 = 1._r8
#endif

! ...............................................
! 1.6 Write out as a restart file [histTimeConst]
! ...............................................

#ifdef RangeCheck
      CALL check_TimeInvariants ()
#endif

      !CALL WRITE_TimeInvariants (lc_year, casename, dir_restart)

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

      IF (p_is_master) write (6,*) ('Successfully Initialize the Land Time-Invariants')

! ----------------------------------------------------------------------
! [2] INITIALIZE TIME-VARYING VARIABLES
! as subgrid vectors of length [numpatch]
! initial run: create the time-varying variables based on :
!              i) observation (NOT CODING CURRENTLY), or
!             ii) some already-known information (NO CODING CURRENTLY), or
!            iii) arbitrarily
! continuation run: time-varying data read in from restart file
! ----------------------------------------------------------------------

! 2.1 current time of model run
! ............................

      CALL initimetype(greenwich)

      IF (p_is_master) THEN
         IF(.not. greenwich)THEN
            write(*,*) char(27)//"[1;31m"//"Notice: greenwich false, local time is used."//char(27)//"[0m"
         ENDIF
      ENDIF


! ................................
! 2.2 cosine of solar zenith angle
! ................................
      calday = calendarday(idate)
      IF (p_is_worker) THEN
         DO i = 1, numpatch
            coszen(i) = orb_coszen(calday, patchlonr(i), patchlatr(i))
         ENDDO
      ENDIF

! ...........................................
!2.3 READ in or GUSSES land state information
! ...........................................

      ! for SOIL INIT of water, temperature
      IF (DEF_USE_SoilInit) THEN

         fsoildat = DEF_file_SoilInit
         IF (p_is_master) THEN
            inquire (file=trim(fsoildat), exist=use_soilini)
         ENDIF
#ifdef USEMPI
         CALL mpi_bcast (use_soilini, 1, MPI_LOGICAL, p_root, p_comm_glb, p_err)
#endif

         IF (use_soilini) THEN

            ! soil layer depth (m)
            !CALL ncio_read_bcast_serial (fsoildat, 'soildepth', soil_z)
           
            nl_soil_ini = size(soil_z)
            
            !CALL gsoil%define_from_file (fsoildat, latname = 'lat', lonname = 'lon')
         
            CALL julian2monthday (idate(1), idate(2), month, mday)
         
            IF (p_is_master) THEN
               !CALL ncio_get_attr (fsoildat, 'zwt', 'missing_value', missing_value)
            ENDIF
#ifdef USEMPI
            CALL mpi_bcast (missing_value, 1, MPI_REAL8, p_root, p_comm_glb, p_err)
#endif

            IF (p_is_io) THEN
               !! soil layer temperature (K)
               !CALL allocate_block_data  (gsoil, soil_t_grid, nl_soil_ini)
               !CALL ncio_read_block_time (fsoildat, 'soiltemp', &
               !   gsoil, nl_soil_ini, month, soil_t_grid)
               !! soil layer wetness (-)
               !CALL allocate_block_data  (gsoil, soil_w_grid, nl_soil_ini)
               !CALL ncio_read_block_time (fsoildat, 'soilwat', &
               !   gsoil, nl_soil_ini, month, soil_w_grid)
               !! water table depth (m)
               !CALL allocate_block_data (gsoil, zwt_grid)
               !CALL ncio_read_block_time (fsoildat, 'zwt', gsoil, month, zwt_grid)
            ENDIF

            IF (p_is_worker) THEN
               IF (numpatch > 0) THEN
                  allocate (soil_t (nl_soil_ini,numpatch))
                  allocate (soil_w (nl_soil_ini,numpatch))
                  allocate (validval (numpatch))
               ENDIF
            ENDIF

            !CALL msoil2p%build_arealweighted (gsoil, landpatch)
            !CALL msoil2p%set_missing_value   (zwt_grid, missing_value, validval)

            !CALL msoil2p%grid2pset (soil_t_grid, nl_soil_ini, soil_t)
            !CALL msoil2p%grid2pset (soil_w_grid, nl_soil_ini, soil_w)
            !CALL msoil2p%grid2pset (zwt_grid, zwt)

            IF (p_is_worker) THEN
               DO i = 1, numpatch
                  IF (.not. validval(i)) THEN
                     IF (patchtype(i) == 3) THEN
                        soil_t(:,i) = 250.
                     ELSE
                        soil_t(:,i) = 280.
                     ENDIF

                     soil_w(:,i) = 1.
                     zwt(i) = 0.

                  ENDIF
               ENDDO
            ENDIF

            IF (allocated(validval)) deallocate(validval)

         ENDIF

      ELSE
         use_soilini = .false.
      ENDIF

      IF (.not. use_soilini) THEN
         !! not used, just for filling arguments
         IF (p_is_worker) THEN
            allocate (soil_z (nl_soil))
            allocate (soil_t (nl_soil,numpatch))
            allocate (soil_w (nl_soil,numpatch))
         ENDIF
      ENDIF

#ifdef BGC
      IF (DEF_USE_CN_INIT) THEN
         fcndat = DEF_file_cn_init
         IF (p_is_master) THEN
            inquire (file=trim(fcndat), exist=use_cnini)
            IF (use_cnini) THEN
               write(*,'(/, 2A)') 'Use carbon & nitrogen and derived equilibrium state ' &
                  // ' to initialize soil & vegetation biogeochemistry: ', trim(fcndat)
            ELSE
               write(*,*) 'no initial data for biogeochemistry: ',trim(fcndat)
            ENDIF
         ENDIF
#ifdef USEMPI
         CALL mpi_bcast (use_cnini, 1, MPI_LOGICAL, p_root, p_comm_glb, p_err)
#endif

         IF (use_cnini) THEN

            CALL gcn%define_from_file (fcndat,"lat","lon")
            CALL mc2p%build_arealweighted (gcn, landpatch)
            CALL mc2f%build_arealweighted (gcn, landpft)

            IF (p_is_io) THEN
               ! soil layer litter & carbon (gC m-3)
               CALL allocate_block_data (gcn, litr1c_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'litr1c_vr', gcn, nl_soil, litr1c_grid)

               CALL allocate_block_data (gcn, litr2c_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'litr2c_vr', gcn, nl_soil, litr2c_grid)

               CALL allocate_block_data (gcn, litr3c_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'litr3c_vr', gcn, nl_soil, litr3c_grid)

               CALL allocate_block_data (gcn, cwdc_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'cwdc_vr', gcn, nl_soil  , cwdc_grid)

               CALL allocate_block_data (gcn, soil1c_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil1c_vr', gcn, nl_soil, soil1c_grid)

               CALL allocate_block_data (gcn, soil2c_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil2c_vr', gcn, nl_soil, soil2c_grid)

               CALL allocate_block_data (gcn, soil3c_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil3c_vr', gcn, nl_soil, soil3c_grid)

               ! soil layer litter & nitrogen (gN m-3)
               CALL allocate_block_data (gcn, litr1n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'litr1n_vr', gcn, nl_soil, litr1n_grid)

               CALL allocate_block_data (gcn, litr2n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'litr2n_vr', gcn, nl_soil, litr2n_grid)

               CALL allocate_block_data (gcn, litr3n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'litr3n_vr', gcn, nl_soil, litr3n_grid)

               CALL allocate_block_data (gcn, cwdn_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'cwdn_vr', gcn, nl_soil  , cwdn_grid)

               CALL allocate_block_data (gcn, soil1n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil1n_vr', gcn, nl_soil, soil1n_grid)

               CALL allocate_block_data (gcn, soil2n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil2n_vr', gcn, nl_soil, soil2n_grid)

               CALL allocate_block_data (gcn, soil3n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil3n_vr', gcn, nl_soil, soil3n_grid)

               CALL allocate_block_data (gcn, soil3n_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'soil3n_vr', gcn, nl_soil, soil3n_grid)

               CALL allocate_block_data (gcn, smin_nh4_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'smin_nh4_vr', gcn, nl_soil, smin_nh4_grid)

               CALL allocate_block_data (gcn, smin_no3_grid, nl_soil)
               CALL ncio_read_block (fcndat, 'smin_no3_vr', gcn, nl_soil, smin_no3_grid)

               CALL allocate_block_data (gcn, leafc_grid)
               CALL ncio_read_block (fcndat, 'leafc', gcn, leafc_grid)

               CALL allocate_block_data (gcn, leafc_storage_grid)
               CALL ncio_read_block (fcndat, 'leafc_storage', gcn, leafc_storage_grid)

               CALL allocate_block_data (gcn, frootc_grid)
               CALL ncio_read_block (fcndat, 'frootc', gcn, frootc_grid)

               CALL allocate_block_data (gcn, frootc_storage_grid)
               CALL ncio_read_block (fcndat, 'frootc_storage', gcn, frootc_storage_grid)

               CALL allocate_block_data (gcn, livestemc_grid)
               CALL ncio_read_block (fcndat, 'livestemc', gcn, livestemc_grid)

               CALL allocate_block_data (gcn, deadstemc_grid)
               CALL ncio_read_block (fcndat, 'deadstemc', gcn, deadstemc_grid)

               CALL allocate_block_data (gcn, livecrootc_grid)
               CALL ncio_read_block (fcndat, 'livecrootc', gcn, livecrootc_grid)

               CALL allocate_block_data (gcn, deadcrootc_grid)
               CALL ncio_read_block (fcndat, 'deadcrootc', gcn, deadcrootc_grid)

            ENDIF

            IF (p_is_worker) THEN

               allocate (litr1c_vr (nl_soil,numpatch))
               allocate (litr2c_vr (nl_soil,numpatch))
               allocate (litr3c_vr (nl_soil,numpatch))
               allocate (cwdc_vr   (nl_soil,numpatch))
               allocate (soil1c_vr (nl_soil,numpatch))
               allocate (soil2c_vr (nl_soil,numpatch))
               allocate (soil3c_vr (nl_soil,numpatch))
               allocate (litr1n_vr (nl_soil,numpatch))
               allocate (litr2n_vr (nl_soil,numpatch))
               allocate (litr3n_vr (nl_soil,numpatch))
               allocate (cwdn_vr   (nl_soil,numpatch))
               allocate (soil1n_vr (nl_soil,numpatch))
               allocate (soil2n_vr (nl_soil,numpatch))
               allocate (soil3n_vr (nl_soil,numpatch))
               allocate (min_nh4_vr(nl_soil,numpatch))
               allocate (min_no3_vr(nl_soil,numpatch))
               allocate (leafcin_p (numpft))
               allocate (leafc_storagein_p (numpft))
               allocate (frootcin_p (numpft))
               allocate (frootc_storagein_p (numpft))
               allocate (livestemcin_p (numpft))
               allocate (deadstemcin_p (numpft))
               allocate (livecrootcin_p (numpft))
               allocate (deadcrootcin_p (numpft))

            ENDIF

            CALL mc2p%grid2pset (litr1c_grid, nl_soil, litr1c_vr)
            CALL mc2p%grid2pset (litr2c_grid, nl_soil, litr2c_vr)
            CALL mc2p%grid2pset (litr3c_grid, nl_soil, litr3c_vr)
            CALL mc2p%grid2pset (cwdc_grid  , nl_soil, cwdc_vr  )
            CALL mc2p%grid2pset (soil1c_grid, nl_soil, soil1c_vr)
            CALL mc2p%grid2pset (soil2c_grid, nl_soil, soil2c_vr)
            CALL mc2p%grid2pset (soil3c_grid, nl_soil, soil3c_vr)
            CALL mc2p%grid2pset (litr1n_grid, nl_soil, litr1n_vr)
            CALL mc2p%grid2pset (litr2n_grid, nl_soil, litr2n_vr)
            CALL mc2p%grid2pset (litr3n_grid, nl_soil, litr3n_vr)
            CALL mc2p%grid2pset (cwdn_grid  , nl_soil, cwdn_vr  )
            CALL mc2p%grid2pset (soil1n_grid, nl_soil, soil1n_vr)
            CALL mc2p%grid2pset (soil2n_grid, nl_soil, soil2n_vr)
            CALL mc2p%grid2pset (soil3n_grid, nl_soil, soil3n_vr)
            CALL mc2p%grid2pset (smin_nh4_grid , nl_soil, min_nh4_vr )
            CALL mc2p%grid2pset (smin_no3_grid , nl_soil, min_no3_vr )
            CALL mc2f%grid2pset (leafc_grid, leafcin_p )
            CALL mc2f%grid2pset (leafc_storage_grid, leafc_storagein_p )
            CALL mc2f%grid2pset (frootc_grid, frootcin_p )
            CALL mc2f%grid2pset (frootc_storage_grid, frootc_storagein_p )
            CALL mc2f%grid2pset (livestemc_grid, livestemcin_p )
            CALL mc2f%grid2pset (deadstemc_grid, deadstemcin_p )
            CALL mc2f%grid2pset (livecrootc_grid, livecrootcin_p )
            CALL mc2f%grid2pset (deadcrootc_grid, deadcrootcin_p )

            IF (p_is_worker) THEN
               DO i = 1, numpatch
                  ps = patch_pft_s(i)
                  pe = patch_pft_e(i)
                  DO nsl = 1, nl_soil
                     decomp_cpools_vr(nsl, i_met_lit, i) = litr1c_vr(nsl, i)
                     decomp_cpools_vr(nsl, i_cel_lit, i) = litr2c_vr(nsl, i)
                     decomp_cpools_vr(nsl, i_lig_lit, i) = litr3c_vr(nsl, i)
                     decomp_cpools_vr(nsl, i_cwd    , i) = cwdc_vr  (nsl, i)
                     decomp_cpools_vr(nsl, i_soil1  , i) = soil1c_vr(nsl, i)
                     decomp_cpools_vr(nsl, i_soil2  , i) = soil2c_vr(nsl, i)
                     decomp_cpools_vr(nsl, i_soil3  , i) = soil3c_vr(nsl, i)
                     decomp_npools_vr(nsl, i_met_lit, i) = litr1n_vr(nsl, i)
                     decomp_npools_vr(nsl, i_cel_lit, i) = litr2n_vr(nsl, i)
                     decomp_npools_vr(nsl, i_lig_lit, i) = litr3n_vr(nsl, i)
                     decomp_npools_vr(nsl, i_cwd    , i) = cwdn_vr  (nsl, i)
                     decomp_npools_vr(nsl, i_soil1  , i) = soil1n_vr(nsl, i)
                     decomp_npools_vr(nsl, i_soil2  , i) = soil2n_vr(nsl, i)
                     decomp_npools_vr(nsl, i_soil3  , i) = soil3n_vr(nsl, i)
                     smin_nh4_vr     (nsl, i)            = min_nh4_vr(nsl,i)
                     smin_no3_vr     (nsl, i)            = min_no3_vr(nsl,i)
                     sminn_vr        (nsl, i)            = min_nh4_vr(nsl,i)+min_no3_vr(nsl,i)
                  ENDDO
                  IF (patchtype(i) == 0)THEN
                     DO m = ps, pe
                        ivt = pftclass(m)
                        IF(isevg(ivt))THEN
                           leafc_p            (m) = leafcin_p(m)
                           frootc_p           (m) = frootcin_p(m)
                        ELSE
                           leafc_p            (m) = leafcin_p(m)
                           leafc_storage_p    (m) = leafc_storagein_p(m)
                           frootc_p           (m) = frootcin_p(m)
                           frootc_storage_p   (m) = frootc_storagein_p(m)
                        ENDIF
                        IF(woody(ivt).eq. 1)THEN
                           deadstemc_p        (m) = deadstemcin_p(m)
                           livestemc_p        (m) = livestemcin_p(m)
                           deadcrootc_p       (m) = deadcrootcin_p(m)
                           livecrootc_p       (m) = livecrootcin_p(m)
                        ENDIF
                     ENDDO
                  ENDIF
               ENDDO
            ENDIF

         ENDIF

      ELSE
         use_cnini = .false.
      ENDIF
#endif

      IF (p_is_worker) THEN
         IF (numpatch > 0) THEN
            allocate (snow_d (numpatch))
         ENDIF
      ENDIF

      IF (DEF_USE_SnowInit) THEN

         fsnowdat = DEF_file_SnowInit
         IF (p_is_master) THEN
            inquire (file=trim(fsnowdat), exist=use_snowini)
         ENDIF
#ifdef USEMPI
         CALL mpi_bcast (use_snowini, 1, MPI_LOGICAL, p_root, p_comm_glb, p_err)
#endif

         IF (use_snowini) THEN

            !CALL gsnow%define_from_file (fsnowdat, latname = 'lat', lonname = 'lon')
         
            CALL julian2monthday (idate(1), idate(2), month, mday)

            IF (p_is_master) THEN
               !CALL ncio_get_attr (fsnowdat, 'snowdepth', 'missing_value', missing_value)
            ENDIF
#ifdef USEMPI
            CALL mpi_bcast (missing_value, 1, MPI_REAL8, p_root, p_comm_glb, p_err)
#endif

            IF (p_is_io) THEN
               !! snow depth (m)
               !CALL allocate_block_data (gsnow, snow_d_grid)
               !CALL ncio_read_block_time (fsnowdat, 'snowdepth', gsnow, month, snow_d_grid)
            ENDIF
            
            IF (p_is_worker) THEN
               IF (numpatch > 0) THEN
                  allocate (validval (numpatch))
               ENDIF
            ENDIF

            !CALL msnow2p%build_arealweighted (gsnow, landpatch)
            !CALL msnow2p%set_missing_value   (snow_d_grid, missing_value, validval)

            !CALL msnow2p%grid2pset (snow_d_grid, snow_d)
            
            IF (p_is_worker) THEN
               WHERE (.not. validval)
                  snow_d = 0.
               END WHERE 
            ENDIF

            IF (allocated(validval)) deallocate(validval)

         ENDIF

      ELSE
         use_snowini = .false.
      ENDIF


      ! for SOIL Water INIT by using water table depth
      fwtd = trim(DEF_dir_runtime) // '/wtd.nc'
      IF (p_is_master) THEN
         inquire (file=trim(fwtd), exist=use_wtd)
         IF (use_soilini) use_wtd = .false.
         IF (use_wtd) THEN
            write(*,'(/, 2A)') 'Use water table depth and derived equilibrium state ' &
               // ' to initialize soil water content: ', trim(fwtd)
         ENDIF
      ENDIF

#ifdef USEMPI
      CALL mpi_bcast (use_wtd, 1, MPI_LOGICAL, p_root, p_comm_glb, p_err)
#endif

      IF (use_wtd) THEN

         CALL julian2monthday (idate(1), idate(2), month, mday)
         !CALL gwtd%define_from_file (fwtd)

         IF (p_is_io) THEN
            !CALL allocate_block_data (gwtd, wtd_xy)
            !CALL ncio_read_block_time (fwtd, 'wtd', gwtd, month, wtd_xy)
         ENDIF

         !CALL m_wtd2p%build_arealweighted (gwtd, landpatch)
         !CALL m_wtd2p%grid2pset (wtd_xy, zwt)

      ENDIF

! ...................
! 2.4 LEAF area index
! ...................
#if(defined DYN_PHENOLOGY)
      ! CREAT fraction of vegetation cover, greenness, leaf area index, stem index
      IF (p_is_worker) THEN

         DO i = 1, numpatch
            IF (use_soilini) THEN
               DO nsl = 1, nl_soil
                  !CALL polint(soil_z,soil_t(:,i),nl_soil_ini,z_soi(nsl),t_soisno(nsl,i))
               ENDDO
            ELSE
               t_soisno(1:,i) = 283.  !@CY: we may not use soilini currently
            ENDIF
         ENDDO

         tlai(:)=0.0; tsai(:)=0.0; green(:)=0.0; fveg(:)=0.0
         DO i = 1, numpatch
            ! Call Ecological Model()  !not commented by CY
            ltyp = patchtype(i)
            m = patchclass(i)
            IF(ltyp < 5) THEN
               CALL lai_empirical(m, nl_soil,rootfr(1:,m), t_soisno(1:,i),tlai(i),tsai(i),fveg(i),green(i))
            ENDIF
         ENDDO

      ENDIF
#else

      year = idate(1)
      jday = idate(2)

      IF (DEF_LAI_MONTHLY) THEN
         CALL julian2monthday (year, jday, month, mday)
         IF (DEF_LAI_CHANGE_YEARLY) THEN
            ! 08/03/2019, yuan: read global LAI/SAI data
            !CALL LAI_readin (year, month, dir_landdata)
#ifdef URBAN_MODEL
            CALL UrbanLAI_readin (year, month, dir_landdata)
#endif
         ELSE
            !CALL LAI_readin (lc_year, month, dir_landdata)
#ifdef URBAN_MODEL
            CALL UrbanLAI_readin (lc_year, month, dir_landdata)
#endif
         ENDIF
      ELSE
         Julian_8day = int(calendarday(idate)-1)/8*8 + 1
         !CALL LAI_readin (year, Julian_8day, dir_landdata)
      ENDIF
#ifdef RangeCheck
      CALL check_vector_data ('LAI ', tlai)
      CALL check_vector_data ('SAI ', tsai)
#endif

#ifdef CROP
         CALL CROP_readin ()
         IF (p_is_worker) THEN
            DO i = 1, numpatch
               IF(patchtype(i) .eq. 0)THEN
                  ps = patch_pft_s(i)
                  pe = patch_pft_e(i)
                  DO m = ps, pe
                     ivt = pftclass(m)
                     IF(ivt >= npcropmin)THEN
                       leafc_p (m) = 0._r8
                       frootc_p(m) = 0._r8
                       tlai    (i) = 0._r8
                       tsai    (i) = 0._r8
                       tlai_p  (m) = 0._r8
                       tsai_p  (m) = 0._r8
                     ENDIF
                  ENDDO
               ENDIF
            ENDDO
         ENDIF
      IF(DEF_USE_IRRIGATION)THEN
         irrig_rate(:) = 0._r8
         deficit_irrig(:) = 0._r8
         sum_irrig(:) = 0._r8
         sum_irrig_count(:) = 0._r8
         n_irrig_steps_left(:) = 0
      ENDIF
#endif
#endif

! ..............................................................................
! 2.5 initialize time-varying variables, as subgrid vectors of length [numpatch]
! ..............................................................................
      ! ------------------------------------------
      ! PLEASE
      ! PLEASE UPDATE
      ! PLEASE UPDATE when have the observed lake status
      IF (p_is_worker) THEN

         t_lake      (:,:) = 285.
         lake_icefrac(:,:) = 0.
         savedtke1   (:)   = tkwat
         !@CY: MOD_Const_Physical.F90:21:  real(r8), parameter :: tkwat  = 0.6

      ENDIF
      ! ------------------------------------------

      IF (p_is_worker) THEN

         !TODO: can be removed as CLMDRIVER.F90 yuan@
         allocate ( z_soisno (maxsnl+1:nl_soil,numpatch) ); z_soisno (:,:) = spval
         allocate ( dz_soisno(maxsnl+1:nl_soil,numpatch) ); dz_soisno(:,:) = spval

         DO i = 1, numpatch
            z_soisno (1:nl_soil ,i) = z_soi (1:nl_soil)
            dz_soisno(1:nl_soil ,i) = dz_soi(1:nl_soil)
         ENDDO

         zc_soimm = z_soi  * 1000.
         zi_soimm(0) = 0.
         zi_soimm(1:nl_soil) = zi_soi * 1000.

         DO i = 1, numpatch
            m = patchclass(i)

            IF (use_wtd) THEN
               zwtmm = zwt(i) * 1000.
            ENDIF

#ifdef Campbell_SOIL_MODEL
            vliq_r(:) = 0.
            prms(1,1:nl_soil) = bsw(1:nl_soil,i)
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
            vliq_r(:) = theta_r(1:nl_soil,i)
            prms(1,1:nl_soil) = alpha_vgm(1:nl_soil,i)
            prms(2,1:nl_soil) = n_vgm    (1:nl_soil,i)
            prms(3,1:nl_soil) = L_vgm    (1:nl_soil,i)
            prms(4,1:nl_soil) = sc_vgm   (1:nl_soil,i)
            prms(5,1:nl_soil) = fc_vgm   (1:nl_soil,i)
#endif

            CALL iniTimeVar(i, patchtype(i)&
               ,porsl(1:,i),psi0(1:,i),hksati(1:,i)&
               ,soil_s_v_alb(i),soil_d_v_alb(i),soil_s_n_alb(i),soil_d_n_alb(i)&
               ,z0m(i),zlnd,htop(i),z0mr(m),chil(m),rho(1:,1:,m),tau(1:,1:,m)&
               ,z_soisno(maxsnl+1:,i),dz_soisno(maxsnl+1:,i)&
               ,t_soisno(maxsnl+1:,i),wliq_soisno(maxsnl+1:,i),wice_soisno(maxsnl+1:,i)&
               ,smp(1:,i),hk(1:,i),zwt(i),wa(i)&
!Plant hydraulic variables
               ,vegwp(1:,i),gs0sun(i),gs0sha(i)&
!END plant hydraulic variables
               ,t_grnd(i),tleaf(i),ldew(i),ldew_rain(i),ldew_snow(i),sag(i),scv(i)&
               ,snowdp(i),fveg(i),fsno(i),sigf(i),green(i),lai(i),sai(i),coszen(i)&
               ,snw_rds(:,i),mss_bcpho(:,i),mss_bcphi(:,i),mss_ocpho(:,i),mss_ocphi(:,i)&
               ,mss_dst1(:,i),mss_dst2(:,i),mss_dst3(:,i),mss_dst4(:,i)&
               ,alb(1:,1:,i),ssun(1:,1:,i),ssha(1:,1:,i)&
               ,ssoi(1:,1:,i),ssno(1:,1:,i),ssno_lyr(1:,1:,:,i)&
               ,thermk(i),extkb(i),extkd(i)&
               ,trad(i),tref(i),qref(i),rst(i),emis(i),zol(i),rib(i)&
               ,ustar(i),qstar(i),tstar(i),fm(i),fh(i),fq(i)&
#ifdef BGC
               ,use_cnini, totlitc(i), totsomc(i), totcwdc(i), decomp_cpools(:,i), decomp_cpools_vr(:,:,i) &
               ,ctrunc_veg(i), ctrunc_soil(i), ctrunc_vr(:,i) &
               ,totlitn(i), totsomn(i), totcwdn(i), decomp_npools(:,i), decomp_npools_vr(:,:,i) &
               ,ntrunc_veg(i), ntrunc_soil(i), ntrunc_vr(:,i) &
               ,totvegc(i), totvegn(i), totcolc(i), totcoln(i), col_endcb(i), col_begcb(i), col_endnb(i), col_begnb(i) &
               ,col_vegendcb(i), col_vegbegcb(i), col_soilendcb(i), col_soilbegcb(i) &
               ,col_vegendnb(i), col_vegbegnb(i), col_soilendnb(i), col_soilbegnb(i) &
               ,col_sminnendnb(i), col_sminnbegnb(i) &
               ,altmax(i) , altmax_lastyear(i), altmax_lastyear_indx(i), lag_npp(i) &
               ,sminn_vr(:,i), sminn(i), smin_no3_vr  (:,i), smin_nh4_vr       (:,i)&
               ,prec10(i), prec60(i), prec365 (i), prec_today(i), prec_daily(:,i), tsoi17(i), rh30(i), accumnstep(i) , skip_balance_check(i) &
!------------------------SASU variables----------------------- ,decomp0_cpools_vr        (:,:,i), decomp0_npools_vr        (:,:,i) &
               ,decomp0_cpools_vr        (:,:,i), decomp0_npools_vr        (:,:,i)    &
               ,I_met_c_vr_acc             (:,i), I_cel_c_vr_acc             (:,i), I_lig_c_vr_acc             (:,i), I_cwd_c_vr_acc             (:,i) &
               ,AKX_met_to_soil1_c_vr_acc  (:,i), AKX_cel_to_soil1_c_vr_acc  (:,i), AKX_lig_to_soil2_c_vr_acc  (:,i), AKX_soil1_to_soil2_c_vr_acc(:,i) &
               ,AKX_cwd_to_cel_c_vr_acc    (:,i), AKX_cwd_to_lig_c_vr_acc    (:,i), AKX_soil1_to_soil3_c_vr_acc(:,i), AKX_soil2_to_soil1_c_vr_acc(:,i) &
               ,AKX_soil2_to_soil3_c_vr_acc(:,i), AKX_soil3_to_soil1_c_vr_acc(:,i) &
               ,AKX_met_exit_c_vr_acc      (:,i), AKX_cel_exit_c_vr_acc      (:,i), AKX_lig_exit_c_vr_acc      (:,i), AKX_cwd_exit_c_vr_acc      (:,i) &
               ,AKX_soil1_exit_c_vr_acc    (:,i), AKX_soil2_exit_c_vr_acc    (:,i), AKX_soil3_exit_c_vr_acc    (:,i) &
               ,diagVX_c_vr_acc          (:,:,i), upperVX_c_vr_acc         (:,:,i), lowerVX_c_vr_acc         (:,:,i) &
               ,I_met_n_vr_acc             (:,i), I_cel_n_vr_acc             (:,i), I_lig_n_vr_acc             (:,i), I_cwd_n_vr_acc             (:,i) &
               ,AKX_met_to_soil1_n_vr_acc  (:,i), AKX_cel_to_soil1_n_vr_acc  (:,i), AKX_lig_to_soil2_n_vr_acc  (:,i), AKX_soil1_to_soil2_n_vr_acc(:,i) &
               ,AKX_cwd_to_cel_n_vr_acc    (:,i), AKX_cwd_to_lig_n_vr_acc    (:,i), AKX_soil1_to_soil3_n_vr_acc(:,i), AKX_soil2_to_soil1_n_vr_acc(:,i) &
               ,AKX_soil2_to_soil3_n_vr_acc(:,i), AKX_soil3_to_soil1_n_vr_acc(:,i) &
               ,AKX_met_exit_n_vr_acc      (:,i), AKX_cel_exit_n_vr_acc      (:,i), AKX_lig_exit_n_vr_acc      (:,i), AKX_cwd_exit_n_vr_acc      (:,i) &
               ,AKX_soil1_exit_n_vr_acc    (:,i), AKX_soil2_exit_n_vr_acc    (:,i), AKX_soil3_exit_n_vr_acc    (:,i) &
               ,diagVX_n_vr_acc          (:,:,i), upperVX_n_vr_acc         (:,:,i), lowerVX_n_vr_acc         (:,:,i) &
   !------------------------------------------------------------
#endif
               ! for SOIL INIT of water, temperature, snow depth
               ,use_soilini, nl_soil_ini, soil_z, soil_t(1:,i), soil_w(1:,i), use_snowini, snow_d(i) &
               ! for SOIL Water INIT by using water table depth
               ,use_wtd, zwtmm, zc_soimm, zi_soimm, vliq_r, nprms, prms)

#ifdef URBAN_MODEL
            IF (m == URBAN) THEN

               u = patch2urban(i)
               lwsun         (u) = 0.   !net longwave radiation of sunlit wall
               lwsha         (u) = 0.   !net longwave radiation of shaded wall
               lgimp         (u) = 0.   !net longwave radiation of impervious road
               lgper         (u) = 0.   !net longwave radiation of pervious road
               lveg          (u) = 0.   !net longwave radiation of vegetation [W/m2]

               t_roofsno   (:,u) = 283. !temperatures of roof layers
               t_wallsun   (:,u) = 283. !temperatures of sunlit wall layers
               t_wallsha   (:,u) = 283. !temperatures of shaded wall layers
               t_gimpsno   (:,u) = 283. !temperatures of impervious road layers
               t_gpersno   (:,u) = 283. !soil temperature [K]
               t_lakesno   (:,u) = 283. !lake soil temperature [K]

               wice_roofsno(:,u) = 0.   !ice lens [kg/m2]
               wice_gimpsno(:,u) = 0.   !ice lens [kg/m2]
               wice_gpersno(:,u) = 0.   !ice lens [kg/m2]
               wice_lakesno(:,u) = 0.   !ice lens [kg/m2]
               wliq_roofsno(:,u) = 0.   !liqui water [kg/m2]
               wliq_gimpsno(:,u) = 0.   !liqui water [kg/m2]
               wliq_gpersno(:,u) = wliq_soisno(:,i) !liqui water [kg/m2]
               wliq_lakesno(:,u) = wliq_soisno(:,i) !liqui water [kg/m2]

               wliq_soisno(: ,i) = 0.
               wliq_soisno(:1,i) = wliq_roofsno(:1,u)*froof(u)
               wliq_soisno(: ,i) = wliq_soisno(: ,i) + wliq_gpersno(: ,u)*(1-froof(u))*fgper(u)
               wliq_soisno(:1,i) = wliq_soisno(:1,i) + wliq_gimpsno(:1,u)*(1-froof(u))*(1-fgper(u))

               snowdp_roof   (u) = 0.   !snow depth [m]
               snowdp_gimp   (u) = 0.   !snow depth [m]
               snowdp_gper   (u) = 0.   !snow depth [m]
               snowdp_lake   (u) = 0.   !snow depth [m]

               z_sno_roof  (:,u) = 0.   !node depth of roof [m]
               z_sno_gimp  (:,u) = 0.   !node depth of impervious [m]
               z_sno_gper  (:,u) = 0.   !node depth pervious [m]
               z_sno_lake  (:,u) = 0.   !node depth lake [m]

               dz_sno_roof (:,u) = 0.   !interface depth of roof [m]
               dz_sno_gimp (:,u) = 0.   !interface depth of impervious [m]
               dz_sno_gper (:,u) = 0.   !interface depth pervious [m]
               dz_sno_lake (:,u) = 0.   !interface depth lake [m]

               t_room        (u) = 283. !temperature of inner building [K]
               troof_inner   (u) = 283. !temperature of inner roof [K]
               twsun_inner   (u) = 283. !temperature of inner sunlit wall [K]
               twsha_inner   (u) = 283. !temperature of inner shaded wall [K]
               Fhac          (u) = 0.   !sensible flux from heat or cool AC [W/m2]
               Fwst          (u) = 0.   !waste heat flux from heat or cool AC [W/m2]
               Fach          (u) = 0.   !flux from inner and outter air exchange [W/m2]

               CALL UrbanIniTimeVar(i,froof(u),fgper(u),flake(u),hwr(u),hroof(u),&
                  alb_roof(:,:,u),alb_wall(:,:,u),alb_gimp(:,:,u),alb_gper(:,:,u),&
                  rho(:,:,m),tau(:,:,m),fveg(i),htop(i),hbot(i),lai(i),sai(i),coszen(i),&
                  fsno_roof(u),fsno_gimp(u),fsno_gper(u),fsno_lake(u),&
                  scv_roof(u),scv_gimp(u),scv_gper(u),scv_lake(u),&
                  sag_roof(u),sag_gimp(u),sag_gper(u),sag_lake(u),t_lake(1,i),&
                  fwsun(u),dfwsun(u),extkd(i),alb(:,:,i),ssun(:,:,i),ssha(:,:,i),sroof(:,:,u),&
                  swsun(:,:,u),swsha(:,:,u),sgimp(:,:,u),sgper(:,:,u),slake(:,:,u))

            ENDIF
#endif
         ENDDO

         DO i = 1, numpatch
            z_sno (maxsnl+1:0,i) = z_soisno (maxsnl+1:0,i)
            dz_sno(maxsnl+1:0,i) = dz_soisno(maxsnl+1:0,i)
         ENDDO

      ENDIF
      ! ------------------------------------------

      ! -----
#ifdef CatchLateralFlow

      CALL element_neighbour_init  (lc_year)
      CALL hillslope_network_init  ()
      CALL river_lake_network_init ()

      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN
            wdsrf(:) = 0.
         ENDIF

         DO i = 1, numelm
            IF (lake_id(i) > 0) THEN
               ps = elm_patch%substt(i)
               pe = elm_patch%subend(i)
               wdsrf(ps:pe) = lakedepth(ps:pe) * 1.0e3 ! m to mm
            ELSE
               IF (hillslope_network(i)%indx(1) == 0) THEN
                  hs = basin_hru%substt(i)
                  ps = hru_patch%substt(hs)
                  pe = hru_patch%subend(hs)
                  wdsrf(ps:pe) = riverdpth(i) * 1.0e3 ! m to mm
               ENDIF
            ENDIF
         ENDDO

         IF (numhru > 0) THEN
            DO i = 1, numhru
               ps = hru_patch%substt(i)
               pe = hru_patch%subend(i)
               wdsrf_hru(i) = sum(wdsrf(ps:pe) * hru_patch%subfrc(ps:pe))
               wdsrf_hru(i) = wdsrf_hru(i) / 1.0e3 ! mm to m
            ENDDO
            veloc_hru(:) = 0
            wdsrf_hru_prev(:) = wdsrf_hru(:)
         ENDIF

         IF (numelm > 0) THEN
            DO i = 1, numelm
               hs = basin_hru%substt(i)
               he = basin_hru%subend(i)
               IF (lake_id(i) <= 0) THEN
                  wdsrf_bsn(i) = minval(hillslope_network(i)%hand + wdsrf_hru(hs:he))
               ELSE
                  ! lake
                  totalvolume  = sum(wdsrf_hru(hs:he) * lakes(i)%area0)
                  wdsrf_bsn(i) = lakes(i)%surface(totalvolume)
               ENDIF
            ENDDO
            veloc_riv(:) = 0
            wdsrf_bsn_prev(:) = wdsrf_bsn(:)
         ENDIF

      ENDIF
#endif

! ...............................................................
! 2.6 Write out the model variables for restart run [histTimeVar]
! ...............................................................

#ifdef RangeCheck
      CALL check_TimeVariables ()
#endif

      IF ( .not. present(lulcc_call) ) THEN
         !! only be called in runing MKINI, LULCC will be executed later
         !CALL WRITE_TimeVariables (idate, lc_year, casename, dir_restart)
      ENDIF

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

      IF (p_is_master) write (6,*) ('Successfully Initialize the Land Time-Vraying Variables')


! --------------------------------------------------
! Deallocates memory for CoLM 1d [numpatch] variables
! --------------------------------------------------
      IF ( .not. present(lulcc_call) ) THEN
         !! only be called in runing MKINI, LULCC will be executed later
         !CALL deallocate_TimeInvariants
         !CALL deallocate_TimeVariables
      ENDIF

      IF (allocated(z_soisno )) deallocate (z_soisno )
      IF (allocated(dz_soisno)) deallocate (dz_soisno)
      !@CY: local in CoLMMAIN.F90

      IF (allocated(soil_z)) deallocate (soil_z)
      IF (allocated(snow_d)) deallocate (snow_d)
      IF (allocated(soil_t)) deallocate (soil_t)
      IF (allocated(soil_w)) deallocate (soil_w)

   END SUBROUTINE initialize

END MODULE MOD_Initialize
! --------------------------------------------------
! EOP
