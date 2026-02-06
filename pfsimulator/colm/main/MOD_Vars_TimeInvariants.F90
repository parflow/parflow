#include <define.h>

! -------------------------------
! Created by Yongjiu Dai, 03/2014
! -------------------------------
MODULE MOD_Vars_PFTimeInvariants
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)

! -----------------------------------------------------------------
! !DESCRIPTION:
! Define PFT time invariables
!
! Added by Hua Yuan, 08/2019
! -----------------------------------------------------------------

   USE MOD_Precision
   USE MOD_Vars_Global
   IMPLICIT NONE
   SAVE

   ! for LULC_IGBP_PFT and LULC_IGBP_PC
   integer , allocatable :: pftclass    (:)    !PFT type
   real(r8), allocatable :: pftfrac     (:)    !PFT fractional cover
   real(r8), allocatable :: htop_p      (:)    !canopy top height [m]
   real(r8), allocatable :: hbot_p      (:)    !canopy bottom height [m]
#ifdef CROP
   real(r8), allocatable :: cropfrac    (:)    !Crop fractional cover
#endif

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: allocate_PFTimeInvariants
   PUBLIC :: READ_PFTimeInvariants
   PUBLIC :: WRITE_PFTimeInvariants
   PUBLIC :: deallocate_PFTimeInvariants
#ifdef RangeCheck
   PUBLIC :: check_PFTimeInvariants
#endif

! PRIVATE MEMBER FUNCTIONS:

!-----------------------------------------------------------------------

CONTAINS

!-----------------------------------------------------------------------

   SUBROUTINE allocate_PFTimeInvariants
   ! --------------------------------------------------------------------
   ! Allocates memory for CoLM PFT 1d [numpft] variables
   ! --------------------------------------------------------------------

   USE MOD_SPMD_Task
   USE MOD_LandPatch, only : numpatch
   USE MOD_LandPFT,   only : numpft
   USE MOD_Precision
   IMPLICIT NONE

      IF (p_is_worker) THEN
         IF (numpft > 0) THEN
            allocate (pftclass      (numpft))
            allocate (pftfrac       (numpft))
            allocate (htop_p        (numpft))
            allocate (hbot_p        (numpft))
#ifdef CROP
            allocate (cropfrac    (numpatch))
#endif
         ENDIF
      ENDIF

   END SUBROUTINE allocate_PFTimeInvariants

   SUBROUTINE READ_PFTimeInvariants (file_restart)

   USE MOD_NetCDFVector
   USE MOD_LandPatch
   USE MOD_LandPFT
   IMPLICIT NONE

   character(len=*), intent(in) :: file_restart

      CALL ncio_read_vector (file_restart, 'pftclass', landpft, pftclass) !
      CALL ncio_read_vector (file_restart, 'pftfrac ', landpft, pftfrac ) !
      CALL ncio_read_vector (file_restart, 'htop_p  ', landpft, htop_p  ) !
      CALL ncio_read_vector (file_restart, 'hbot_p  ', landpft, hbot_p  ) !
#ifdef CROP
      CALL ncio_read_vector (file_restart, 'cropfrac ', landpatch, cropfrac) !
#endif

   END SUBROUTINE READ_PFTimeInvariants

   SUBROUTINE WRITE_PFTimeInvariants (file_restart)

   USE MOD_NetCDFVector
   USE MOD_LandPFT
   USE MOD_LandPatch
   USE MOD_Namelist
   USE MOD_Vars_Global
   IMPLICIT NONE

   ! Local variables
   character(len=*), intent(in) :: file_restart
   integer :: compress

      compress = DEF_REST_CompressLevel

      CALL ncio_create_file_vector (file_restart, landpft)
      CALL ncio_define_dimension_vector (file_restart, landpft, 'pft')

      CALL ncio_write_vector (file_restart, 'pftclass', 'pft', landpft, pftclass, compress) !
      CALL ncio_write_vector (file_restart, 'pftfrac ', 'pft', landpft, pftfrac , compress) !
      CALL ncio_write_vector (file_restart, 'htop_p  ', 'pft', landpft, htop_p  , compress) !
      CALL ncio_write_vector (file_restart, 'hbot_p  ', 'pft', landpft, hbot_p  , compress) !

#ifdef CROP
      CALL ncio_define_dimension_vector (file_restart, landpatch, 'patch')
      CALL ncio_write_vector (file_restart, 'cropfrac', 'patch', landpatch, cropfrac, compress) !
#endif

   END SUBROUTINE WRITE_PFTimeInvariants

   SUBROUTINE deallocate_PFTimeInvariants
! --------------------------------------------------
! Deallocates memory for CoLM PFT 1d [numpft] variables
! --------------------------------------------------
   USE MOD_SPMD_Task
   USE MOD_LandPFT

      IF (p_is_worker) THEN
         IF (numpft > 0) THEN
            deallocate (pftclass)
            deallocate (pftfrac )
            deallocate (htop_p  )
            deallocate (hbot_p  )
#ifdef CROP
            deallocate (cropfrac)
#endif
         ENDIF
      ENDIF

   END SUBROUTINE deallocate_PFTimeInvariants

#ifdef RangeCheck
   SUBROUTINE check_PFTimeInvariants ()

   USE MOD_RangeCheck
   IMPLICIT NONE

      CALL check_vector_data ('pftfrac', pftfrac) !
      CALL check_vector_data ('htop_p ', htop_p ) !
      CALL check_vector_data ('hbot_p ', hbot_p ) !
#ifdef CROP
      CALL check_vector_data ('cropfrac', cropfrac) !
#endif

   END SUBROUTINE check_PFTimeInvariants
#endif

#endif
END MODULE MOD_Vars_PFTimeInvariants

MODULE MOD_Vars_TimeInvariants
! -------------------------------
! Created by Yongjiu Dai, 03/2014
! -------------------------------

   USE MOD_Precision
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_Vars_PFTimeInvariants
#endif
#ifdef BGC
   USE MOD_BGC_Vars_TimeInvariants
#endif
#ifdef URBAN_MODEL
   USE MOD_Urban_Vars_TimeInvariants
#endif
   IMPLICIT NONE
   SAVE

! -----------------------------------------------------------------
! surface classification and soil information
   integer,  allocatable :: patchclass     (:)  !index of land cover type of the patches at the fraction > 0
   integer,  allocatable :: patchtype      (:)  !land patch type
   logical,  allocatable :: patchmask      (:)  !patch mask

   real(r8), allocatable :: patchlatr      (:)  !latitude in radians
   real(r8), allocatable :: patchlonr      (:)  !longitude in radians

   real(r8), allocatable :: lakedepth      (:)  !lake depth
   real(r8), allocatable :: dz_lake      (:,:)  !new lake scheme

   real(r8), allocatable :: soil_s_v_alb   (:)  !albedo of visible of the saturated soil
   real(r8), allocatable :: soil_d_v_alb   (:)  !albedo of visible of the dry soil
   real(r8), allocatable :: soil_s_n_alb   (:)  !albedo of near infrared of the saturated soil
   real(r8), allocatable :: soil_d_n_alb   (:)  !albedo of near infrared of the dry soil

   real(r8), allocatable :: vf_quartz    (:,:)  !volumetric fraction of quartz within mineral soil
   real(r8), allocatable :: vf_gravels   (:,:)  !volumetric fraction of gravels
   real(r8), allocatable :: vf_om        (:,:)  !volumetric fraction of organic matter
   real(r8), allocatable :: vf_sand      (:,:)  !volumetric fraction of sand
   real(r8), allocatable :: wf_gravels   (:,:)  !gravimetric fraction of gravels
   real(r8), allocatable :: wf_sand      (:,:)  !gravimetric fraction of sand
   real(r8), allocatable :: OM_density   (:,:)  !OM density (kg/m3)
   real(r8), allocatable :: BD_all       (:,:)  !bulk density of soil (GRAVELS + ORGANIC MATTER + Mineral Soils,kg/m3)

   real(r8), allocatable :: wfc          (:,:)  !field capacity
   real(r8), allocatable :: porsl        (:,:)  !fraction of soil that is voids [-]
   real(r8), allocatable :: psi0         (:,:)  !minimum soil suction [mm] (NOTE: "-" valued)
   real(r8), allocatable :: bsw          (:,:)  !clapp and hornbereger "b" parameter [-]
   real(r8), allocatable :: theta_r      (:,:)  !residual moisture content [-]
   real(r8), allocatable :: BVIC         (:,:)  !b parameter in Fraction of saturated soil in a grid calculated by VIC
#ifdef vanGenuchten_Mualem_SOIL_MODEL
   real(r8), allocatable :: alpha_vgm    (:,:)  !a parameter corresponding approximately to the inverse of the air-entry value
   real(r8), allocatable :: L_vgm        (:,:)  !pore-connectivity parameter [dimensionless]
   real(r8), allocatable :: n_vgm        (:,:)  !a shape parameter [dimensionless]
   real(r8), allocatable :: sc_vgm       (:,:)  !saturation at the air entry value in the classical vanGenuchten model [-]
   real(r8), allocatable :: fc_vgm       (:,:)  !a scaling factor by using air entry value in the Mualem model [-]
#endif

   real(r8), allocatable :: vic_b_infilt (:)
   real(r8), allocatable :: vic_Dsmax    (:)
   real(r8), allocatable :: vic_Ds       (:)
   real(r8), allocatable :: vic_Ws       (:)
   real(r8), allocatable :: vic_c        (:)

   real(r8), allocatable :: hksati       (:,:)  !hydraulic conductivity at saturation [mm h2o/s]
   real(r8), allocatable :: csol         (:,:)  !heat capacity of soil solids [J/(m3 K)]
   real(r8), allocatable :: k_solids     (:,:)  !thermal conductivity of soil solids [W/m-K]
   real(r8), allocatable :: dksatu       (:,:)  !thermal conductivity of saturated soil [W/m-K]
   real(r8), allocatable :: dksatf       (:,:)  !thermal conductivity of saturated frozen soil [W/m-K]
   real(r8), allocatable :: dkdry        (:,:)  !thermal conductivity for dry soil  [W/(m-K)]
   real(r8), allocatable :: BA_alpha     (:,:)  !alpha in Balland and Arp(2005) thermal conductivity scheme
   real(r8), allocatable :: BA_beta      (:,:)  !beta in Balland and Arp(2005) thermal conductivity scheme
   real(r8), allocatable :: htop           (:)  !canopy top height [m]
   real(r8), allocatable :: hbot           (:)  !canopy bottom height [m]

   real(r8), allocatable :: dbedrock       (:)  !depth to bedrock
   integer , allocatable :: ibedrock       (:)  !bedrock level

   real(r8), allocatable :: topoelv (:)  !elevation above sea level [m]
   real(r8), allocatable :: topostd (:)  !standard deviation of elevation [m]

   real(r8) :: zlnd                             !roughness length for soil [m]
   real(r8) :: zsno                             !roughness length for snow [m]
   real(r8) :: csoilc                           !drag coefficient for soil under canopy [-]
   real(r8) :: dewmx                            !maximum dew
   real(r8) :: wtfact                           !fraction of model area with high water table
   real(r8) :: capr                             !tuning factor to turn first layer T into surface T
   real(r8) :: cnfac                            !Crank Nicholson factor between 0 and 1
   real(r8) :: ssi                              !irreducible water saturation of snow
   real(r8) :: wimp                             !water impremeable IF porosity less than wimp
   real(r8) :: pondmx                           !ponding depth (mm)
   real(r8) :: smpmax                           !wilting point potential in mm
   real(r8) :: smpmin                           !restriction for min of soil poten. (mm)
   real(r8) :: trsmx0                           !max transpiration for moist soil+100% veg.  [mm/s]
   real(r8) :: tcrit                            !critical temp. to determine rain or snow
   real(r8) :: wetwatmax                        !maximum wetland water (mm)

   ! Used for downscaling
   real(r8), allocatable    :: svf_patches (:)                                           ! sky view factor
   real(r8), allocatable    :: cur_patches (:)                                           ! curvature
   real(r8), allocatable    :: sf_lut_patches (:,:,:)                                    ! look up table of shadow factor of a patch
   real(r8), allocatable    :: asp_type_patches        (:,:)                             ! topographic aspect of each character of one patch
   real(r8), allocatable    :: slp_type_patches        (:,:)                             ! topographic slope of each character of one patch
   real(r8), allocatable    :: area_type_patches       (:,:)                             ! area percentage of each character of one patch

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: allocate_TimeInvariants
   PUBLIC :: deallocate_TimeInvariants
   PUBLIC :: READ_TimeInvariants
   PUBLIC :: WRITE_TimeInvariants

! PRIVATE MEMBER FUNCTIONS:

!-----------------------------------------------------------------------

CONTAINS

!-----------------------------------------------------------------------

   SUBROUTINE allocate_TimeInvariants (numpatch)
   ! --------------------------------------------------------------------
   ! Allocates memory for CoLM 1d [numpatch] variables
   ! --------------------------------------------------------------------

   USE MOD_Precision
   USE MOD_Vars_Global
   USE MOD_SPMD_Task
   !USE MOD_LandPatch, only: numpatch
   IMPLICIT NONE
   integer :: numpatch

      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN

            allocate (patchclass           (numpatch))
            allocate (patchtype            (numpatch))
            allocate (patchmask            (numpatch))

            allocate (patchlonr            (numpatch))
            allocate (patchlatr            (numpatch))

            allocate (lakedepth            (numpatch))
            allocate (dz_lake      (nl_lake,numpatch))

            allocate (soil_s_v_alb         (numpatch))
            allocate (soil_d_v_alb         (numpatch))
            allocate (soil_s_n_alb         (numpatch))
            allocate (soil_d_n_alb         (numpatch))

            allocate (vf_quartz    (nl_soil,numpatch))
            allocate (vf_gravels   (nl_soil,numpatch))
            allocate (vf_om        (nl_soil,numpatch))
            allocate (vf_sand      (nl_soil,numpatch))
            allocate (wf_gravels   (nl_soil,numpatch))
            allocate (wf_sand      (nl_soil,numpatch))
            allocate (OM_density   (nl_soil,numpatch))
            allocate (BD_all       (nl_soil,numpatch))
            allocate (wfc          (nl_soil,numpatch))
            allocate (porsl        (nl_soil,numpatch))
            allocate (psi0         (nl_soil,numpatch))
            allocate (bsw          (nl_soil,numpatch))
            allocate (theta_r      (nl_soil,numpatch))
            allocate (BVIC         (nl_soil,numpatch))

#ifdef vanGenuchten_Mualem_SOIL_MODEL
            allocate (alpha_vgm    (nl_soil,numpatch))
            allocate (L_vgm        (nl_soil,numpatch))
            allocate (n_vgm        (nl_soil,numpatch))
            allocate (sc_vgm       (nl_soil,numpatch))
            allocate (fc_vgm       (nl_soil,numpatch))
#endif

            allocate (vic_b_infilt (numpatch))
            allocate (vic_Dsmax    (numpatch))
            allocate (vic_Ds       (numpatch))
            allocate (vic_Ws       (numpatch))
            allocate (vic_c        (numpatch))

            allocate (hksati       (nl_soil,numpatch))
            allocate (csol         (nl_soil,numpatch))
            allocate (k_solids     (nl_soil,numpatch))
            allocate (dksatu       (nl_soil,numpatch))
            allocate (dksatf       (nl_soil,numpatch))
            allocate (dkdry        (nl_soil,numpatch))
            allocate (BA_alpha     (nl_soil,numpatch))
            allocate (BA_beta      (nl_soil,numpatch))
            allocate (htop                 (numpatch))
            allocate (hbot                 (numpatch))
            allocate (dbedrock             (numpatch))
            allocate (ibedrock             (numpatch))
            allocate (topoelv              (numpatch))
            allocate (topostd              (numpatch))
      
            ! Used for downscaling
            allocate (svf_patches          (numpatch))
            allocate (asp_type_patches     (num_type,numpatch))
            allocate (slp_type_patches     (num_type,numpatch))
            allocate (area_type_patches    (num_type,numpatch))
            allocate (sf_lut_patches       (num_azimuth,num_zenith,numpatch))
            allocate (cur_patches          (numpatch))
      ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL allocate_PFTimeInvariants
#endif

#ifdef BGC
      CALL allocate_BGCTimeInvariants
#endif

#ifdef URBAN_MODEL
      CALL allocate_UrbanTimeInvariants
#endif

   ENDIF

   END SUBROUTINE allocate_TimeInvariants

   !---------------------------------------
   SUBROUTINE WRITE_TimeInvariants (rank)

      USE MOD_SPMD_Task
      USE MOD_Namelist, only : DEF_USE_BEDROCK, DEF_USE_Forcing_Downscaling

      IMPLICIT NONE
      integer :: rank
      character*100 RI

      IF (p_is_master) THEN
         write(*,'(A29)') 'Write Time Invariants done.'
      ENDIF

      write(RI,*) rank
      open(40,file='CoLM.TimeInvariants.rst.'//trim(adjustl(RI)),form='unformatted')

      write(40) patchclass
      write(40) patchtype
      write(40) patchmask

      write(40) patchlonr
      write(40) patchlatr

      write(40) lakedepth
      write(40) dz_lake

      write(40) soil_s_v_alb
      write(40) soil_d_v_alb
      write(40) soil_s_n_alb
      write(40) soil_d_n_alb

      write(40) vf_quartz
      write(40) vf_gravels
      write(40) vf_om
      write(40) vf_sand
      write(40) wf_gravels
      write(40) wf_sand
      write(40) OM_density
      write(40) BD_all
      write(40) wfc
      write(40) porsl
      write(40) psi0
      write(40) bsw
      write(40) theta_r
      write(40) BVIC

#ifdef vanGenuchten_Mualem_SOIL_MODEL
      write(40) alpha_vgm
      write(40) L_vgm
      write(40) n_vgm
      write(40) sc_vgm
      write(40) fc_vgm
#endif

      write(40) vic_b_infilt
      write(40) vic_Dsmax
      write(40) vic_Ds
      write(40) vic_Ws
      write(40) vic_c

      write(40) hksati
      write(40) csol
      write(40) k_solids
      write(40) dksatu
      write(40) dksatf
      write(40) dkdry
      write(40) BA_alpha
      write(40) BA_beta

      write(40) htop
      write(40) hbot

      IF(DEF_USE_BEDROCK)THEN
          write(40) dbedrock
          write(40) ibedrock
      ENDIF

      write(40) topoelv
      write(40) topostd

      IF (DEF_USE_Forcing_Downscaling) THEN
         write(40) svf_patches
         write(40) cur_patches
         write(40) slp_type_patches
         write(40) asp_type_patches
         write(40) area_type_patches
         write(40) sf_lut_patches
      ENDIF

      write(40) zlnd
      write(40) zsno
      write(40) csoilc
      write(40) dewmx
      write(40) wtfact
      write(40) capr
      write(40) cnfac
      write(40) ssi
      write(40) wimp
      write(40) pondmx
      write(40) smpmax
      write(40) smpmin
      write(40) trsmx0
      write(40) tcrit
      write(40) wetwatmax

      close(40)

   END SUBROUTINE WRITE_TimeInvariants

   SUBROUTINE READ_TimeInvariants (rank)

      USE MOD_SPMD_Task
      USE MOD_Namelist, only : DEF_USE_BEDROCK, DEF_USE_Forcing_Downscaling

      IMPLICIT NONE
      integer :: rank
      character*100 RI

      write(RI,*) rank
      open(40,file='CoLM.TimeInvariants.rst.'//trim(adjustl(RI)),form='unformatted')

      read(40) patchclass
      read(40) patchtype
      read(40) patchmask

      read(40) patchlonr
      read(40) patchlatr

      read(40) lakedepth
      read(40) dz_lake

      read(40) soil_s_v_alb
      read(40) soil_d_v_alb
      read(40) soil_s_n_alb
      read(40) soil_d_n_alb

      read(40) vf_quartz
      read(40) vf_gravels
      read(40) vf_om
      read(40) vf_sand
      read(40) wf_gravels
      read(40) wf_sand
      read(40) OM_density
      read(40) BD_all
      read(40) wfc
      read(40) porsl
      read(40) psi0
      read(40) bsw
      read(40) theta_r
      read(40) BVIC

#ifdef vanGenuchten_Mualem_SOIL_MODEL
      read(40) alpha_vgm
      read(40) L_vgm
      read(40) n_vgm
      read(40) sc_vgm
      read(40) fc_vgm
#endif

      read(40) vic_b_infilt
      read(40) vic_Dsmax
      read(40) vic_Ds
      read(40) vic_Ws
      read(40) vic_c

      read(40) hksati
      read(40) csol
      read(40) k_solids
      read(40) dksatu
      read(40) dksatf
      read(40) dkdry
      read(40) BA_alpha
      read(40) BA_beta

      read(40) htop
      read(40) hbot

      IF(DEF_USE_BEDROCK)THEN
          read(40) dbedrock
          read(40) ibedrock
      ENDIF

      read(40) topoelv
      read(40) topostd

      IF (DEF_USE_Forcing_Downscaling) THEN
         read(40) svf_patches
         read(40) cur_patches
         read(40) slp_type_patches
         read(40) asp_type_patches
         read(40) area_type_patches
         read(40) sf_lut_patches
      ENDIF

      read(40) zlnd
      read(40) zsno
      read(40) csoilc
      read(40) dewmx
      read(40) wtfact
      read(40) capr
      read(40) cnfac
      read(40) ssi
      read(40) wimp
      read(40) pondmx
      read(40) smpmax
      read(40) smpmin
      read(40) trsmx0
      read(40) tcrit
      read(40) wetwatmax

      close(40)

      IF (p_is_master) THEN
         write(*,'(A29)') 'Loading Time Invariants done.'
      ENDIF

   END SUBROUTINE READ_TimeInvariants

   SUBROUTINE deallocate_TimeInvariants (numpatch)

   USE MOD_Namelist, only: DEF_USE_Forcing_Downscaling 
   USE MOD_SPMD_Task
   !USE MOD_LandPatch, only: numpatch

   IMPLICIT NONE
   integer :: numpatch
      ! --------------------------------------------------
      ! Deallocates memory for CoLM 1d [numpatch] variables
      ! --------------------------------------------------

      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN

            deallocate (patchclass     )
            deallocate (patchtype      )
            deallocate (patchmask      )

            deallocate (patchlonr      )
            deallocate (patchlatr      )

            deallocate (lakedepth      )
            deallocate (dz_lake        )

            deallocate (soil_s_v_alb   )
            deallocate (soil_d_v_alb   )
            deallocate (soil_s_n_alb   )
            deallocate (soil_d_n_alb   )

            deallocate (vf_quartz      )
            deallocate (vf_gravels     )
            deallocate (vf_om          )
            deallocate (vf_sand        )
            deallocate (wf_gravels     )
            deallocate (wf_sand        )
            deallocate (OM_density     )
            deallocate (BD_all         )
            deallocate (wfc            )
            deallocate (porsl          )
            deallocate (psi0           )
            deallocate (bsw            )
            deallocate (theta_r        )
            deallocate (BVIC           )

#ifdef vanGenuchten_Mualem_SOIL_MODEL
            deallocate (alpha_vgm      )
            deallocate (L_vgm          )
            deallocate (n_vgm          )
            deallocate (sc_vgm         )
            deallocate (fc_vgm         )
#endif
            deallocate (vic_b_infilt   )
            deallocate (vic_Dsmax      )
            deallocate (vic_Ds         )
            deallocate (vic_Ws         )
            deallocate (vic_c          )

            deallocate (hksati         )
            deallocate (csol           )
            deallocate (k_solids       )
            deallocate (dksatu         )
            deallocate (dksatf         )
            deallocate (dkdry          )
            deallocate (BA_alpha       )
            deallocate (BA_beta        )

            deallocate (htop           )
            deallocate (hbot           )

            deallocate (dbedrock       )
            deallocate (ibedrock       )

            deallocate (topoelv        )
            deallocate (topostd        )

            IF (DEF_USE_Forcing_Downscaling) THEN
               deallocate(slp_type_patches  )
               deallocate(svf_patches       )
               deallocate(asp_type_patches  )
               deallocate(area_type_patches )
               deallocate(sf_lut_patches    )
               deallocate(cur_patches       )
            ENDIF

         ENDIF
      ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL deallocate_PFTimeInvariants
#endif

#ifdef BGC
      CALL deallocate_BGCTimeInvariants
#endif

#ifdef URBAN_MODEL
      CALL deallocate_UrbanTimeInvariants
#endif
   END SUBROUTINE deallocate_TimeInvariants

#ifdef RangeCheck
   !---------------------------------------
   SUBROUTINE check_TimeInvariants ()

   USE MOD_SPMD_Task
   USE MOD_RangeCheck
   USE MOD_Namelist, only : DEF_USE_BEDROCK, DEF_USE_Forcing_Downscaling

   IMPLICIT NONE

      IF (p_is_master) THEN
         write(*,'(/,A29)') 'Checking Time Invariants ...'
      ENDIF

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

      CALL check_vector_data ('lakedepth    [m]     ', lakedepth   ) !
      CALL check_vector_data ('dz_lake      [m]     ', dz_lake     ) ! new lake scheme

      CALL check_vector_data ('soil_s_v_alb [-]     ', soil_s_v_alb) ! albedo of visible of the saturated soil
      CALL check_vector_data ('soil_d_v_alb [-]     ', soil_d_v_alb) ! albedo of visible of the dry soil
      CALL check_vector_data ('soil_s_n_alb [-]     ', soil_s_n_alb) ! albedo of near infrared of the saturated soil
      CALL check_vector_data ('soil_d_n_alb [-]     ', soil_d_n_alb) ! albedo of near infrared of the dry soil
      CALL check_vector_data ('vf_quartz    [m3/m3] ', vf_quartz   ) ! volumetric fraction of quartz within mineral soil
      CALL check_vector_data ('vf_gravels   [m3/m3] ', vf_gravels  ) ! volumetric fraction of gravels
      CALL check_vector_data ('vf_om        [m3/m3] ', vf_om       ) ! volumetric fraction of organic matter
      CALL check_vector_data ('vf_sand      [m3/m3] ', vf_sand     ) ! volumetric fraction of sand
      CALL check_vector_data ('wf_gravels   [kg/kg] ', wf_gravels  ) ! gravimetric fraction of gravels
      CALL check_vector_data ('wf_sand      [kg/kg] ', wf_sand     ) ! gravimetric fraction of sand
      CALL check_vector_data ('OM_density   [kg/m3] ', OM_density  ) ! OM density
      CALL check_vector_data ('BD_all       [kg/m3] ', BD_all      ) ! bulk density of soils
      CALL check_vector_data ('wfc          [m3/m3] ', wfc         ) ! field capacity
      CALL check_vector_data ('porsl        [m3/m3] ', porsl       ) ! fraction of soil that is voids [-]
      CALL check_vector_data ('psi0         [mm]    ', psi0        ) ! minimum soil suction [mm] (NOTE: "-" valued)
      CALL check_vector_data ('bsw          [-]     ', bsw         ) ! clapp and hornbereger "b" parameter [-]
#ifdef vanGenuchten_Mualem_SOIL_MODEL
      CALL check_vector_data ('theta_r      [m3/m3] ', theta_r     ) ! residual moisture content [-]
      CALL check_vector_data ('alpha_vgm    [-]     ', alpha_vgm   ) ! a parameter corresponding approximately to the inverse of the air-entry value
      CALL check_vector_data ('L_vgm        [-]     ', L_vgm       ) ! pore-connectivity parameter [dimensionless]
      CALL check_vector_data ('n_vgm        [-]     ', n_vgm       ) ! a shape parameter [dimensionless]
      CALL check_vector_data ('sc_vgm       [-]     ', sc_vgm      ) ! saturation at the air entry value in the classical vanGenuchten model [-]
      CALL check_vector_data ('fc_vgm       [-]     ', fc_vgm      ) ! a scaling factor by using air entry value in the Mualem model [-]
#endif
      CALL check_vector_data ('hksati       [mm/s]  ', hksati      ) ! hydraulic conductivity at saturation [mm h2o/s]
      CALL check_vector_data ('csol         [J/m3/K]', csol        ) ! heat capacity of soil solids [J/(m3 K)]
      CALL check_vector_data ('k_solids     [W/m/K] ', k_solids    ) ! thermal conductivity of soil solids [W/m-K]
      CALL check_vector_data ('dksatu       [W/m/K] ', dksatu      ) ! thermal conductivity of unfrozen saturated soil [W/m-K]
      CALL check_vector_data ('dksatf       [W/m/K] ', dksatf      ) ! thermal conductivity of frozen saturated soil [W/m-K]
      CALL check_vector_data ('dkdry        [W/m/K] ', dkdry       ) ! thermal conductivity for dry soil  [W/(m-K)]
      CALL check_vector_data ('BA_alpha     [-]     ', BA_alpha    ) ! alpha in Balland and Arp(2005) thermal conductivity scheme
      CALL check_vector_data ('BA_beta      [-]     ', BA_beta     ) ! beta in Balland and Arp(2005) thermal conductivity scheme

      CALL check_vector_data ('htop         [m]     ', htop        )
      CALL check_vector_data ('hbot         [m]     ', hbot        )

      IF(DEF_USE_BEDROCK)THEN
         CALL check_vector_data ('dbedrock     [m]     ', dbedrock    ) !
      ENDIF

      CALL check_vector_data ('topoelv      [m]     ', topoelv     ) !
      CALL check_vector_data ('topostd      [m]     ', topostd     ) !
      CALL check_vector_data ('BVIC        [-]      ', BVIC        ) !

      IF (DEF_USE_Forcing_Downscaling) THEN
         CALL check_vector_data ('slp_type_patches     [rad] ' , slp_type_patches)      ! slope
         CALL check_vector_data ('svf_patches          [-] '   , svf_patches)           ! sky view factor
         CALL check_vector_data ('asp_type_patches     [rad] ' , asp_type_patches)      ! aspect
         CALL check_vector_data ('area_type_patches    [-] '   , area_type_patches)     ! area percent
         CALL check_vector_data ('cur_patches          [-]'    , cur_patches )
         CALL check_vector_data ('sf_lut_patches       [-] '   , sf_lut_patches)        ! shadow mask
      ENDIF

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

      IF (p_is_master) THEN
         write(*,'(/,A)') 'Checking Constants ...'
         write(*,'(A,E20.10)') 'zlnd   [m]    ', zlnd   ! roughness length for soil [m]
         write(*,'(A,E20.10)') 'zsno   [m]    ', zsno   ! roughness length for snow [m]
         write(*,'(A,E20.10)') 'csoilc [-]    ', csoilc ! drag coefficient for soil under canopy [-]
         write(*,'(A,E20.10)') 'dewmx  [mm]   ', dewmx  ! maximum dew
         write(*,'(A,E20.10)') 'wtfact [-]    ', wtfact ! fraction of model area with high water table
         write(*,'(A,E20.10)') 'capr   [-]    ', capr   ! tuning factor to turn first layer T into surface T
         write(*,'(A,E20.10)') 'cnfac  [-]    ', cnfac  ! Crank Nicholson factor between 0 and 1
         write(*,'(A,E20.10)') 'ssi    [-]    ', ssi    ! irreducible water saturation of snow
         write(*,'(A,E20.10)') 'wimp   [m3/m3]', wimp   ! water impremeable IF porosity less than wimp
         write(*,'(A,E20.10)') 'pondmx [mm]   ', pondmx ! ponding depth (mm)
         write(*,'(A,E20.10)') 'smpmax [mm]   ', smpmax ! wilting point potential in mm
         write(*,'(A,E20.10)') 'smpmin [mm]   ', smpmin ! restriction for min of soil poten. (mm)
         write(*,'(A,E20.10)') 'trsmx0 [mm/s] ', trsmx0 ! max transpiration for moist soil+100% veg.  [mm/s]
         write(*,'(A,E20.10)') 'tcrit  [K]    ', tcrit  ! critical temp. to determine rain or snow
         write(*,'(A,E20.10)') 'wetwatmax [mm]', wetwatmax ! maximum wetland water (mm)
      ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL check_PFTimeInvariants
#endif

#ifdef BGC
      CALL check_BGCTimeInvariants
#endif

   END SUBROUTINE check_TimeInvariants
#endif

END MODULE MOD_Vars_TimeInvariants
! ---------- EOP ------------
