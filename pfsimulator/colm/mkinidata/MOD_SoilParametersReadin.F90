#include <define.h>

MODULE MOD_SoilParametersReadin

!------------------------------------------------------------------------------------------
! DESCRIPTION:
! Read in soil parameters; make unit conversion for soil physical process modeling;
! soil parameters 8 layers => 10 layers
!
! Original author: Yongjiu Dai, 03/2014
!
! Revisions:
! Nan Wei, 01/2019: read more parameters from mksrfdata results
! Shupeng Zhang and Nan Wei, 01/2022: porting codes to parallel version
!------------------------------------------------------------------------------------------

   USE MOD_Precision
   IMPLICIT NONE
   SAVE

   ! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: soil_parameters_readin

CONTAINS

   SUBROUTINE soil_parameters_readin (dir_landdata, lc_year)

   USE MOD_Precision
   USE MOD_Vars_Global, only: nl_soil
   USE MOD_Namelist, only: DEF_SOIL_REFL_SCHEME
   USE MOD_SPMD_Task
   USE MOD_NetCDFVector
   USE MOD_LandPatch
   USE MOD_Vars_TimeInvariants
#ifdef SinglePoint
   USE MOD_SingleSrfdata
#endif
   USE MOD_SoilColorRefl

   IMPLICIT NONE

   ! ----------------------------------------------------------------------
   integer, intent(in) :: lc_year    ! which year of land cover data used
   character(len=*), intent(in) :: dir_landdata

   ! Local Variables

   real(r8), allocatable :: soil_vf_quartz_mineral_s_l (:) ! volumetric fraction of quartz within mineral soil
   real(r8), allocatable :: soil_vf_gravels_s_l        (:) ! volumetric fraction of gravels
   real(r8), allocatable :: soil_vf_om_s_l             (:) ! volumetric fraction of organic matter
   real(r8), allocatable :: soil_vf_sand_s_l           (:) ! volumetric fraction of sand
   real(r8), allocatable :: soil_wf_gravels_s_l        (:) ! gravimetric fraction of gravels
   real(r8), allocatable :: soil_wf_sand_s_l           (:) ! gravimetric fraction of sand
   real(r8), allocatable :: soil_OM_density_s_l        (:) ! OM_density (kg/m3)
   real(r8), allocatable :: soil_BD_all_s_l            (:) ! bulk density of soil (GRAVELS + OM + mineral soils, kg/m3)
   real(r8), allocatable :: soil_theta_s_l             (:) ! saturated water content (cm3/cm3)
   real(r8), allocatable :: soil_psi_s_l               (:) ! matric potential at saturation (cm)
   real(r8), allocatable :: soil_lambda_l              (:) ! pore size distribution index (dimensionless)
#ifdef vanGenuchten_Mualem_SOIL_MODEL
   real(r8), allocatable :: soil_theta_r_l   (:)  ! residual water content (cm3/cm3)
   real(r8), allocatable :: soil_alpha_vgm_l (:)
   real(r8), allocatable :: soil_L_vgm_l     (:)
   real(r8), allocatable :: soil_n_vgm_l     (:)
#endif
   real(r8), allocatable :: soil_k_s_l     (:)  ! saturated hydraulic conductivity (cm/day)
   real(r8), allocatable :: soil_csol_l    (:)  ! heat capacity of soil solids [J/(m3 K)]
   real(r8), allocatable :: soil_k_solids_l(:)  ! thermal conductivity of minerals soil [W/m-K]
   real(r8), allocatable :: soil_tksatu_l  (:)  ! thermal conductivity of saturated unforzen soil [W/m-K]
   real(r8), allocatable :: soil_tksatf_l  (:)  ! thermal conductivity of saturated forzen soil [W/m-K]
   real(r8), allocatable :: soil_tkdry_l   (:)  ! thermal conductivity for dry soil  [W/(m-K)]
   real(r8), allocatable :: soil_BA_alpha_l(:)  ! alpha in Balland and Arp(2005) thermal conductivity scheme
   real(r8), allocatable :: soil_BA_beta_l (:)  ! beta in Balland and Arp(2005) thermal conductivity scheme

   integer  :: ipatch, m, nsl  ! indices

   character(len=256) :: c
   character(len=256) :: landdir, lndname, cyear
   logical :: is_singlepoint

      ! ...............................................................
      write(cyear,'(i4.4)') lc_year
      landdir = trim(dir_landdata) // '/soil/' // trim(cyear)

      ! write(*,*) 'soil parameter readin',landdir
      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN

            allocate ( soil_vf_quartz_mineral_s_l (numpatch) )
            allocate ( soil_vf_gravels_s_l        (numpatch) )
            allocate ( soil_vf_om_s_l             (numpatch) )
            allocate ( soil_vf_sand_s_l           (numpatch) )
            allocate ( soil_wf_gravels_s_l        (numpatch) )
            allocate ( soil_wf_sand_s_l           (numpatch) )
            allocate ( soil_OM_density_s_l        (numpatch) )
            allocate ( soil_BD_all_s_l            (numpatch) )
            allocate ( soil_theta_s_l             (numpatch) )
            allocate ( soil_psi_s_l               (numpatch) )
            allocate ( soil_lambda_l              (numpatch) )
#ifdef vanGenuchten_Mualem_SOIL_MODEL
            allocate ( soil_theta_r_l   (numpatch) )
            allocate ( soil_alpha_vgm_l (numpatch) )
            allocate ( soil_L_vgm_l     (numpatch) )
            allocate ( soil_n_vgm_l     (numpatch) )
#endif
            allocate ( soil_k_s_l     (numpatch) )
            allocate ( soil_csol_l    (numpatch) )
            allocate ( soil_k_solids_l(numpatch) )
            allocate ( soil_tksatu_l  (numpatch) )
            allocate ( soil_tksatf_l  (numpatch) )
            allocate ( soil_tkdry_l   (numpatch) )
            allocate ( soil_BA_alpha_l(numpatch) )
            allocate ( soil_BA_beta_l (numpatch) )
         ENDIF

      ENDIF

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

      DO nsl = 1, 8

#ifdef SinglePoint
         soil_vf_quartz_mineral_s_l (:) = SITE_soil_vf_quartz_mineral (nsl)
         soil_vf_gravels_s_l        (:) = SITE_soil_vf_gravels        (nsl)
         soil_vf_om_s_l             (:) = SITE_soil_vf_om             (nsl)
         soil_vf_sand_s_l           (:) = SITE_soil_vf_sand           (nsl)
         soil_wf_gravels_s_l        (:) = SITE_soil_wf_gravels        (nsl)
         soil_wf_sand_s_l           (:) = SITE_soil_wf_sand           (nsl)
         soil_OM_density_s_l        (:) = SITE_soil_OM_density        (nsl)
         soil_BD_all_s_l            (:) = SITE_soil_BD_all            (nsl)
         soil_theta_s_l             (:) = SITE_soil_theta_s           (nsl)
         soil_psi_s_l               (:) = SITE_soil_psi_s             (nsl)
         soil_lambda_l              (:) = SITE_soil_lambda            (nsl)
#ifdef vanGenuchten_Mualem_SOIL_MODEL
         soil_theta_r_l   (:) = SITE_soil_theta_r  (nsl)
         soil_alpha_vgm_l (:) = SITE_soil_alpha_vgm(nsl)
         soil_L_vgm_l     (:) = SITE_soil_L_vgm    (nsl)
         soil_n_vgm_l     (:) = SITE_soil_n_vgm    (nsl)
#endif
         soil_k_s_l     (:) = SITE_soil_k_s      (nsl)
         soil_csol_l    (:) = SITE_soil_csol     (nsl)
         soil_k_solids_l(:) = SITE_soil_k_solids (nsl)
         soil_tksatu_l  (:) = SITE_soil_tksatu   (nsl)
         soil_tksatf_l  (:) = SITE_soil_tksatf   (nsl)
         soil_tkdry_l   (:) = SITE_soil_tkdry    (nsl)
         soil_BA_alpha_l(:) = SITE_soil_BA_alpha (nsl)
         soil_BA_beta_l (:) = SITE_soil_BA_beta  (nsl)

#else
         write(c,'(i1)') nsl

         ! (1) read in the volumetric fraction of quartz within mineral soil
         lndname = trim(landdir)//'/vf_quartz_mineral_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'vf_quartz_mineral_s_l'//trim(c)//'_patches', landpatch, soil_vf_quartz_mineral_s_l)

         ! (2) read in the volumetric fraction of gravels
         lndname = trim(landdir)//'/vf_gravels_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'vf_gravels_s_l'//trim(c)//'_patches', landpatch, soil_vf_gravels_s_l)

         ! (3) read in the volumetric fraction of organic matter
         lndname = trim(landdir)//'/vf_om_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'vf_om_s_l'//trim(c)//'_patches', landpatch, soil_vf_om_s_l)

         ! (4) read in the volumetric fraction of sand
         lndname = trim(landdir)//'/vf_sand_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'vf_sand_s_l'//trim(c)//'_patches', landpatch, soil_vf_sand_s_l)

         ! (5) read in the gravimetric fraction of gravels
         lndname = trim(landdir)//'/wf_gravels_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'wf_gravels_s_l'//trim(c)//'_patches', landpatch, soil_wf_gravels_s_l)

         ! (6) read in the gravimetric fraction of sand
         lndname = trim(landdir)//'/wf_sand_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'wf_sand_s_l'//trim(c)//'_patches', landpatch, soil_wf_sand_s_l)

         ! (7) read in the saturated water content [cm3/cm3]
         lndname = trim(landdir)//'/theta_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'theta_s_l'//trim(c)//'_patches', landpatch, soil_theta_s_l)

         ! (8) read in the matric potential at saturation [cm]
         lndname = trim(landdir)//'/psi_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'psi_s_l'//trim(c)//'_patches', landpatch, soil_psi_s_l)

         ! (9) read in the pore size distribution index [dimensionless]
         lndname = trim(landdir)//'/lambda_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'lambda_l'//trim(c)//'_patches', landpatch, soil_lambda_l)

#ifdef vanGenuchten_Mualem_SOIL_MODEL
         ! (10) read in residual water content [cm3/cm3]
         lndname = trim(landdir)//'/theta_r_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'theta_r_l'//trim(c)//'_patches', landpatch, soil_theta_r_l)

         ! (11) read in alpha in VGM model
         lndname = trim(landdir)//'/alpha_vgm_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'alpha_vgm_l'//trim(c)//'_patches', landpatch, soil_alpha_vgm_l)

         ! (12) read in L in VGM model
         lndname = trim(landdir)//'/L_vgm_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'L_vgm_l'//trim(c)//'_patches', landpatch, soil_L_vgm_l)

         ! (13) read in n in VGM model
         lndname = trim(landdir)//'/n_vgm_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'n_vgm_l'//trim(c)//'_patches', landpatch, soil_n_vgm_l)
#endif

         ! (14) read in the saturated hydraulic conductivity [cm/day]
         lndname = trim(landdir)//'/k_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'k_s_l'//trim(c)//'_patches', landpatch, soil_k_s_l)

         ! (15) read in the heat capacity of soil solids [J/(m3 K)]
         lndname = trim(landdir)//'/csol_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'csol_l'//trim(c)//'_patches', landpatch, soil_csol_l)

         ! (16) read in the thermal conductivity of unfrozen saturated soil [W/m-K]
         lndname = trim(landdir)//'/tksatu_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'tksatu_l'//trim(c)//'_patches', landpatch, soil_tksatu_l)

         ! (17) read in the thermal conductivity of frozen saturated soil [W/m-K]
         lndname = trim(landdir)//'/tksatf_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'tksatf_l'//trim(c)//'_patches', landpatch, soil_tksatf_l)

         ! (18) read in the thermal conductivity for dry soil [W/(m-K)]
         lndname = trim(landdir)//'/tkdry_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'tkdry_l'//trim(c)//'_patches', landpatch, soil_tkdry_l)

         ! (19) read in the thermal conductivity of solid soil [W/m-K]
         lndname = trim(landdir)//'/k_solids_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'k_solids_l'//trim(c)//'_patches', landpatch, soil_k_solids_l)

         ! (20) read in the parameter alpha in the Balland V. and P. A. Arp (2005) model
         lndname = trim(landdir)//'/BA_alpha_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'BA_alpha_l'//trim(c)//'_patches', landpatch, soil_BA_alpha_l)

         ! (21) read in the parameter beta in the Balland V. and P. A. Arp (2005) model
         lndname = trim(landdir)//'/BA_beta_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'BA_beta_l'//trim(c)//'_patches', landpatch, soil_BA_beta_l)

         ! (22) read in the OM density (kg/m3)
         lndname = trim(landdir)//'/OM_density_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'OM_density_s_l'//trim(c)//'_patches', landpatch, soil_OM_density_s_l)

         ! (23) read in the bulk density of soil (kg/m3)
         lndname = trim(landdir)//'/BD_all_s_l'//trim(c)//'_patches.nc'
         CALL ncio_read_vector (lndname, 'BD_all_s_l'//trim(c)//'_patches', landpatch, soil_BD_all_s_l)

#endif

         IF (p_is_worker) THEN

            DO ipatch = 1, numpatch
               m = landpatch%settyp(ipatch)
               IF( m == 0 )THEN     ! ocean
                  vf_quartz (nsl,ipatch) = -1.e36
                  vf_gravels(nsl,ipatch) = -1.e36
                  vf_om     (nsl,ipatch) = -1.e36
                  vf_sand   (nsl,ipatch) = -1.e36
                  wf_gravels(nsl,ipatch) = -1.e36
                  wf_sand   (nsl,ipatch) = -1.e36
                  OM_density(nsl,ipatch) = -1.e36
                  BD_all    (nsl,ipatch) = -1.e36
                  wfc       (nsl,ipatch) = -1.e36
                  porsl     (nsl,ipatch) = -1.e36
                  psi0      (nsl,ipatch) = -1.e36
                  bsw       (nsl,ipatch) = -1.e36
                  theta_r   (nsl,ipatch) = -1.e36
#ifdef vanGenuchten_Mualem_SOIL_MODEL
                  alpha_vgm (nsl,ipatch) = -1.e36
                  L_vgm     (nsl,ipatch) = -1.e36
                  n_vgm     (nsl,ipatch) = -1.e36
#endif
                  hksati    (nsl,ipatch) = -1.e36
                  csol      (nsl,ipatch) = -1.e36
                  k_solids  (nsl,ipatch) = -1.e36
                  dksatu    (nsl,ipatch) = -1.e36
                  dksatf    (nsl,ipatch) = -1.e36
                  dkdry     (nsl,ipatch) = -1.e36
                  BA_alpha  (nsl,ipatch) = -1.e36
                  BA_beta   (nsl,ipatch) = -1.e36
               ELSE                 ! non ocean
                  vf_quartz  (nsl,ipatch) = soil_vf_quartz_mineral_s_l(ipatch)
                  vf_gravels (nsl,ipatch) = soil_vf_gravels_s_l       (ipatch)
                  vf_om      (nsl,ipatch) = soil_vf_om_s_l            (ipatch)
                  vf_sand    (nsl,ipatch) = soil_vf_sand_s_l          (ipatch)
                  wf_gravels (nsl,ipatch) = soil_wf_gravels_s_l       (ipatch)
                  wf_sand    (nsl,ipatch) = soil_wf_sand_s_l          (ipatch)
                  OM_density (nsl,ipatch) = soil_OM_density_s_l       (ipatch)
                  BD_all     (nsl,ipatch) = soil_BD_all_s_l           (ipatch)
                  porsl      (nsl,ipatch) = soil_theta_s_l            (ipatch)        ! cm/cm
                  psi0       (nsl,ipatch) = soil_psi_s_l              (ipatch) * 10.  ! cm -> mm
                  bsw        (nsl,ipatch) = 1./soil_lambda_l          (ipatch)        ! dimensionless
                  wfc        (nsl,ipatch) = (-339.9/soil_psi_s_l(ipatch))**(-1.0*soil_lambda_l(ipatch))&
                                          * soil_theta_s_l(ipatch)
#ifdef vanGenuchten_Mualem_SOIL_MODEL
                  psi0       (nsl,ipatch) = -10.      ! mm
                  theta_r    (nsl,ipatch) = soil_theta_r_l  (ipatch)
                  alpha_vgm  (nsl,ipatch) = soil_alpha_vgm_l(ipatch)
                  L_vgm      (nsl,ipatch) = soil_L_vgm_l    (ipatch)
                  n_vgm      (nsl,ipatch) = soil_n_vgm_l    (ipatch)
                  wfc        (nsl,ipatch) = soil_theta_r_l  (ipatch)+(soil_theta_s_l(ipatch)-soil_theta_r_l(ipatch))*&
                             (1+(soil_alpha_vgm_l(ipatch)*339.9)**soil_n_vgm_l(ipatch))**(1.0/soil_n_vgm_l(ipatch)-1)
#else
                  theta_r    (nsl,ipatch) = 0.
#endif
                  hksati     (nsl,ipatch) = soil_k_s_l      (ipatch) * 10./86400.  ! cm/day -> mm/s
                  csol       (nsl,ipatch) = soil_csol_l     (ipatch)               ! J/(m2 K)
                  k_solids   (nsl,ipatch) = soil_k_solids_l (ipatch)               ! W/(m K)
                  dksatu     (nsl,ipatch) = soil_tksatu_l   (ipatch)               ! W/(m K)
                  dksatf     (nsl,ipatch) = soil_tksatf_l   (ipatch)               ! W/(m K)
                  dkdry      (nsl,ipatch) = soil_tkdry_l    (ipatch)               ! W/(m K)
                  BA_alpha   (nsl,ipatch) = soil_BA_alpha_l (ipatch)
                  BA_beta    (nsl,ipatch) = soil_BA_beta_l  (ipatch)
               ENDIF
            ENDDO

         ENDIF

      ENDDO

      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN
            deallocate ( soil_vf_quartz_mineral_s_l )
            deallocate ( soil_vf_gravels_s_l        )
            deallocate ( soil_vf_om_s_l             )
            deallocate ( soil_vf_sand_s_l           )
            deallocate ( soil_wf_gravels_s_l        )
            deallocate ( soil_wf_sand_s_l           )
            deallocate ( soil_OM_density_s_l        )
            deallocate ( soil_BD_all_s_l            )
            deallocate ( soil_theta_s_l             )
            deallocate ( soil_psi_s_l               )
            deallocate ( soil_lambda_l              )
#ifdef vanGenuchten_Mualem_SOIL_MODEL
            deallocate ( soil_theta_r_l   )
            deallocate ( soil_alpha_vgm_l )
            deallocate ( soil_L_vgm_l     )
            deallocate ( soil_n_vgm_l     )
#endif
            deallocate ( soil_k_s_l     )
            deallocate ( soil_csol_l    )
            deallocate ( soil_k_solids_l)
            deallocate ( soil_tksatu_l  )
            deallocate ( soil_tksatf_l  )
            deallocate ( soil_tkdry_l   )
            deallocate ( soil_BA_alpha_l)
            deallocate ( soil_BA_beta_l )
         ENDIF

      ENDIF

      ! The parameters of the top NINTH soil layers were given by datasets
      ! [0-0.045 (LAYER 1-2), 0.045-0.091, 0.091-0.166, 0.166-0.289,
      !  0.289-0.493, 0.493-0.829, 0.829-1.383 and 1.383-2.296 m].
      ! The NINTH layer's soil parameters will assigned to the bottom soil layer (2.296 - 3.8019m).

      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN
            DO nsl = 9, 2, -1
               vf_quartz  (nsl,:) = vf_quartz (nsl-1,:)
               vf_gravels (nsl,:) = vf_gravels(nsl-1,:)
               vf_om      (nsl,:) = vf_om     (nsl-1,:)
               vf_sand    (nsl,:) = vf_sand   (nsl-1,:)
               wf_gravels (nsl,:) = wf_gravels(nsl-1,:)
               wf_sand    (nsl,:) = wf_sand   (nsl-1,:)
               OM_density (nsl,:) = OM_density(nsl-1,:)
               BD_all     (nsl,:) = BD_all    (nsl-1,:)
               wfc        (nsl,:) = wfc       (nsl-1,:)
               porsl      (nsl,:) = porsl     (nsl-1,:)
               psi0       (nsl,:) = psi0      (nsl-1,:)
               bsw        (nsl,:) = bsw       (nsl-1,:)
               theta_r    (nsl,:) = theta_r   (nsl-1,:)
#ifdef vanGenuchten_Mualem_SOIL_MODEL
               alpha_vgm  (nsl,:) = alpha_vgm (nsl-1,:)
               L_vgm      (nsl,:) = L_vgm     (nsl-1,:)
               n_vgm      (nsl,:) = n_vgm     (nsl-1,:)
#endif
               hksati     (nsl,:) = hksati    (nsl-1,:)
               csol       (nsl,:) = csol      (nsl-1,:)
               k_solids   (nsl,:) = k_solids  (nsl-1,:)
               dksatu     (nsl,:) = dksatu    (nsl-1,:)
               dksatf     (nsl,:) = dksatf    (nsl-1,:)
               dkdry      (nsl,:) = dkdry     (nsl-1,:)
               BA_alpha   (nsl,:) = BA_alpha  (nsl-1,:)
               BA_beta    (nsl,:) = BA_beta   (nsl-1,:)
            ENDDO

            DO nsl = nl_soil, 10, -1
               vf_quartz  (nsl,:) = vf_quartz (9,:)
               vf_gravels (nsl,:) = vf_gravels(9,:)
               vf_om      (nsl,:) = vf_om     (9,:)
               vf_sand    (nsl,:) = vf_sand   (9,:)
               wf_gravels (nsl,:) = wf_gravels(9,:)
               wf_sand    (nsl,:) = wf_sand   (9,:)
               OM_density (nsl,:) = OM_density(9,:)
               BD_all     (nsl,:) = BD_all    (9,:)
               wfc        (nsl,:) = wfc       (9,:)
               porsl      (nsl,:) = porsl     (9,:)
               psi0       (nsl,:) = psi0      (9,:)
               bsw        (nsl,:) = bsw       (9,:)
               theta_r    (nsl,:) = theta_r   (9,:)
#ifdef vanGenuchten_Mualem_SOIL_MODEL
               alpha_vgm  (nsl,:) = alpha_vgm (9,:)
               L_vgm      (nsl,:) = L_vgm     (9,:)
               n_vgm      (nsl,:) = n_vgm     (9,:)
#endif
               hksati     (nsl,:) = hksati    (9,:)
               csol       (nsl,:) = csol      (9,:)
               k_solids   (nsl,:) = k_solids  (9,:)
               dksatu     (nsl,:) = dksatu    (9,:)
               dksatf     (nsl,:) = dksatf    (9,:)
               dkdry      (nsl,:) = dkdry     (9,:)
               BA_alpha   (nsl,:) = BA_alpha  (9,:)
               BA_beta    (nsl,:) = BA_beta   (9,:)
            ENDDO

         ENDIF
      ENDIF

      ! Soil reflectance of broadband of visible(_v) and near-infrared(_n) of the sarurated(_s) and dry(_d) soil
      ! SCHEME 1: Guessed soil color type according to land cover classes
      IF (DEF_SOIL_REFL_SCHEME .eq. 1) THEN
         IF (p_is_worker) THEN
            DO ipatch = 1, numpatch
               m = landpatch%settyp(ipatch)
               CALL soil_color_refl(m,soil_s_v_alb(ipatch),soil_d_v_alb(ipatch),&
                  soil_s_n_alb(ipatch),soil_d_n_alb(ipatch))
            ENDDO
         ENDIF
      ENDIF

      ! SCHEME 2: Read a global soil color map from CLM
      IF (DEF_SOIL_REFL_SCHEME .eq. 2) THEN

#ifdef SinglePoint
         soil_s_v_alb(:) = SITE_soil_s_v_alb
         soil_d_v_alb(:) = SITE_soil_d_v_alb
         soil_s_n_alb(:) = SITE_soil_s_n_alb
         soil_d_n_alb(:) = SITE_soil_d_n_alb
#else
         ! (1) Read in the albedo of visible of the saturated soil
         lndname = trim(landdir)//'/soil_s_v_alb_patches.nc'
         CALL ncio_read_vector (lndname, 'soil_s_v_alb', landpatch, soil_s_v_alb)

         ! (2) Read in the albedo of visible of the dry soil
         lndname = trim(landdir)//'/soil_d_v_alb_patches.nc'
         CALL ncio_read_vector (lndname, 'soil_d_v_alb', landpatch, soil_d_v_alb)

         ! (3) Read in the albedo of near infrared of the saturated soil
         lndname = trim(landdir)//'/soil_s_n_alb_patches.nc'
         CALL ncio_read_vector (lndname, 'soil_s_n_alb', landpatch, soil_s_n_alb)

         ! (4) Read in the albedo of near infrared of the dry soil
         lndname = trim(landdir)//'/soil_d_n_alb_patches.nc'
         CALL ncio_read_vector (lndname, 'soil_d_n_alb', landpatch, soil_d_n_alb)
#endif
      ENDIF

   END SUBROUTINE soil_parameters_readin

END MODULE MOD_SoilParametersReadin
! --------------------------------------------------
! EOP
