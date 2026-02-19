#include <define.h>

MODULE MOD_IniTimeVariable

!-----------------------------------------------------------------------
   USE MOD_Precision
#ifdef BGC
   USE MOD_BGC_CNSummary, only: CNDriverSummarizeStates, CNDriverSummarizeFluxes
#endif
   IMPLICIT NONE
   SAVE

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: IniTimeVar


CONTAINS

   SUBROUTINE IniTimeVar(ipatch, patchtype&
                     ,porsl,psi0,hksati,soil_s_v_alb,soil_d_v_alb,soil_s_n_alb,soil_d_n_alb&
                     ,z0m,zlnd,htop,z0mr,chil,rho,tau,z_soisno,dz_soisno&
                     ,t_soisno,wliq_soisno,wice_soisno,smp,hk,zwt,wa&
!Plant Hydraulic parameters
                     ,vegwp,gs0sun,gs0sha&
!End plant hydraulic parameter
                     ,t_grnd,tleaf,ldew,ldew_rain,ldew_snow,sag,scv&
                     ,snowdp,fveg,fsno,sigf,green,lai,sai,coszen&
                     ,snw_rds,mss_bcpho,mss_bcphi,mss_ocpho,mss_ocphi&
                     ,mss_dst1,mss_dst2,mss_dst3,mss_dst4&
                     ,alb,ssun,ssha,ssoi,ssno,ssno_lyr,thermk,extkb,extkd&
                     ,trad,tref,qref,rst,emis,zol,rib&
                     ,ustar,qstar,tstar,fm,fh,fq&
#if(defined BGC)
                     ,use_cnini, totlitc, totsomc, totcwdc, decomp_cpools, decomp_cpools_vr, ctrunc_veg, ctrunc_soil, ctrunc_vr &
                     ,totlitn, totsomn, totcwdn, decomp_npools, decomp_npools_vr, ntrunc_veg, ntrunc_soil, ntrunc_vr &
                     ,totvegc, totvegn, totcolc, totcoln, col_endcb, col_begcb, col_endnb, col_begnb &
                     ,col_vegendcb, col_vegbegcb, col_soilendcb, col_soilbegcb &
                     ,col_vegendnb, col_vegbegnb, col_soilendnb, col_soilbegnb &
                     ,col_sminnendnb, col_sminnbegnb &
                     ,altmax, altmax_lastyear, altmax_lastyear_indx, lag_npp &
                     ,sminn_vr, sminn, smin_no3_vr, smin_nh4_vr &
                     ,prec10, prec60, prec365, prec_today, prec_daily, tsoi17, rh30, accumnstep, skip_balance_check &
!------------------------SASU variables-----------------
                     ,decomp0_cpools_vr          , decomp0_npools_vr           &
                     ,I_met_c_vr_acc             , I_cel_c_vr_acc             , I_lig_c_vr_acc             , I_cwd_c_vr_acc              &
                     ,AKX_met_to_soil1_c_vr_acc  , AKX_cel_to_soil1_c_vr_acc  , AKX_lig_to_soil2_c_vr_acc  , AKX_soil1_to_soil2_c_vr_acc &
                     ,AKX_cwd_to_cel_c_vr_acc    , AKX_cwd_to_lig_c_vr_acc    , AKX_soil1_to_soil3_c_vr_acc, AKX_soil2_to_soil1_c_vr_acc &
                     ,AKX_soil2_to_soil3_c_vr_acc, AKX_soil3_to_soil1_c_vr_acc &
                     ,AKX_met_exit_c_vr_acc      , AKX_cel_exit_c_vr_acc      , AKX_lig_exit_c_vr_acc      , AKX_cwd_exit_c_vr_acc       &
                     ,AKX_soil1_exit_c_vr_acc    , AKX_soil2_exit_c_vr_acc    , AKX_soil3_exit_c_vr_acc     &
                     ,diagVX_c_vr_acc            , upperVX_c_vr_acc           , lowerVX_c_vr_acc            &
                     ,I_met_n_vr_acc             , I_cel_n_vr_acc             , I_lig_n_vr_acc             , I_cwd_n_vr_acc              &
                     ,AKX_met_to_soil1_n_vr_acc  , AKX_cel_to_soil1_n_vr_acc  , AKX_lig_to_soil2_n_vr_acc  , AKX_soil1_to_soil2_n_vr_acc &
                     ,AKX_cwd_to_cel_n_vr_acc    , AKX_cwd_to_lig_n_vr_acc    , AKX_soil1_to_soil3_n_vr_acc, AKX_soil2_to_soil1_n_vr_acc &
                     ,AKX_soil2_to_soil3_n_vr_acc, AKX_soil3_to_soil1_n_vr_acc &
                     ,AKX_met_exit_n_vr_acc      , AKX_cel_exit_n_vr_acc      , AKX_lig_exit_n_vr_acc      , AKX_cwd_exit_n_vr_acc       &
                     ,AKX_soil1_exit_n_vr_acc    , AKX_soil2_exit_n_vr_acc    , AKX_soil3_exit_n_vr_acc     &
                     ,diagVX_n_vr_acc            , upperVX_n_vr_acc           , lowerVX_n_vr_acc           &
!------------------------------------------------------------
#endif
                     ,use_soilini, nl_soil_ini, soil_z,   soil_t,   soil_w, use_snowini, snow_d &
                     ,use_wtd,     zwtmm,       zc_soimm, zi_soimm, vliq_r, nprms, prms)

!=======================================================================
! Created by Yongjiu Dai, 09/15/1999
! Revised by Yongjiu Dai, 08/30/2002
!                         03/2014
!=======================================================================

   USE MOD_Precision
   USE MOD_Utils
   USE MOD_Const_Physical, only: tfrz, denh2o, denice
   USE MOD_Vars_TimeVariables, only: tlai, tsai, wdsrf
   USE MOD_Const_PFT, only: isevg, woody, leafcn, frootcn, livewdcn, deadwdcn, slatop
   !USE MOD_Vars_TimeInvariants, only : ibedrock, dbedrock
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_LandPFT, only : patch_pft_s, patch_pft_e
   USE MOD_Vars_PFTimeInvariants
   USE MOD_Vars_PFTimeVariables
#endif
   USE MOD_Vars_Global
   USE MOD_Albedo
   USE MOD_Namelist
   !USE MOD_Hydro_SoilWater
   USE MOD_Hydro_SoilFunction
   !USE MOD_SnowFraction
   USE MOD_SPMD_Task

   IMPLICIT NONE

   integer, intent(in) ::        &!
         ipatch,                 &! patch index
         patchtype                ! index for land cover type [-]

   real(r8), intent(in) ::       &!
         fveg,                   &! fraction of vegetation cover
         green,                  &! leaf greenness
         coszen,                 &! cosine of solar zenith angle
         soil_s_v_alb,           &! albedo of visible of the saturated soil
         soil_d_v_alb,           &! albedo of visible of the dry soil
         soil_s_n_alb,           &! albedo of near infrared of the saturated soil
         soil_d_n_alb,           &! albedo of near infrared of the dry soil
         zlnd,                   &! aerodynamic roughness length over soil surface [m]
         z0mr,                   &! ratio to calculate roughness length z0m
         htop,                   &! Caonpy top height [m]
         chil,                   &! leaf angle distribution factor
         rho(2,2),               &! leaf reflectance (iw=iband, il=life and dead)
         tau(2,2),               &! leaf transmittance (iw=iband, il=life and dead)
         porsl(1:nl_soil),       &! porosity of soil
         psi0 (1:nl_soil),       &! saturated soil suction (mm) (NEGATIVE)
         hksati(1:nl_soil)        ! hydraulic conductivity at saturation [mm h2o/s]

   real(r8), intent(inout) ::    &
         z0m                      ! aerodynamic roughness length [m]

   logical, intent(in)  :: use_soilini
#ifdef BGC
   logical, intent(in)  :: use_cnini
#endif
   integer, intent(in)  :: nl_soil_ini
   real(r8), intent(in) ::       &!
         soil_z(nl_soil_ini),    &! soil layer depth for initial (m)
         soil_t(nl_soil_ini),    &! soil temperature from initial file (K)
         soil_w(nl_soil_ini)      ! soil wetness from initial file (-)

   logical,  intent(in) :: use_snowini
   real(r8), intent(in) :: snow_d ! snow depth (m)

   logical,  intent(in) :: use_wtd
   real(r8), intent(in) :: zwtmm
   real(r8), intent(in) :: zc_soimm(1:nl_soil)
   real(r8), intent(in) :: zi_soimm(0:nl_soil)
   real(r8), intent(in) :: vliq_r  (1:nl_soil)
   integer,  intent(in) :: nprms
   real(r8), intent(in) :: prms(nprms, 1:nl_soil)

   real(r8), intent(inout) ::    &!
         z_soisno (maxsnl+1:nl_soil),   &! node depth [m]
         dz_soisno(maxsnl+1:nl_soil)     ! layer thickness [m]

   real(r8), intent(out) ::      &!
         t_soisno (maxsnl+1:nl_soil),   &! soil temperature [K]
         wliq_soisno(maxsnl+1:nl_soil), &! liquid water in layers [kg/m2]
         wice_soisno(maxsnl+1:nl_soil), &! ice lens in layers [kg/m2]
         smp        (1:nl_soil)       , &! soil matrix potential
         hk         (1:nl_soil)       , &! soil hydraulic conductance
!Plant Hydraulic parameters
         vegwp(1:nvegwcs),       &! vegetation water potential
         gs0sun,                 &! working copy of sunlit stomata conductance
         gs0sha,                 &! working copy of shalit stomata conductance
!end plant hydraulic parameters
         t_grnd,                 &! ground surface temperature [K]
         tleaf,                  &! sunlit leaf temperature [K]
!#ifdef CLM5_INTERCEPTION
         ldew_rain,              &! depth of rain on foliage [mm]
         ldew_snow,              &! depth of snow on foliage [mm]
!#endif
         ldew,                   &! depth of water on foliage [mm]
         sag,                    &! non dimensional snow age [-]
         scv,                    &! snow cover, water equivalent [mm]
         snowdp,                 &! snow depth [meter]
         fsno,                   &! fraction of snow cover on ground
         sigf,                   &! fraction of veg cover, excluding snow-covered veg [-]
         lai,                    &! leaf area index
         sai,                    &! stem area index

         alb (2,2),              &! averaged albedo [-]
         ssun(2,2),              &! sunlit canopy absorption for solar radiation
         ssha(2,2),              &! shaded canopy absorption for solar radiation
         ssoi(2,2),              &! ground soil absorption [-]
         ssno(2,2),              &! ground snow absorption [-]
         thermk,                 &! canopy gap fraction for tir radiation
         extkb,                  &! (k, g(mu)/mu) direct solar extinction coefficient
         extkd,                  &! diffuse and scattered diffuse PAR extinction coefficient
         wa                       ! water storage in aquifer [mm]
   real(r8), intent(inout) ::    &!
         zwt                      ! the depth to water table [m]

   real(r8), intent(out) ::      &!
         snw_rds  ( maxsnl+1:0 ), &! effective grain radius (col,lyr) [microns, m-6]
         mss_bcphi( maxsnl+1:0 ), &! mass concentration of hydrophilic BC (col,lyr) [kg/kg]
         mss_bcpho( maxsnl+1:0 ), &! mass concentration of hydrophobic BC (col,lyr) [kg/kg]
         mss_ocphi( maxsnl+1:0 ), &! mass concentration of hydrophilic OC (col,lyr) [kg/kg]
         mss_ocpho( maxsnl+1:0 ), &! mass concentration of hydrophobic OC (col,lyr) [kg/kg]
         mss_dst1 ( maxsnl+1:0 ), &! mass concentration of dust aerosol species 1 (col,lyr) [kg/kg]
         mss_dst2 ( maxsnl+1:0 ), &! mass concentration of dust aerosol species 2 (col,lyr) [kg/kg]
         mss_dst3 ( maxsnl+1:0 ), &! mass concentration of dust aerosol species 3 (col,lyr) [kg/kg]
         mss_dst4 ( maxsnl+1:0 ), &! mass concentration of dust aerosol species 4 (col,lyr) [kg/kg]
         ssno_lyr (2,2,maxsnl+1:1 ), &! snow layer absorption [-]

                     ! Additional variables required by reginal model (WRF & RSM)
                     ! ---------------------------------------------------------
         trad,                   &! radiative temperature of surface [K]
         tref,                   &! 2 m height air temperature [kelvin]
         qref,                   &! 2 m height air specific humidity
         rst,                    &! canopy stomatal resistance (s/m)
         emis,                   &! averaged bulk surface emissivity
         zol,                    &! dimensionless height (z/L) used in Monin-Obukhov theory
         rib,                    &! bulk Richardson number in surface layer
         ustar,                  &! u* in similarity theory [m/s]
         qstar,                  &! q* in similarity theory [kg/kg]
         tstar,                  &! t* in similarity theory [K]
         fm,                     &! integral of profile function for momentum
         fh,                     &! integral of profile function for heat
         fq                       ! integral of profile function for moisture

#ifdef BGC
   real(r8),intent(out) ::      &
        totlitc               , &
        totsomc               , &
        totcwdc               , &
        totvegc               , &
        totcolc               , &
        totlitn               , &
        totsomn               , &
        totcwdn               , &
        totvegn               , &
        totcoln               , &
        col_endcb             , &
        col_begcb             , &
        col_vegendcb          , &
        col_vegbegcb          , &
        col_soilendcb         , &
        col_soilbegcb         , &
        col_endnb             , &
        col_begnb             , &
        col_vegendnb          , &
        col_vegbegnb          , &
        col_soilendnb         , &
        col_soilbegnb         , &
        col_sminnendnb        , &
        col_sminnbegnb        , &
        decomp_cpools_vr          (nl_soil_full,ndecomp_pools), &
        decomp_cpools             (ndecomp_pools)             , &
        ctrunc_vr                 (nl_soil)              , &
        ctrunc_veg            , &
        ctrunc_soil           , &
        altmax                , &
        altmax_lastyear       , &
        lag_npp
   integer, intent(out) :: altmax_lastyear_indx
   real(r8),intent(out) ::      &
        decomp_npools_vr          (nl_soil_full,ndecomp_pools), &
        decomp_npools             (ndecomp_pools)             , &
        ntrunc_vr                 (nl_soil)              , &
        ntrunc_veg            , &
        ntrunc_soil           , &
        sminn_vr                  (nl_soil)                   , &
        sminn                 , &
        smin_no3_vr               (nl_soil)                   , &
        smin_nh4_vr               (nl_soil)                   , &
        prec10                                                , &
        prec60                                                , &
        prec365                                               , &
        prec_today                                            , &
        prec_daily                (365)                       , &
        tsoi17                                                , &
        rh30                                                  , &
        accumnstep

 !---------------SASU variables-----------------------
   real(r8),intent(out) ::      &
        decomp0_cpools_vr          (nl_soil,ndecomp_pools)  , &
        decomp0_npools_vr          (nl_soil,ndecomp_pools)  , &

        I_met_c_vr_acc             (nl_soil)  , &
        I_cel_c_vr_acc             (nl_soil)  , &
        I_lig_c_vr_acc             (nl_soil)  , &
        I_cwd_c_vr_acc             (nl_soil)  , &
        AKX_met_to_soil1_c_vr_acc  (nl_soil)  , &
        AKX_cel_to_soil1_c_vr_acc  (nl_soil)  , &
        AKX_lig_to_soil2_c_vr_acc  (nl_soil)  , &
        AKX_soil1_to_soil2_c_vr_acc(nl_soil)  , &
        AKX_cwd_to_cel_c_vr_acc    (nl_soil)  , &
        AKX_cwd_to_lig_c_vr_acc    (nl_soil)  , &
        AKX_soil1_to_soil3_c_vr_acc(nl_soil)  , &
        AKX_soil2_to_soil1_c_vr_acc(nl_soil)  , &
        AKX_soil2_to_soil3_c_vr_acc(nl_soil)  , &
        AKX_soil3_to_soil1_c_vr_acc(nl_soil)  , &
        AKX_met_exit_c_vr_acc      (nl_soil)  , &
        AKX_cel_exit_c_vr_acc      (nl_soil)  , &
        AKX_lig_exit_c_vr_acc      (nl_soil)  , &
        AKX_cwd_exit_c_vr_acc      (nl_soil)  , &
        AKX_soil1_exit_c_vr_acc    (nl_soil)  , &
        AKX_soil2_exit_c_vr_acc    (nl_soil)  , &
        AKX_soil3_exit_c_vr_acc    (nl_soil)  , &

        I_met_n_vr_acc             (nl_soil)  , &
        I_cel_n_vr_acc             (nl_soil)  , &
        I_lig_n_vr_acc             (nl_soil)  , &
        I_cwd_n_vr_acc             (nl_soil)  , &
        AKX_met_to_soil1_n_vr_acc  (nl_soil)  , &
        AKX_cel_to_soil1_n_vr_acc  (nl_soil)  , &
        AKX_lig_to_soil2_n_vr_acc  (nl_soil)  , &
        AKX_soil1_to_soil2_n_vr_acc(nl_soil)  , &
        AKX_cwd_to_cel_n_vr_acc    (nl_soil)  , &
        AKX_cwd_to_lig_n_vr_acc    (nl_soil)  , &
        AKX_soil1_to_soil3_n_vr_acc(nl_soil)  , &
        AKX_soil2_to_soil1_n_vr_acc(nl_soil)  , &
        AKX_soil2_to_soil3_n_vr_acc(nl_soil)  , &
        AKX_soil3_to_soil1_n_vr_acc(nl_soil)  , &
        AKX_met_exit_n_vr_acc      (nl_soil)  , &
        AKX_cel_exit_n_vr_acc      (nl_soil)  , &
        AKX_lig_exit_n_vr_acc      (nl_soil)  , &
        AKX_cwd_exit_n_vr_acc      (nl_soil)  , &
        AKX_soil1_exit_n_vr_acc    (nl_soil)  , &
        AKX_soil2_exit_n_vr_acc    (nl_soil)  , &
        AKX_soil3_exit_n_vr_acc    (nl_soil)  , &

        diagVX_c_vr_acc            (nl_soil,ndecomp_pools)  , &
        upperVX_c_vr_acc           (nl_soil,ndecomp_pools)  , &
        lowerVX_c_vr_acc           (nl_soil,ndecomp_pools)  , &
        diagVX_n_vr_acc            (nl_soil,ndecomp_pools)  , &
        upperVX_n_vr_acc           (nl_soil,ndecomp_pools)  , &
        lowerVX_n_vr_acc           (nl_soil,ndecomp_pools)

   !----------------------------------------------------
   logical, intent(out) :: &
        skip_balance_check

#endif

   integer j, snl, m, ivt
   real(r8) wet(nl_soil), zi_soi_a(0:nl_soil), psi, vliq, wt, ssw, oro, rhosno_ini, a

   ! SNICAR
   real(r8) pg_snow                 ! snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8) snofrz     (maxsnl+1:0) ! snow freezing rate (col,lyr) [kg m-2 s-1]

   integer ps, pe

      !-----------------------------------------------------------------------
      IF(patchtype <= 5)THEN ! land grid

         ! (1) SOIL temperature, water and SNOW
         ! Variables: t_soisno, wliq_soisno, wice_soisno
         !            snowdp, sag, scv, fsno, snl, z_soisno, dz_soisno
         IF (use_soilini) THEN

            zi_soi_a(:) = (/0._r8, zi_soi/)

            DO j = 1, nl_soil
               CALL polint(soil_z,soil_t,nl_soil_ini,z_soisno(j),t_soisno(j))
            ENDDO

            IF (patchtype <= 1) THEN ! soil or urban

               DO j = 1, nl_soil

                  CALL polint(soil_z,soil_w,nl_soil_ini,z_soisno(j),wet(j))

                  wet(j) = min(max(wet(j),0.), porsl(j))

                  IF (zwt <= zi_soi_a(j-1))  THEN
                     wet(j) = porsl(j)
                  ELSEIF (zwt < zi_soi_a(j)) THEN
                     wet(j) = ((zi_soi_a(j)-zwt)*porsl(j) + (zwt-zi_soi_a(j-1))*wet(j)) &
                        / (zi_soi_a(j)-zi_soi_a(j-1))
                  ENDIF

                  IF(t_soisno(j).ge.tfrz)THEN
                     wliq_soisno(j) = wet(j)*dz_soisno(j)*denh2o
                     wice_soisno(j) = 0.
                  ELSE
                     wliq_soisno(j) = 0.
                     wice_soisno(j) = wet(j)*dz_soisno(j)*denice
                  ENDIF
               ENDDO

               ! get wa from zwt
               IF (zwt > zi_soi_a(nl_soil)) THEN
                  psi  = psi0(nl_soil) - (zwt*1000. - zi_soi_a(nl_soil)*1000.) * 0.5
                  vliq = soil_vliq_from_psi (psi, porsl(nl_soil), vliq_r(nl_soil), psi0(nl_soil), &
                     nprms, prms(:,nl_soil))
                  wa   = -(zwt*1000. - zi_soi_a(nl_soil)*1000.)*(porsl(nl_soil)-vliq)
               ELSE
                  wa = 0.
               ENDIF

            ELSEIF ((patchtype == 2) .or. (patchtype == 4)) THEN ! (2) wetland or (4) lake

               DO j = 1, nl_soil
                  IF(t_soisno(j).ge.tfrz)THEN
                     wliq_soisno(j) = porsl(j)*dz_soisno(j)*denh2o
                     wice_soisno(j) = 0.
                  ELSE
                     wliq_soisno(j) = 0.
                     wice_soisno(j) = porsl(j)*dz_soisno(j)*denice
                  ENDIF
               ENDDO

               wa = 0.

            ELSEIF (patchtype == 3) THEN ! land ice

               DO j = 1, nl_soil
                  wliq_soisno(j) = 0.
                  wice_soisno(j) = dz_soisno(j)*denice
               ENDDO

               wa = 0.

            ENDIF

            IF (.not. DEF_USE_VariablySaturatedFlow) THEN
               wa = wa + 5000.
            ENDIF

         ELSE

            ! soil temperature, water content
            DO j = 1, nl_soil
               IF(patchtype==3)THEN !land ice
                  t_soisno(j) = 253.
                  wliq_soisno(j) = 0.
                  wice_soisno(j) = dz_soisno(j)*denice
               ELSE
                  t_soisno(j) = 283.
                  wliq_soisno(j) = dz_soisno(j)*porsl(j)*denh2o
                  wice_soisno(j) = 0.
               ENDIF
            ENDDO

         ENDIF

         z0m = htop * z0mr
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
         IF(patchtype==0)THEN
            ps = patch_pft_s(ipatch)
            pe = patch_pft_e(ipatch)
            IF (ps>0 .and. pe>0) THEN
               z0m_p(ps:pe) = htop_p(ps:pe) * z0mr
            ENDIF
         ENDIF
#endif

         IF (use_snowini) THEN

            rhosno_ini = 250.
            snowdp = snow_d
            sag    = 0.
            scv    = snowdp*rhosno_ini

            !! 08/02/2019, yuan: NOTE! need to be changed in future.
            !! 12/05/2023, yuan: DONE for snowini, change sai.
            !CALL snowfraction (tlai(ipatch),tsai(ipatch),z0m,zlnd,scv,snowdp,wt,sigf,fsno)
            !CALL snow_ini (patchtype,maxsnl,snowdp,snl,z_soisno,dz_soisno)

            lai = tlai(ipatch)
            sai = tsai(ipatch) * sigf

            IF (patchtype == 0) THEN
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
               ps = patch_pft_s(ipatch)
               pe = patch_pft_e(ipatch)
               CALL snowfraction_pftwrap (ipatch,zlnd,scv,snowdp,wt,sigf,fsno)
               IF(DEF_USE_LAIFEEDBACK)THEN
                  lai = sum(lai_p(ps:pe)*pftfrac(ps:pe))
               ELSE
                  lai_p(ps:pe) = tlai_p(ps:pe)
                  lai = tlai(ipatch)
               ENDIF
               sai_p(ps:pe) = tsai_p(ps:pe) * sigf_p(ps:pe)
               sai = sum(sai_p(ps:pe)*pftfrac(ps:pe))
#endif
            ENDIF

            IF(snl.lt.0)THEN
               DO j = snl+1, 0
                  t_soisno(j) = min(tfrz-1., t_soisno(1))
                  wliq_soisno(j) = 0.
                  wice_soisno(j) = dz_soisno(j)*rhosno_ini         !m * kg m-3 = kg m-2
               ENDDO
            ENDIF

            IF(snl>maxsnl)THEN
               t_soisno   (maxsnl+1:snl) = -999.
               wice_soisno(maxsnl+1:snl) = 0.
               wliq_soisno(maxsnl+1:snl) = 0.
               z_soisno   (maxsnl+1:snl) = 0.
               dz_soisno  (maxsnl+1:snl) = 0.
            ENDIF

         ELSE

            snowdp = 0.
            sag    = 0.
            scv    = 0.
            fsno   = 0.
            snl    = 0

            ! snow temperature and water content
            t_soisno   (maxsnl+1:0) = -999.
            wice_soisno(maxsnl+1:0) = 0.
            wliq_soisno(maxsnl+1:0) = 0.
            z_soisno   (maxsnl+1:0) = 0.
            dz_soisno  (maxsnl+1:0) = 0.

         ENDIF

         ! (2) SOIL aquifer and water table
         ! Variables: wa, zwt
         IF (.not. use_wtd) THEN

            IF (.not. use_soilini) THEN
               IF (DEF_USE_VariablySaturatedFlow) THEN
                  wa  = 0.
                  zwt = zi_soimm(nl_soil)/1000.
               ELSE
                  ! water table depth (initially at 1.0 m below the model bottom; wa when zwt
                  !                    is below the model bottom zi(nl_soil)
                  wa  = 4800.                             !assuming aquifer capacity is 5000 mm
                  zwt = (25. + z_soisno(nl_soil))+dz_soisno(nl_soil)/2. - wa/1000./0.2 !to result in zwt = zi(nl_soil) + 1.0 m
               ENDIF
            ENDIF
         ELSE
            IF (patchtype <= 1) THEN
               !CALL get_water_equilibrium_state (zwtmm, nl_soil, wliq_soisno(1:nl_soil), smp, hk, wa, &
               !   zc_soimm, zi_soimm, porsl, vliq_r, psi0, hksati, nprms, prms)
            ELSE
               wa  = 0.
               zwt = 0.
            ENDIF

            IF (.not. DEF_USE_VariablySaturatedFlow) THEN
               wa = wa + 5000.
            ENDIF
         ENDIF

         ! (3) soil matrix potential hydraulic conductivity
         ! Variables: smp, hk
         DO j = 1, nl_soil
            IF ((patchtype==3) .or. (t_soisno(j) < tfrz)) THEN !land ice or frozen soil
               smp(j) = 1.e3 * 0.3336e6/9.80616*(t_soisno(j)-tfrz)/t_soisno(j)
               hk(j)  = 0.
            ELSE
               vliq   = wliq_soisno(j) / (zi_soimm(j)-zi_soimm(j-1))
               smp(j) = soil_psi_from_vliq (vliq, porsl(j), vliq_r(j), psi0(j), nprms, prms(:,j))
               hk (j) = soil_hk_from_psi   (smp(j), psi0(j), hksati(j), nprms, prms(:,j))
            ENDIF
         ENDDO

         ! (4) Vegetation water and temperature
         ! Variables: ldew_rain, ldew_snow, ldew, t_leaf, vegwp, gs0sun, gs0sha
         ldew_rain  = 0.
         ldew_snow  = 0.
         ldew  = 0.
         tleaf = t_soisno(1)
         IF(DEF_USE_PLANTHYDRAULICS)THEN
            vegwp(1:nvegwcs) = -2.5e4
            gs0sun = 1.0e4
            gs0sha = 1.0e4
         ENDIF

         IF (patchtype == 0) THEN
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
            ps = patch_pft_s(ipatch)
            pe = patch_pft_e(ipatch)
            ldew_rain_p(ps:pe) = 0.
            ldew_snow_p(ps:pe) = 0.
            ldew_p(ps:pe) = 0.
            tleaf_p(ps:pe)= t_soisno(1)
            tref_p(ps:pe) = t_soisno(1)
            qref_p(ps:pe) = 0.3
            IF(DEF_USE_PLANTHYDRAULICS)THEN
               vegwp_p(1:nvegwcs,ps:pe) = -2.5e4
               gs0sun_p(ps:pe) = 1.0e4
               gs0sha_p(ps:pe) = 1.0e4
            ENDIF
#endif
         ENDIF

         ! (5) Ground
         ! Variables: t_grnd, wdsrf
         t_grnd = t_soisno(1)
         wdsrf  = 0.

         ! (6) Leaf area
         ! Variables: sigf, lai, sai

         IF (.not. use_snowini) THEN
            IF (patchtype == 0) THEN
#if (defined LULC_USGS || defined LULC_IGBP)
               sigf = fveg
               lai  = tlai(ipatch)
               sai  = tsai(ipatch) * sigf
#endif

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
               ps = patch_pft_s(ipatch)
               pe = patch_pft_e(ipatch)
               sigf_p (ps:pe)  = 1.
               lai_p(ps:pe)    = tlai_p(ps:pe)
               sai_p(ps:pe)    = tsai_p(ps:pe) * sigf_p(ps:pe)

               sigf  = 1.
               lai   = tlai(ipatch)
               sai   = sum(sai_p(ps:pe) * pftfrac(ps:pe))
#endif
            ELSE
               sigf  = fveg
               lai   = tlai(ipatch)
               sai   = tsai(ipatch) * sigf
            ENDIF
         ENDIF

         ! (7) SNICAR
         ! Variables: snw_rds, mss_bcpho, mss_bcphi, mss_ocpho, mss_ocphi,
         !            mss_dst1, mss_dst2, mss_dst3, mss_dst4
         snw_rds   (:) = 54.526_r8
         mss_bcpho (:) = 0.
         mss_bcphi (:) = 0.
         mss_ocpho (:) = 0.
         mss_ocphi (:) = 0.
         mss_dst1  (:) = 0.
         mss_dst2  (:) = 0.
         mss_dst3  (:) = 0.
         mss_dst4  (:) = 0.

#ifdef BGC
         totlitc                         = 0.0
         totsomc                         = 0.0
         totcwdc                         = 0.0
         totvegc                         = 0.0
         totcolc                         = 0.0
         totlitn                         = 0.0
         totsomn                         = 0.0
         totcwdn                         = 0.0
         totvegn                         = 0.0
         col_endcb                       = 0.0
         col_begcb                       = 0.0
         col_vegendcb                    = 0.0
         col_vegbegcb                    = 0.0
         col_soilendcb                   = 0.0
         col_soilbegcb                   = 0.0
         col_endnb                       = 0.0
         col_begnb                       = 0.0
         col_vegendnb                    = 0.0
         col_vegbegnb                    = 0.0
         col_soilendnb                   = 0.0
         col_soilbegnb                   = 0.0
         IF(.not. use_cnini)THEN
            decomp_cpools_vr          (:,:) = 0.0
         ENDIF
         decomp_cpools             (:)   = 0.0
         ctrunc_vr                 (:)   = 0.0
         ctrunc_veg                      = 0.0
         ctrunc_soil                     = 0.0
         altmax                          = 10.0
         altmax_lastyear                 = 10.0
         altmax_lastyear_indx            = 10
         lag_npp                         = 0.0
         IF(.not. use_cnini)THEN
            decomp_npools_vr          (:,:) = 0.0
         ENDIF
         decomp_npools             (:)   = 0.0
         ntrunc_vr                 (:)   = 0.0
         ntrunc_veg                      = 0.0
         ntrunc_soil                     = 0.0
         IF(.not. use_cnini)THEN
            smin_no3_vr               (:)   = 5.0
            smin_nh4_vr               (:)   = 5.0
            sminn_vr                  (:)   = 10.0
         ENDIF
         sminn                           = 0.0
         DO j = 1, nl_soil
            sminn                        = sminn + sminn_vr(j) * dz_soisno(j)
         ENDDO
         col_sminnendnb                  = sminn
         col_sminnbegnb                  = sminn
         totcoln                         = totvegn + totcwdn + totlitn + totsomn + sminn + ntrunc_veg + ntrunc_soil
         prec10                          = 0._r8
         prec60                          = 0._r8
         prec365                         = 0._r8
         prec_today                      = 0._r8
         prec_daily                (:)   = 0._r8
         tsoi17                          = 273.15_r8
         rh30                            = 0._r8
         accumnstep                      = 0._r8

         !---------------SASU variables-----------------------
         decomp0_cpools_vr         (:,:) = 0.0
         I_met_c_vr_acc              (:) = 0.0
         I_cel_c_vr_acc              (:) = 0.0
         I_lig_c_vr_acc              (:) = 0.0
         I_cwd_c_vr_acc              (:) = 0.0
         AKX_met_to_soil1_c_vr_acc   (:) = 0.0
         AKX_cel_to_soil1_c_vr_acc   (:) = 0.0
         AKX_lig_to_soil2_c_vr_acc   (:) = 0.0
         AKX_soil1_to_soil2_c_vr_acc (:) = 0.0
         AKX_cwd_to_cel_c_vr_acc     (:) = 0.0
         AKX_cwd_to_lig_c_vr_acc     (:) = 0.0
         AKX_soil1_to_soil3_c_vr_acc (:) = 0.0
         AKX_soil2_to_soil1_c_vr_acc (:) = 0.0
         AKX_soil2_to_soil3_c_vr_acc (:) = 0.0
         AKX_soil3_to_soil1_c_vr_acc (:) = 0.0
         AKX_met_exit_c_vr_acc       (:) = 0.0
         AKX_cel_exit_c_vr_acc       (:) = 0.0
         AKX_lig_exit_c_vr_acc       (:) = 0.0
         AKX_cwd_exit_c_vr_acc       (:) = 0.0
         AKX_soil1_exit_c_vr_acc     (:) = 0.0
         AKX_soil2_exit_c_vr_acc     (:) = 0.0
         AKX_soil3_exit_c_vr_acc     (:) = 0.0

         decomp0_npools_vr         (:,:) = 0.0
         I_met_n_vr_acc              (:) = 0.0
         I_cel_n_vr_acc              (:) = 0.0
         I_lig_n_vr_acc              (:) = 0.0
         I_cwd_n_vr_acc              (:) = 0.0
         AKX_met_to_soil1_n_vr_acc   (:) = 0.0
         AKX_cel_to_soil1_n_vr_acc   (:) = 0.0
         AKX_lig_to_soil2_n_vr_acc   (:) = 0.0
         AKX_soil1_to_soil2_n_vr_acc (:) = 0.0
         AKX_cwd_to_cel_n_vr_acc     (:) = 0.0
         AKX_cwd_to_lig_n_vr_acc     (:) = 0.0
         AKX_soil1_to_soil3_n_vr_acc (:) = 0.0
         AKX_soil2_to_soil1_n_vr_acc (:) = 0.0
         AKX_soil2_to_soil3_n_vr_acc (:) = 0.0
         AKX_soil3_to_soil1_n_vr_acc (:) = 0.0
         AKX_met_exit_n_vr_acc       (:) = 0.0
         AKX_cel_exit_n_vr_acc       (:) = 0.0
         AKX_lig_exit_n_vr_acc       (:) = 0.0
         AKX_cwd_exit_n_vr_acc       (:) = 0.0
         AKX_soil1_exit_n_vr_acc     (:) = 0.0
         AKX_soil2_exit_n_vr_acc     (:) = 0.0
         AKX_soil3_exit_n_vr_acc     (:) = 0.0

         diagVX_c_vr_acc           (:,:) = 0.0
         upperVX_c_vr_acc          (:,:) = 0.0
         lowerVX_c_vr_acc          (:,:) = 0.0
         diagVX_n_vr_acc           (:,:) = 0.0
         upperVX_n_vr_acc          (:,:) = 0.0
         lowerVX_n_vr_acc          (:,:) = 0.0

         !----------------------------------------------------
         skip_balance_check              = .false.

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
         IF (patchtype == 0) THEN
            DO m = ps, pe
               ivt = pftclass(m)
               IF(ivt .eq.  0)THEN  !no vegetation
                  leafc_p                  (m) = 0.0
                  leafc_storage_p          (m) = 0.0
                  leafn_p                  (m) = 0.0
                  leafn_storage_p          (m) = 0.0
                  frootc_p                 (m) = 0.0
                  frootc_storage_p         (m) = 0.0
                  frootn_p                 (m) = 0.0
                  frootn_storage_p         (m) = 0.0
               ELSE
                  IF(isevg(ivt))THEN
                     IF(.not. use_cnini)THEN
                        leafc_p            (m) = 100.0
                        frootc_p           (m) = 0.0
                     ENDIF
                     leafc_storage_p       (m) = 0.0
                     frootc_storage_p      (m) = 0.0
                  ELSE IF(ivt >= npcropmin) THEN
                     leafc_p               (m) = 0.0
                     leafc_storage_p       (m) = 0.0
                     frootc_p              (m) = 0.0
                     frootc_storage_p      (m) = 0.0
                  ELSE
                     IF(.not. use_cnini)THEN
                        leafc_p            (m) = 0.0
                        leafc_storage_p    (m) = 100.0
                        frootc_p           (m) = 0.0
                        frootc_storage_p   (m) = 0.0
                     ENDIF
                  ENDIF
                  leafn_p                     (m) = leafc_p        (m) / leafcn  (ivt)
                  leafn_storage_p             (m) = leafc_storage_p(m) / leafcn  (ivt)
                  frootn_p                    (m) = frootc_p        (m) / frootcn  (ivt)
                  frootn_storage_p            (m) = frootc_storage_p(m) / frootcn  (ivt)
               ENDIF
               IF(woody(ivt) .eq. 1)THEN
                  IF(.not. use_cnini)THEN
                     deadstemc_p              (m) = 0.1
                     livestemc_p              (m) = 0.0
                     deadcrootc_p             (m) = 0.0
                     livecrootc_p             (m) = 0.0
                  ENDIF
                  livestemn_p                 (m) = livestemc_p    (m) / livewdcn(ivt)
                  deadstemn_p                 (m) = deadstemc_p    (m) / deadwdcn(ivt)
                  livecrootn_p                (m) = livecrootc_p   (m) / livewdcn(ivt)
                  deadcrootn_p                (m) = deadcrootc_p   (m) / deadwdcn(ivt)
               ELSE
                  livestemc_p                 (m) = 0.0
                  deadstemc_p                 (m) = 0.0
                  livestemn_p                 (m) = 0.0
                  deadstemn_p                 (m) = 0.0
                  livecrootc_p                (m) = 0.0
                  deadcrootc_p                (m) = 0.0
                  livecrootn_p                (m) = 0.0
                  deadcrootn_p                (m) = 0.0
               ENDIF
               !            totcolc = totcolc + (leafc_p(m) + leafc_storage_p(m) + deadstemc_p(m))* pftfrac(m)
               !            totvegc = totvegc + (leafc_p(m) + leafc_storage_p(m) + deadstemc_p(m))* pftfrac(m)
               !            totcoln = totcoln + (leafn_p(m) + leafn_storage_p(m) + deadstemn_p(m))* pftfrac(m)
               !            totvegn = totvegn + (leafn_p(m) + leafn_storage_p(m) + deadstemn_p(m))* pftfrac(m)
            ENDDO
            IF(DEF_USE_OZONESTRESS)THEN
               o3uptakesun_p            (ps:pe) = 0._r8
               o3uptakesha_p            (ps:pe) = 0._r8
            ENDIF
            leafc_xfer_p             (ps:pe) = 0.0
            frootc_xfer_p            (ps:pe) = 0.0
            livestemc_storage_p      (ps:pe) = 0.0
            livestemc_xfer_p         (ps:pe) = 0.0
            deadstemc_storage_p      (ps:pe) = 0.0
            deadstemc_xfer_p         (ps:pe) = 0.0
            livecrootc_storage_p     (ps:pe) = 0.0
            livecrootc_xfer_p        (ps:pe) = 0.0
            deadcrootc_storage_p     (ps:pe) = 0.0
            deadcrootc_xfer_p        (ps:pe) = 0.0
            grainc_p                 (ps:pe) = 0.0
            grainc_storage_p         (ps:pe) = 0.0
            grainc_xfer_p            (ps:pe) = 0.0
            cropseedc_deficit_p      (ps:pe) = 0.0
            xsmrpool_p               (ps:pe) = 0.0
            gresp_storage_p          (ps:pe) = 0.0
            gresp_xfer_p             (ps:pe) = 0.0
            cpool_p                  (ps:pe) = 0.0
            ctrunc_p                 (ps:pe) = 0.0
            cropprod1c_p             (ps:pe) = 0.0

            leafn_xfer_p             (ps:pe) = 0.0
            frootn_storage_p         (ps:pe) = 0.0
            frootn_xfer_p            (ps:pe) = 0.0
            livestemn_storage_p      (ps:pe) = 0.0
            livestemn_xfer_p         (ps:pe) = 0.0
            deadstemn_storage_p      (ps:pe) = 0.0
            deadstemn_xfer_p         (ps:pe) = 0.0
            livecrootn_storage_p     (ps:pe) = 0.0
            livecrootn_xfer_p        (ps:pe) = 0.0
            deadcrootn_storage_p     (ps:pe) = 0.0
            deadcrootn_xfer_p        (ps:pe) = 0.0
            grainn_p                 (ps:pe) = 0.0
            grainn_storage_p         (ps:pe) = 0.0
            grainn_xfer_p            (ps:pe) = 0.0
            npool_p                  (ps:pe) = 0.0
            ntrunc_p                 (ps:pe) = 0.0
            cropseedn_deficit_p      (ps:pe) = 0.0
            retransn_p               (ps:pe) = 0.0

            harvdate_p               (ps:pe) = 99999999

            tempsum_potential_gpp_p  (ps:pe) = 0.0
            tempmax_retransn_p       (ps:pe) = 0.0
            tempavg_tref_p           (ps:pe) = 0.0
            tempsum_npp_p            (ps:pe) = 0.0
            tempsum_litfall_p        (ps:pe) = 0.0
            annsum_potential_gpp_p   (ps:pe) = 0.0
            annmax_retransn_p        (ps:pe) = 0.0
            annavg_tref_p            (ps:pe) = 280.0
            annsum_npp_p             (ps:pe) = 0.0
            annsum_litfall_p         (ps:pe) = 0.0

            bglfr_p                  (ps:pe) = 0.0
            bgtr_p                   (ps:pe) = 0.0
            lgsf_p                   (ps:pe) = 0.0
            gdd0_p                   (ps:pe) = 0.0
            gdd8_p                   (ps:pe) = 0.0
            gdd10_p                  (ps:pe) = 0.0
            gdd020_p                 (ps:pe) = 0.0
            gdd820_p                 (ps:pe) = 0.0
            gdd1020_p                (ps:pe) = 0.0
            nyrs_crop_active_p       (ps:pe) = 0

            offset_flag_p            (ps:pe) = 0.0
            offset_counter_p         (ps:pe) = 0.0
            onset_flag_p             (ps:pe) = 0.0
            onset_counter_p          (ps:pe) = 0.0
            onset_gddflag_p          (ps:pe) = 0.0
            onset_gdd_p              (ps:pe) = 0.0
            onset_fdd_p              (ps:pe) = 0.0
            onset_swi_p              (ps:pe) = 0.0
            offset_fdd_p             (ps:pe) = 0.0
            offset_swi_p             (ps:pe) = 0.0
            dormant_flag_p           (ps:pe) = 1.0
            prev_leafc_to_litter_p   (ps:pe) = 0.0
            prev_frootc_to_litter_p  (ps:pe) = 0.0
            days_active_p            (ps:pe) = 0.0

            burndate_p               (ps:pe) = 10000
            grain_flag_p             (ps:pe) = 0.0

#ifdef CROP
            ! crop variables
            croplive_p               (ps:pe) = .false.
            hui_p                    (ps:pe) =  spval
            gddplant_p               (ps:pe) =  spval
            peaklai_p                (ps:pe) =  0
            aroot_p                  (ps:pe) =  spval
            astem_p                  (ps:pe) =  spval
            arepr_p                  (ps:pe) =  spval
            aleaf_p                  (ps:pe) =  spval
            astemi_p                 (ps:pe) =  spval
            aleafi_p                 (ps:pe) =  spval
            gddmaturity_p            (ps:pe) =  spval

            cropplant_p              (ps:pe) = .false.
            idop_p                   (ps:pe) = 99999999
            cumvd_p                  (ps:pe) = spval
            vf_p                     (ps:pe) = 0._r8
            cphase_p                 (ps:pe) = 4._r8
            fert_counter_p           (ps:pe) = 0._r8
            tref_min_p               (ps:pe) = 273.15_r8
            tref_max_p               (ps:pe) = 273.15_r8
            tref_min_inst_p          (ps:pe) = spval
            tref_max_inst_p          (ps:pe) = spval
            latbaset_p               (ps:pe) = spval
            fert_p                   (ps:pe) = 0._r8
#endif

            IF(DEF_USE_LAIFEEDBACK)THEN
               tlai_p                (ps:pe) = slatop(pftclass(ps:pe)) * leafc_p(ps:pe)
               tlai_p                (ps:pe) = max(0._r8, tlai_p(ps:pe))
               lai_p                 (ps:pe) = tlai_p(ps:pe)
               lai                           = sum(lai_p(ps:pe) * pftfrac(ps:pe))
            ENDIF

#ifdef BGC
            CALL CNDriverSummarizeStates(ipatch,ps,pe,nl_soil,dz_soi,ndecomp_pools,.true.)
#endif

            ! SASU varaibles
            leafc0_p                 (ps:pe) = 0.0
            leafc0_storage_p         (ps:pe) = 0.0
            leafc0_xfer_p            (ps:pe) = 0.0
            frootc0_p                (ps:pe) = 0.0
            frootc0_storage_p        (ps:pe) = 0.0
            frootc0_xfer_p           (ps:pe) = 0.0
            livestemc0_p             (ps:pe) = 0.0
            livestemc0_storage_p     (ps:pe) = 0.0
            livestemc0_xfer_p        (ps:pe) = 0.0
            deadstemc0_p             (ps:pe) = 0.0
            deadstemc0_storage_p     (ps:pe) = 0.0
            deadstemc0_xfer_p        (ps:pe) = 0.0
            livecrootc0_p            (ps:pe) = 0.0
            livecrootc0_storage_p    (ps:pe) = 0.0
            livecrootc0_xfer_p       (ps:pe) = 0.0
            deadcrootc0_p            (ps:pe) = 0.0
            deadcrootc0_storage_p    (ps:pe) = 0.0
            deadcrootc0_xfer_p       (ps:pe) = 0.0
            grainc0_p                (ps:pe) = 0.0
            grainc0_storage_p        (ps:pe) = 0.0
            grainc0_xfer_p           (ps:pe) = 0.0

            leafn0_p                 (ps:pe) = 0.0
            leafn0_storage_p         (ps:pe) = 0.0
            leafn0_xfer_p            (ps:pe) = 0.0
            frootn0_p                (ps:pe) = 0.0
            frootn0_storage_p        (ps:pe) = 0.0
            frootn0_xfer_p           (ps:pe) = 0.0
            livestemn0_p             (ps:pe) = 0.0
            livestemn0_storage_p     (ps:pe) = 0.0
            livestemn0_xfer_p        (ps:pe) = 0.0
            deadstemn0_p             (ps:pe) = 0.0
            deadstemn0_storage_p     (ps:pe) = 0.0
            deadstemn0_xfer_p        (ps:pe) = 0.0
            livecrootn0_p            (ps:pe) = 0.0
            livecrootn0_storage_p    (ps:pe) = 0.0
            livecrootn0_xfer_p       (ps:pe) = 0.0
            deadcrootn0_p            (ps:pe) = 0.0
            deadcrootn0_storage_p    (ps:pe) = 0.0
            deadcrootn0_xfer_p       (ps:pe) = 0.0
            grainn0_p                (ps:pe) = 0.0
            grainn0_storage_p        (ps:pe) = 0.0
            grainn0_xfer_p           (ps:pe) = 0.0
            retransn0_p              (ps:pe) = 0.0

            I_leafc_p_acc            (ps:pe) = 0._r8
            I_leafc_st_p_acc         (ps:pe) = 0._r8
            I_frootc_p_acc           (ps:pe) = 0._r8
            I_frootc_st_p_acc        (ps:pe) = 0._r8
            I_livestemc_p_acc        (ps:pe) = 0._r8
            I_livestemc_st_p_acc     (ps:pe) = 0._r8
            I_deadstemc_p_acc        (ps:pe) = 0._r8
            I_deadstemc_st_p_acc     (ps:pe) = 0._r8
            I_livecrootc_p_acc       (ps:pe) = 0._r8
            I_livecrootc_st_p_acc    (ps:pe) = 0._r8
            I_deadcrootc_p_acc       (ps:pe) = 0._r8
            I_deadcrootc_st_p_acc    (ps:pe) = 0._r8
            I_grainc_p_acc           (ps:pe) = 0._r8
            I_grainc_st_p_acc        (ps:pe) = 0._r8
            I_leafn_p_acc            (ps:pe) = 0._r8
            I_leafn_st_p_acc         (ps:pe) = 0._r8
            I_frootn_p_acc           (ps:pe) = 0._r8
            I_frootn_st_p_acc        (ps:pe) = 0._r8
            I_livestemn_p_acc        (ps:pe) = 0._r8
            I_livestemn_st_p_acc     (ps:pe) = 0._r8
            I_deadstemn_p_acc        (ps:pe) = 0._r8
            I_deadstemn_st_p_acc     (ps:pe) = 0._r8
            I_livecrootn_p_acc       (ps:pe) = 0._r8
            I_livecrootn_st_p_acc    (ps:pe) = 0._r8
            I_deadcrootn_p_acc       (ps:pe) = 0._r8
            I_deadcrootn_st_p_acc    (ps:pe) = 0._r8
            I_grainn_p_acc           (ps:pe) = 0._r8
            I_grainn_st_p_acc        (ps:pe) = 0._r8

            AKX_leafc_xf_to_leafc_p_acc                 (ps:pe) = 0._r8
            AKX_frootc_xf_to_frootc_p_acc               (ps:pe) = 0._r8
            AKX_livestemc_xf_to_livestemc_p_acc         (ps:pe) = 0._r8
            AKX_deadstemc_xf_to_deadstemc_p_acc         (ps:pe) = 0._r8
            AKX_livecrootc_xf_to_livecrootc_p_acc       (ps:pe) = 0._r8
            AKX_deadcrootc_xf_to_deadcrootc_p_acc       (ps:pe) = 0._r8
            AKX_grainc_xf_to_grainc_p_acc               (ps:pe) = 0._r8
            AKX_livestemc_to_deadstemc_p_acc            (ps:pe) = 0._r8
            AKX_livecrootc_to_deadcrootc_p_acc          (ps:pe) = 0._r8

            AKX_leafc_st_to_leafc_xf_p_acc              (ps:pe) = 0._r8
            AKX_frootc_st_to_frootc_xf_p_acc            (ps:pe) = 0._r8
            AKX_livestemc_st_to_livestemc_xf_p_acc      (ps:pe) = 0._r8
            AKX_deadstemc_st_to_deadstemc_xf_p_acc      (ps:pe) = 0._r8
            AKX_livecrootc_st_to_livecrootc_xf_p_acc    (ps:pe) = 0._r8
            AKX_deadcrootc_st_to_deadcrootc_xf_p_acc    (ps:pe) = 0._r8
            AKX_grainc_st_to_grainc_xf_p_acc            (ps:pe) = 0._r8

            AKX_leafc_exit_p_acc                        (ps:pe) = 0._r8
            AKX_frootc_exit_p_acc                       (ps:pe) = 0._r8
            AKX_livestemc_exit_p_acc                    (ps:pe) = 0._r8
            AKX_deadstemc_exit_p_acc                    (ps:pe) = 0._r8
            AKX_livecrootc_exit_p_acc                   (ps:pe) = 0._r8
            AKX_deadcrootc_exit_p_acc                   (ps:pe) = 0._r8
            AKX_grainc_exit_p_acc                       (ps:pe) = 0._r8

            AKX_leafc_st_exit_p_acc                     (ps:pe) = 0._r8
            AKX_frootc_st_exit_p_acc                    (ps:pe) = 0._r8
            AKX_livestemc_st_exit_p_acc                 (ps:pe) = 0._r8
            AKX_deadstemc_st_exit_p_acc                 (ps:pe) = 0._r8
            AKX_livecrootc_st_exit_p_acc                (ps:pe) = 0._r8
            AKX_deadcrootc_st_exit_p_acc                (ps:pe) = 0._r8
            AKX_grainc_st_exit_p_acc                    (ps:pe) = 0._r8

            AKX_leafc_xf_exit_p_acc                     (ps:pe) = 0._r8
            AKX_frootc_xf_exit_p_acc                    (ps:pe) = 0._r8
            AKX_livestemc_xf_exit_p_acc                 (ps:pe) = 0._r8
            AKX_deadstemc_xf_exit_p_acc                 (ps:pe) = 0._r8
            AKX_livecrootc_xf_exit_p_acc                (ps:pe) = 0._r8
            AKX_deadcrootc_xf_exit_p_acc                (ps:pe) = 0._r8
            AKX_grainc_xf_exit_p_acc                    (ps:pe) = 0._r8

            AKX_leafn_xf_to_leafn_p_acc                 (ps:pe) = 0._r8
            AKX_frootn_xf_to_frootn_p_acc               (ps:pe) = 0._r8
            AKX_livestemn_xf_to_livestemn_p_acc         (ps:pe) = 0._r8
            AKX_deadstemn_xf_to_deadstemn_p_acc         (ps:pe) = 0._r8
            AKX_livecrootn_xf_to_livecrootn_p_acc       (ps:pe) = 0._r8
            AKX_deadcrootn_xf_to_deadcrootn_p_acc       (ps:pe) = 0._r8
            AKX_grainn_xf_to_grainn_p_acc               (ps:pe) = 0._r8
            AKX_livestemn_to_deadstemn_p_acc            (ps:pe) = 0._r8
            AKX_livecrootn_to_deadcrootn_p_acc          (ps:pe) = 0._r8

            AKX_leafn_st_to_leafn_xf_p_acc              (ps:pe) = 0._r8
            AKX_frootn_st_to_frootn_xf_p_acc            (ps:pe) = 0._r8
            AKX_livestemn_st_to_livestemn_xf_p_acc      (ps:pe) = 0._r8
            AKX_deadstemn_st_to_deadstemn_xf_p_acc      (ps:pe) = 0._r8
            AKX_livecrootn_st_to_livecrootn_xf_p_acc    (ps:pe) = 0._r8
            AKX_deadcrootn_st_to_deadcrootn_xf_p_acc    (ps:pe) = 0._r8
            AKX_grainn_st_to_grainn_xf_p_acc            (ps:pe) = 0._r8

            AKX_leafn_to_retransn_p_acc                 (ps:pe) = 0._r8
            AKX_frootn_to_retransn_p_acc                (ps:pe) = 0._r8
            AKX_livestemn_to_retransn_p_acc             (ps:pe) = 0._r8
            AKX_livecrootn_to_retransn_p_acc            (ps:pe) = 0._r8

            AKX_retransn_to_leafn_p_acc                 (ps:pe) = 0._r8
            AKX_retransn_to_frootn_p_acc                (ps:pe) = 0._r8
            AKX_retransn_to_livestemn_p_acc             (ps:pe) = 0._r8
            AKX_retransn_to_deadstemn_p_acc             (ps:pe) = 0._r8
            AKX_retransn_to_livecrootn_p_acc            (ps:pe) = 0._r8
            AKX_retransn_to_deadcrootn_p_acc            (ps:pe) = 0._r8
            AKX_retransn_to_grainn_p_acc                (ps:pe) = 0._r8

            AKX_retransn_to_leafn_st_p_acc              (ps:pe) = 0._r8
            AKX_retransn_to_frootn_st_p_acc             (ps:pe) = 0._r8
            AKX_retransn_to_livestemn_st_p_acc          (ps:pe) = 0._r8
            AKX_retransn_to_deadstemn_st_p_acc          (ps:pe) = 0._r8
            AKX_retransn_to_livecrootn_st_p_acc         (ps:pe) = 0._r8
            AKX_retransn_to_deadcrootn_st_p_acc         (ps:pe) = 0._r8
            AKX_retransn_to_grainn_st_p_acc             (ps:pe) = 0._r8

            AKX_leafn_exit_p_acc                        (ps:pe) = 0._r8
            AKX_frootn_exit_p_acc                       (ps:pe) = 0._r8
            AKX_livestemn_exit_p_acc                    (ps:pe) = 0._r8
            AKX_deadstemn_exit_p_acc                    (ps:pe) = 0._r8
            AKX_livecrootn_exit_p_acc                   (ps:pe) = 0._r8
            AKX_deadcrootn_exit_p_acc                   (ps:pe) = 0._r8
            AKX_grainn_exit_p_acc                       (ps:pe) = 0._r8
            AKX_retransn_exit_p_acc                     (ps:pe) = 0._r8

            AKX_leafn_st_exit_p_acc                     (ps:pe) = 0._r8
            AKX_frootn_st_exit_p_acc                    (ps:pe) = 0._r8
            AKX_livestemn_st_exit_p_acc                 (ps:pe) = 0._r8
            AKX_deadstemn_st_exit_p_acc                 (ps:pe) = 0._r8
            AKX_livecrootn_st_exit_p_acc                (ps:pe) = 0._r8
            AKX_deadcrootn_st_exit_p_acc                (ps:pe) = 0._r8
            AKX_grainn_st_exit_p_acc                    (ps:pe) = 0._r8

            AKX_leafn_xf_exit_p_acc                     (ps:pe) = 0._r8
            AKX_frootn_xf_exit_p_acc                    (ps:pe) = 0._r8
            AKX_livestemn_xf_exit_p_acc                 (ps:pe) = 0._r8
            AKX_deadstemn_xf_exit_p_acc                 (ps:pe) = 0._r8
            AKX_livecrootn_xf_exit_p_acc                (ps:pe) = 0._r8
            AKX_deadcrootn_xf_exit_p_acc                (ps:pe) = 0._r8
            AKX_grainn_xf_exit_p_acc                    (ps:pe) = 0._r8

         ENDIF
#endif
#endif

         ! (8) surface albedo
         ! Variables: alb, ssun, ssha, ssno, thermk, extkb, extkd
         wt      = 0.
         pg_snow = 0.
         snofrz (:) = 0.
         ssw = min(1.,1.e-3*wliq_soisno(1)/dz_soisno(1))
         CALL albland (ipatch,patchtype,1800._r8,soil_s_v_alb,soil_d_v_alb,soil_s_n_alb,soil_d_n_alb,&
            chil,rho,tau,fveg,green,lai,sai,max(0.001,coszen),&
            wt,fsno,scv,scv,sag,ssw,pg_snow,273.15_r8,t_grnd,t_soisno(:1),dz_soisno(:1),&
            snl,wliq_soisno,wice_soisno,snw_rds,snofrz,&
            mss_bcpho,mss_bcphi,mss_ocpho,mss_ocphi,&
            mss_dst1,mss_dst2,mss_dst3,mss_dst4,&
            alb,ssun,ssha,ssoi,ssno,ssno_lyr,thermk,extkb,extkd)
      ELSE                 !ocean grid
         t_soisno(:) = 300.
         wice_soisno(:) = 0.
         wliq_soisno(:) = 1000.
         z_soisno (maxsnl+1:0) = 0.
         dz_soisno(maxsnl+1:0) = 0.
         sigf   = 0.
         fsno   = 0.
         ldew_rain  = 0.
         ldew_snow  = 0.
         ldew   = 0.
         scv    = 0.
         sag    = 0.
         snowdp = 0.
         tleaf  = 300.
         IF(DEF_USE_PLANTHYDRAULICS)THEN
            vegwp(1:nvegwcs) = -2.5e4
            gs0sun = 1.0e4
            gs0sha = 1.0e4
         ENDIF
         t_grnd = 300.

         oro = 0
         CALL albocean (oro,scv,coszen,alb)
         ssun(:,:) = 0.0
         ssha(:,:) = 0.0
         ssoi(:,:) = 0.0
         ssno(:,:) = 0.0
         ssno_lyr(:,:,:) = 0.0
         thermk = 0.0
         extkb = 0.0
         extkd = 0.0
      ENDIF

      ! Additional variables required by reginal model (WRF & RSM)
      ! totally arbitrarily assigned here
      trad  = t_grnd
      tref  = t_grnd
      qref  = 0.3
      !   rst   = 1.e36
      emis  = 1.0
      zol   = -1.0
      rib   = -0.1
      ustar = 0.25
      qstar = 0.001
      tstar = -1.5
      fm    = alog(30.)
      fh    = alog(30.)
      fq    = alog(30.)

   END SUBROUTINE IniTimeVar
   !-----------------------------------------------------------------------
   ! EOP


   SUBROUTINE snow_ini(patchtype,maxsnl,snowdp,snl,z_soisno,dz_soisno)

   ! Snow spatial discretization initially

   USE MOD_Precision
   IMPLICIT NONE

   integer,  intent(in) :: maxsnl    !maximum of snow layers
   integer,  intent(in) :: patchtype !index for land cover type [-]
   real(r8), intent(in) :: snowdp    !snow depth [m]
   real(r8), intent(out) :: z_soisno (maxsnl+1:0) !node depth [m]
   real(r8), intent(out) :: dz_soisno(maxsnl+1:0) !layer thickness [m]
   integer,  intent(out) :: snl                   !number of snow layer
   real(r8) zi
   integer i

      dz_soisno(:0) = 0.
      z_soisno(:0) = 0.
      snl = 0
      IF(patchtype.le.3)THEN !non water bodies

         IF(snowdp.lt.0.01)THEN
            snl = 0
         ELSE
            IF(snowdp>=0.01 .and. snowdp<=0.03)THEN
               snl = -1
               dz_soisno(0)  = snowdp
            ELSE IF(snowdp>0.03 .and. snowdp<=0.04)THEN
               snl = -2
               dz_soisno(-1) = snowdp/2.
               dz_soisno( 0) = dz_soisno(-1)
            ELSE IF(snowdp>0.04 .and. snowdp<=0.07)THEN
               snl = -2
               dz_soisno(-1) = 0.02
               dz_soisno( 0) = snowdp - dz_soisno(-1)
            ELSE IF(snowdp>0.07 .and. snowdp<=0.12)THEN
               snl = -3
               dz_soisno(-2) = 0.02
               dz_soisno(-1) = (snowdp - 0.02)/2.
               dz_soisno( 0) = dz_soisno(-1)
            ELSE IF(snowdp>0.12 .and. snowdp<=0.18)THEN
               snl = -3
               dz_soisno(-2) = 0.02
               dz_soisno(-1) = 0.05
               dz_soisno( 0) = snowdp - dz_soisno(-2) - dz_soisno(-1)
            ELSE IF(snowdp>0.18 .and. snowdp<=0.29)THEN
               snl = -4
               dz_soisno(-3) = 0.02
               dz_soisno(-2) = 0.05
               dz_soisno(-1) = (snowdp - dz_soisno(-3) - dz_soisno(-2))/2.
               dz_soisno( 0) = dz_soisno(-1)
            ELSE IF(snowdp>0.29 .and. snowdp<=0.41)THEN
               snl = -4
               dz_soisno(-3) = 0.02
               dz_soisno(-2) = 0.05
               dz_soisno(-1) = 0.11
               dz_soisno( 0) = snowdp - dz_soisno(-3) - dz_soisno(-2) - dz_soisno(-1)
            ELSE IF(snowdp>0.41 .and. snowdp<=0.64)THEN
               snl = -5
               dz_soisno(-4) = 0.02
               dz_soisno(-3) = 0.05
               dz_soisno(-2) = 0.11
               dz_soisno(-1) = (snowdp - dz_soisno(-4) - dz_soisno(-3) - dz_soisno(-2))/2.
               dz_soisno( 0) = dz_soisno(-1)
            ELSE IF(snowdp>0.64)THEN
               snl = -5
               dz_soisno(-4) = 0.02
               dz_soisno(-3) = 0.05
               dz_soisno(-2) = 0.11
               dz_soisno(-1) = 0.23
               dz_soisno( 0) = snowdp - dz_soisno(-4) - dz_soisno(-3) - dz_soisno(-2) - dz_soisno(-1)
            ENDIF

            zi = 0.
            DO i = 0, snl+1, -1
               z_soisno(i) = zi - dz_soisno(i)/2.
               zi = -zi-dz_soisno(i)
            ENDDO
         ENDIF

      ENDIF

   END SUBROUTINE snow_ini

END MODULE MOD_IniTimeVariable
!-----------------------------------------------------------------------
! EOP
