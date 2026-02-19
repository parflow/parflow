#include <define.h>
#ifdef CROP
MODULE MOD_Irrigation

!  DESCRIPTION:
!      This MODULE has all irrigation related subroutines for irrigated crop at either IGBP/USGS or PFT Land type classification and even in the C and N cycle.
   USE MOD_Precision
   USE MOD_TimeManager
   USE MOD_Namelist, only: DEF_simulation_time
   ! ,DEF_IRRIGATION_METHOD
   USE MOD_Const_Physical, only: tfrz
   USE MOD_Const_PFT, only: irrig_crop
   USE MOD_Vars_Global, only: irrig_start_time, irrig_max_depth, irrig_threshold_fraction, irrig_min_cphase, irrig_max_cphase, irrig_time_per_day
   USE MOD_Qsadv, only: qsadv
   USE MOD_Vars_TimeInvariants, only: &
#ifdef vanGenuchten_Mualem_SOIL_MODEL
       theta_r, alpha_vgm, n_vgm, L_vgm, fc_vgm, sc_vgm,&
#endif
       porsl, psi0, bsw
   USE MOD_Vars_TimeVariables, only : tref, t_soisno, wliq_soisno, irrig_rate, deficit_irrig, sum_irrig, sum_irrig_count, n_irrig_steps_left, &
       tairday, usday, vsday, pairday, rnetday, fgrndday, potential_evapotranspiration
   USE MOD_Vars_PFTimeInvariants, only: pftclass
   USE MOD_Vars_PFTimeVariables, only: irrig_method_p
   USE MOD_BGC_Vars_PFTimeVariables, only: cphase_p
   USE MOD_Vars_1DForcing, only: forc_t, forc_frl, forc_psrf, forc_us, forc_vs
   USE MOD_Vars_1DFluxes, only: sabg, sabvsun, sabvsha, olrg, fgrnd
   USE MOD_Hydro_SoilFunction, only: soil_vliq_from_psi

   IMPLICIT NONE

   PUBLIC :: CalIrrigationNeeded
   PUBLIC :: CalIrrigationApplicationFluxes

   !   local variable
   integer :: irrig_method_drip = 1
   integer :: irrig_method_sprinkler = 2
   integer :: irrig_method_flood = 3
   integer :: irrig_method_paddy = 4

CONTAINS

   SUBROUTINE CalIrrigationNeeded(i,ps,pe,idate,nl_soil,nbedrock,z_soi,dz_soi,deltim,dlon,npcropmin)

   !   DESCRIPTION:
   !   This SUBROUTINE is used to calculate how much irrigation needed in each irrigated crop patch
   integer , intent(in) :: i
   integer , intent(in) :: ps, pe
   integer , intent(in) :: idate(3)
   integer , intent(in) :: nl_soil
   integer , intent(in) :: nbedrock
   real(r8), intent(in) :: z_soi(1:nl_soil)
   real(r8), intent(in) :: dz_soi(1:nl_soil)
   real(r8), intent(in) :: deltim
   real(r8), intent(in) :: dlon
   integer , intent(in) :: npcropmin

   ! local
   integer :: m
   integer :: irrig_nsteps_per_day
   logical :: check_for_irrig

      ! !   calculate last day potential evapotranspiration
      ! CALL CalPotentialEvapotranspiration(i,idate,dlon,deltim)

      !   calculate whether irrigation needed
      CALL PointNeedsCheckForIrrig(i,ps,pe,idate,deltim,dlon,npcropmin,check_for_irrig)

      !   calculate irrigation needed
      IF (check_for_irrig) THEN
         CALL CalIrrigationPotentialNeeded(i,ps,pe,nl_soil,nbedrock,z_soi,dz_soi)
         ! CALL CalIrrigationLimitedNeeded(i,ps,pe)
      ENDIF

      !   calculate irrigation rate kg/m2->mm/s
      IF ((check_for_irrig) .and. (deficit_irrig(i) > 0)) THEN
         irrig_nsteps_per_day = nint(irrig_time_per_day/deltim)
         irrig_rate(i) = deficit_irrig(i)/deltim/irrig_nsteps_per_day
         n_irrig_steps_left(i) = irrig_nsteps_per_day
         sum_irrig(i) = sum_irrig(i) + deficit_irrig(i)
         sum_irrig_count(i) = sum_irrig_count(i) + 1._r8
      ENDIF

      ! !   zero irrigation at the END of growing season
      ! DO m = ps, pe
      !     IF (cphase_p(m) >= 4._r8) THEN
      !         sum_irrig(i) = 0._r8
      !         sum_irrig_count(i) = 0._r8
      !     ENDIF
      ! ENDDO
   END SUBROUTINE CalIrrigationNeeded


   SUBROUTINE CalIrrigationPotentialNeeded(i,ps,pe,nl_soil,nbedrock,z_soi,dz_soi)

   !   DESCRIPTION:
   !   This SUBROUTINE is used to calculate how much irrigation needed in each irrigated crop patch without water supply restriction
   integer , intent(in) :: i
   integer , intent(in) :: ps, pe
   integer , intent(in) :: nbedrock
   integer , intent(in) :: nl_soil
   real(r8), intent(in) :: z_soi(1:nl_soil)
   real(r8), intent(in) :: dz_soi(1:nl_soil)

   !   local variables
   integer  :: j
   integer  :: m
   logical  :: reached_max_depth
   real(r8) :: h2osoi_liq_tot
   real(r8) :: h2osoi_liq_target_tot
   real(r8) :: h2osoi_liq_wilting_point_tot
   real(r8) :: h2osoi_liq_saturation_capacity_tot
   real(r8) :: h2osoi_liq_wilting_point(1:nl_soil)
   real(r8) :: h2osoi_liq_field_capacity(1:nl_soil)
   real(r8) :: h2osoi_liq_saturation_capacity(1:nl_soil)
   real(r8) :: h2osoi_liq_at_threshold

   real(r8) :: smpswc = -1.5e5
   real(r8) :: smpsfc = -3.3e3

      !   initialize local variables
      reached_max_depth = .false.
      h2osoi_liq_tot = 0._r8
      h2osoi_liq_target_tot = 0._r8
      h2osoi_liq_wilting_point_tot = 0._r8
      h2osoi_liq_saturation_capacity_tot = 0._r8

      ! !   single site initialization
      ! DO m = ps, pe
      !     irrig_method_p(m) = DEF_IRRIGATION_METHOD
      ! ENDDO

!   calculate wilting point and field capacity
      DO j = 1, nl_soil
         IF (t_soisno(j,i) > tfrz .and. porsl(j,i) >= 1.e-6) THEN
#ifdef Campbell_SOIL_MODEL
            h2osoi_liq_wilting_point(j) = 1000.*dz_soi(j)*porsl(j,i)*((smpswc/psi0(j,i))**(-1/bsw(j,i)))
            h2osoi_liq_field_capacity(j) = 1000.*dz_soi(j)*porsl(j,i)*((smpsfc/psi0(j,i))**(-1/bsw(j,i)))
            h2osoi_liq_saturation_capacity(j) = 1000.*dz_soi(j)*porsl(j,i)
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
            h2osoi_liq_wilting_point(j) = soil_vliq_from_psi(smpswc, porsl(j,i), theta_r(j,i), psi0(j,i), 5, &
              (/alpha_vgm(j,i), n_vgm(j,i), L_vgm(j,i), sc_vgm(j,i), fc_vgm(j,i)/))
            h2osoi_liq_wilting_point(j) = 1000.*dz_soi(j)*h2osoi_liq_wilting_point(j)
            h2osoi_liq_field_capacity(j) = soil_vliq_from_psi(smpsfc, porsl(j,i), theta_r(j,i), psi0(j,i), 5, &
              (/alpha_vgm(j,i), n_vgm(j,i), L_vgm(j,i), sc_vgm(j,i), fc_vgm(j,i)/))
            h2osoi_liq_field_capacity(j) = 1000.*dz_soi(j)*h2osoi_liq_field_capacity(j)
            h2osoi_liq_saturation_capacity(j) = 1000.*dz_soi(j)*porsl(j,i)
#endif
         ENDIF
      ENDDO

      !   calculate total irrigation needed in all soil layers
      DO m = ps, pe
         DO j = 1, nl_soil
            IF (.not. reached_max_depth) THEN
               IF (z_soi(j) > irrig_max_depth) THEN
                  reached_max_depth = .true.
               ELSEIF (j > nbedrock) THEN
                  reached_max_depth = .true.
               ELSEIF (t_soisno(j,i) <= tfrz) THEN
                  reached_max_depth = .true.
               ELSE
                  h2osoi_liq_tot = h2osoi_liq_tot + wliq_soisno(j,i)
                  h2osoi_liq_wilting_point_tot = h2osoi_liq_wilting_point_tot + h2osoi_liq_wilting_point(j)
                  IF (irrig_method_p(m) == irrig_method_drip .or. irrig_method_p(m) == irrig_method_sprinkler) THEN
                      h2osoi_liq_target_tot = h2osoi_liq_target_tot + h2osoi_liq_field_capacity(j)
                  !   irrigation threshold at field capacity, but irrigation amount at saturation capacity
                  ELSEIF (irrig_method_p(m) == irrig_method_flood) THEN
                      h2osoi_liq_target_tot = h2osoi_liq_target_tot + h2osoi_liq_field_capacity(j)
                      h2osoi_liq_saturation_capacity_tot = h2osoi_liq_saturation_capacity_tot + h2osoi_liq_saturation_capacity(j)
                  ELSEIF (irrig_method_p(m) == irrig_method_paddy) THEN
                      h2osoi_liq_target_tot = h2osoi_liq_target_tot + h2osoi_liq_saturation_capacity(j)
                  ELSE
                      h2osoi_liq_target_tot = h2osoi_liq_target_tot + h2osoi_liq_field_capacity(j)
                  ENDIF
               ENDIF
            ENDIF
         ENDDO
      ENDDO

      !   calculate irrigation threshold
      deficit_irrig(i) = 0._r8
      h2osoi_liq_at_threshold = h2osoi_liq_wilting_point_tot + irrig_threshold_fraction * (h2osoi_liq_target_tot - h2osoi_liq_wilting_point_tot)

      !   calculate total irrigation
      DO m = ps, pe
         IF (h2osoi_liq_tot < h2osoi_liq_at_threshold) THEN
            IF (irrig_method_p(m) == irrig_method_sprinkler) THEN
                deficit_irrig(i) = h2osoi_liq_target_tot - h2osoi_liq_tot
                ! deficit_irrig(i) = h2osoi_liq_target_tot - h2osoi_liq_tot + potential_evapotranspiration(i)
            ELSEIF (irrig_method_p(m) == irrig_method_flood) THEN
                deficit_irrig(i) = h2osoi_liq_saturation_capacity_tot - h2osoi_liq_tot
            ELSE
                deficit_irrig(i) = h2osoi_liq_at_threshold - h2osoi_liq_tot
            ENDIF
         ELSE
            deficit_irrig(i) = 0
         ENDIF
      ENDDO

   END SUBROUTINE CalIrrigationPotentialNeeded

   SUBROUTINE CalIrrigationApplicationFluxes(i,ps,pe,deltim,qflx_irrig_drip,qflx_irrig_sprinkler,qflx_irrig_flood,qflx_irrig_paddy,irrig_flag)
      !   DESCRIPTION:
      !   This SUBROUTINE is used to calculate irrigation application fluxes for each irrigated crop patch
      integer , intent(in) :: i
      integer , intent(in) :: ps, pe
      real(r8), intent(in) :: deltim
      integer , intent(in) :: irrig_flag  ! 1 IF sprinker, 2 IF others
      real(r8), intent(out):: qflx_irrig_drip,qflx_irrig_sprinkler,qflx_irrig_flood,qflx_irrig_paddy

      integer :: m

      qflx_irrig_drip = 0._r8
      qflx_irrig_sprinkler = 0._r8
      qflx_irrig_flood = 0._r8
      qflx_irrig_paddy = 0._r8

      ! !   single site initialization
      ! DO m = ps, pe
      !     irrig_method_p(m) = DEF_IRRIGATION_METHOD
      ! ENDDO

      !   add irrigation fluxes to precipitation or land surface
      DO m = ps, pe
         IF (n_irrig_steps_left(i) > 0) THEN
            IF ((irrig_flag == 1) .and. (irrig_method_p(m) == irrig_method_sprinkler)) THEN
               qflx_irrig_sprinkler = irrig_rate(i)
               n_irrig_steps_left(i) = n_irrig_steps_left(i) -1
               deficit_irrig(i) = deficit_irrig(i) - irrig_rate(i)*deltim
            ELSEIF (irrig_flag == 2) THEN
               IF (irrig_method_p(m) == irrig_method_drip) THEN
                   qflx_irrig_drip = irrig_rate(i)
               ELSEIF (irrig_method_p(m) == irrig_method_flood) THEN
                   qflx_irrig_flood = irrig_rate(i)
               ELSEIF (irrig_method_p(m) == irrig_method_paddy) THEN
                   qflx_irrig_paddy = irrig_rate(i)
               ELSEIF ((irrig_method_p(m) /= irrig_method_drip) .and. (irrig_method_p(m) /= irrig_method_sprinkler) &
                   .and. (irrig_method_p(m) /= irrig_method_flood) .and. (irrig_method_p(m) /= irrig_method_paddy)) THEN
                   qflx_irrig_drip = irrig_rate(i)
               ENDIF
               n_irrig_steps_left(i) = n_irrig_steps_left(i) -1
               deficit_irrig(i) = deficit_irrig(i) - irrig_rate(i)*deltim
            ENDIF
            IF (deficit_irrig(i) < 0._r8) THEN
               deficit_irrig(i) = 0._r8
            ENDIF
         ELSE
             irrig_rate(i) = 0._r8
         ENDIF
      ENDDO
   END SUBROUTINE CalIrrigationApplicationFluxes

   SUBROUTINE PointNeedsCheckForIrrig(i,ps,pe,idate,deltim,dlon,npcropmin,check_for_irrig)
   !   DESCRIPTION:
   !   This SUBROUTINE is used to calculate whether irrigation needed in each patch
   integer , intent(in) :: i
   integer , intent(in) :: ps, pe
   integer , intent(in) :: idate(3)
   real(r8), intent(in) :: deltim
   real(r8), intent(in) :: dlon
   integer , intent(in) :: npcropmin
   logical , intent(out):: check_for_irrig

   !   local variable
   integer :: m, ivt
   real(r8):: ldate(3)
   real(r8):: seconds_since_irrig_start_time

      DO m = ps, pe
         ivt = pftclass(m)
         IF ((ivt >= npcropmin) .and. (irrig_crop(ivt)) .and. &
            (cphase_p(m) >= irrig_min_cphase) .and. (cphase_p(m)<irrig_max_cphase)) THEN
            IF (DEF_simulation_time%greenwich) THEN
                CALL gmt2local(idate, dlon, ldate)
                seconds_since_irrig_start_time = ldate(3) - irrig_start_time + deltim
            ELSE
                seconds_since_irrig_start_time = idate(3) - irrig_start_time + deltim
            ENDIF
            IF ((seconds_since_irrig_start_time >= 0._r8) .and. (seconds_since_irrig_start_time < deltim)) THEN
                check_for_irrig = .true.
            ELSE
                check_for_irrig = .false.
            ENDIF
         ELSE
            check_for_irrig = .false.
         ENDIF
      ENDDO

   END SUBROUTINE PointNeedsCheckForIrrig

   ! SUBROUTINE CalPotentialEvapotranspiration(i,idate,dlon,deltim)
   !     !   DESCRIPTION:
   !     !   This SUBROUTINE is used to calculate daily potential evapotranspiration
   !     integer , intent(in) :: i
   !     integer , intent(in) :: idate(3)
   !     real(r8), intent(in) :: dlon
   !     real(r8), intent(in) :: deltim

   !     !   local variable
   !     real(r8):: ldate(3)
   !     real(r8):: seconds_since_irrig_start_time
   !     real(r8) :: es,esdT,qs,qsdT     ! saturation vapour pressure
   !     real(r8) :: evsat               ! vapour pressure
   !     real(r8) :: ur                  ! wind speed
   !     real(r8) :: delta               ! slope of saturation vapour pressure curve
   !     real(r8) :: gamma               ! Psychrometric constant

   !     IF (DEF_simulation_time%greenwich) THEN
   !         CALL gmt2local(idate, dlon, ldate)
   !         seconds_since_irrig_start_time = ldate(3) - irrig_start_time + deltim
   !     ELSE
   !         seconds_since_irrig_start_time = idate(3) - irrig_start_time + deltim
   !     ENDIF

   !     IF (((seconds_since_irrig_start_time-deltim) >= 0) .and. ((seconds_since_irrig_start_time-deltim) < deltim)) THEN
   !         tairday(i) = (forc_t(i)-tfrz)*deltim/86400
   !         usday(i) = forc_us(i)*deltim/86400
   !         vsday(i) = forc_vs(i)*deltim/86400
   !         pairday(i) = forc_psrf(i)*deltim/86400/1000
   !         rnetday(i) = (sabg(i)+sabvsun(i)+sabvsha(i)-olrg(i)+forc_frl(i))*deltim/1000000
   !         fgrndday(i) = fgrnd(i)*deltim/1000000
   !     ELSE
   !         tairday(i) = tairday(i) + (forc_t(i)-tfrz)*deltim/86400
   !         usday(i) = usday(i) + forc_us(i)*deltim/86400
   !         vsday(i) = vsday(i) + forc_vs(i)*deltim/86400
   !         pairday(i) = pairday(i) + forc_psrf(i)*deltim/86400/1000
   !         rnetday(i) = rnetday(i) + (sabg(i)+sabvsun(i)+sabvsha(i)-olrg(i)+forc_frl(i))*deltim/1000000
   !         fgrndday(i) = fgrndday(i) + fgrnd(i)*deltim/1000000
   !     ENDIF

   !     IF ((seconds_since_irrig_start_time >= 0) .and. (seconds_since_irrig_start_time < deltim)) THEN
   !         CALL qsadv(tairday(i),pairday(i),es,esdT,qs,qsdT)
   !         IF (tairday(i) > 0)THEN
   !             evsat = 0.611*EXP(17.27*tairday(i)/(tairday(i)+237.3))
   !         ELSE
   !             evsat = 0.611*EXP(21.87*tairday(i)/(tairday(i)+265.5))
   !         ENDIF
   !         ur = max(0.1,sqrt(usday(i)*usday(i)+vsday(i)*vsday(i)))
   !         delta = 4098*evsat/((tairday(i)+237.3)*(tairday(i)+237.3))
   !         gamma = 0.665*0.001*pairday(i)
   !         potential_evapotranspiration(i) = (0.408*delta*(rnetday(i)-fgrndday(i))+gamma*(900/(tairday(i)+273))*ur* &
   !             (evsat-es))/(delta+(gamma*(1+0.34*ur)))
   !     ENDIF
   ! END SUBROUTINE CalPotentialEvapotranspiration

END MODULE MOD_Irrigation
#endif
