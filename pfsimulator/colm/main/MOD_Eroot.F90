#include <define.h>

MODULE MOD_Eroot

!-----------------------------------------------------------------------
   USE MOD_Precision
   IMPLICIT NONE
   SAVE

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: eroot


!-----------------------------------------------------------------------

CONTAINS

!-----------------------------------------------------------------------


   SUBROUTINE eroot (nl_soil,trsmx0,porsl, &
#ifdef Campbell_SOIL_MODEL
         bsw, &
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
         theta_r, alpha_vgm, n_vgm, L_vgm, sc_vgm, fc_vgm, &
#endif
         psi0,rootfr, dz_soisno,t_soisno,wliq_soisno,rootr, &
         etrc,rstfac,veg_water_stress_typepf,wilting_pointpf,field_capacitypf, &
         pf_vol_liq,pf_press)

  !=======================================================================
  ! !DESCRIPTION:
  ! effective root fraction and maximum possible transpiration rate
  ! Original author : Yongjiu Dai, 08/30/2002
  ! Revision author : Shupeng Zhang and Xingjie Lu.
  !=======================================================================

   USE MOD_Precision
   USE MOD_Const_Physical, only : tfrz
#ifdef vanGenuchten_Mualem_SOIL_MODEL
   USE MOD_Hydro_SoilFunction, only : soil_psi_from_vliq
#endif
   IMPLICIT NONE

  !-----------------------Argument-----------------------------------------

   integer, intent(in) :: nl_soil            ! upper bound of array
   integer, intent(in) :: veg_water_stress_typepf
   
   real(r8), intent(in) :: wilting_pointpf, field_capacitypf
   real(r8), intent(in) :: pf_vol_liq(1:nl_soil), pf_press(1:nl_soil)
   real(r8), intent(in) :: trsmx0            ! max transpiration for moist soil+100% veg.[mm/s]
   real(r8), intent(in) :: porsl(1:nl_soil)  ! soil porosity [-]
#ifdef Campbell_SOIL_MODEL
   real(r8), intent(in) :: bsw(1:nl_soil)    ! Clapp-Hornberger "B"
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
   real(r8), intent(in) :: theta_r  (1:nl_soil)
   real(r8), intent(in) :: alpha_vgm(1:nl_soil)
   real(r8), intent(in) :: n_vgm    (1:nl_soil)
   real(r8), intent(in) :: L_vgm    (1:nl_soil)
   real(r8), intent(in) :: sc_vgm   (1:nl_soil)
   real(r8), intent(in) :: fc_vgm   (1:nl_soil)
#endif
   real(r8), intent(in) :: psi0(1:nl_soil)   ! saturated soil suction (mm) (NEGATIVE)
   real(r8), intent(in) :: rootfr(1:nl_soil) ! fraction of roots in a layer,
   real(r8), intent(in) :: dz_soisno(1:nl_soil)   ! layer thickness (m)
   real(r8), intent(in) :: t_soisno(1:nl_soil)    ! soil/snow skin temperature (K)
   real(r8), intent(in) :: wliq_soisno(1:nl_soil) ! liquid water (kg/m2)

   real(r8), intent(out) :: rootr(1:nl_soil) ! root resistance of a layer, all layers add to 1
   real(r8), intent(out) :: etrc             ! maximum possible transpiration rate (mm h2o/s)
   real(r8), intent(out) :: rstfac           ! factor of soil water stress for photosynthesis

  !-----------------------Local Variables------------------------------

   real(r8) roota             ! accumulates root resistance factors
   real(r8) rresis(1:nl_soil) ! soil water contribution to root resistance
   real(r8) s_node            ! vol_liq/porosity
   real(r8) smpmax            ! wilting point potential in mm
   real(r8) smp_node          ! matrix potential

   integer i                  ! loop counter

  !-----------------------End Variables list---------------------------

      ! transpiration potential(etrc) and root resistance factors (rstfac)

      roota = 1.e-10         ! must be non-zero to begin
      DO i = 1, nl_soil

         IF(t_soisno(i)>tfrz .and. porsl(i)>=1.e-6)THEN
            smpmax = -1.5e5
            s_node = max(wliq_soisno(i)/(1000.*dz_soisno(i)*porsl(i)),0.001)
            s_node = min(1., s_node)
#ifdef Campbell_SOIL_MODEL
            smp_node = max(smpmax, psi0(i)*s_node**(-bsw(i)))
#endif
#ifdef vanGenuchten_Mualem_SOIL_MODEL
            smp_node = soil_psi_from_vliq ( s_node*(porsl(i)-theta_r(i)) + theta_r(i), &
               porsl(i), theta_r(i), psi0(i), &
               5, (/alpha_vgm(i), n_vgm(i), L_vgm(i), sc_vgm(i), fc_vgm(i)/))
            smp_node = max(smpmax, smp_node)
#endif
            rresis(i) =(1.-smp_node/smpmax)/(1.-psi0(i)/smpmax)

            !@CY:
            select case (veg_water_stress_typepf)
            case (0)     ! none
            rresis(i) = 1.0d0
            case (1)     ! pressure type
            rresis(i) = ((wilting_pointpf*1000.d0 - pf_press(i))/(wilting_pointpf*1000.d0 - field_capacitypf*1000.d0) )
            case (2)     ! SM type
            rresis(i) = (pf_vol_liq(i) - wilting_pointpf*porsl(i)) / &
                (field_capacitypf*porsl(i) - wilting_pointpf*porsl(i))
            end select

            if (rresis(i) < 0.) rresis(i) = 0.
            if (rresis(i) > 1.) rresis(i) = 1.

            rootr(i) = rootfr(i)*rresis(i)
            roota = roota + rootr(i)
        ELSE
            rootr(i) = 0.
        ENDIF

      ENDDO

      ! normalize root resistances to get layer contribution to ET
      rootr(:) = rootr(:)/roota

      ! determine maximum possible transpiration rate
      etrc = trsmx0*roota
      rstfac = roota

   END SUBROUTINE eroot

END MODULE MOD_Eroot
