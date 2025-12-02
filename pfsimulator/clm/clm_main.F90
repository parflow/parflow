!#include <misc.h>

subroutine clm_main (clm,day,gmt,clm_forc_veg)

  ! CLM Model/Science PHILOSOPHY:
  !   The Common Land Model (CLM) is being developed by a
  !   grass-roots collaboration of scientists who have an interest in
  !   making a general land model available for public use. By grass
  !   roots, we mean that the project is not being controlled by any
  !   single organization or scientist, but rather, the scientific
  !   steering is judged by the community. However, the project began
  !   at a sub-group meeting at the 1998 NCAR CSM meeting, and there is
  !   a plan to implement the CLM into the NCAR CSM by early 2000.  The
  !   CLM development philosophy is that only proven and well-tested
  !   physical parameterizations and numerical schemes shall be
  !   used. The current version of the CLM includes superior components
  !   from each of three contributing models: LSM (G. Bonan, NCAR),
  !   BATS (R. Dickinson) and IAP (Y.-J. Dai). The CLM code management
  !   will be similar to open source, in that, use of the model implies
  !   that any scientific gain will be included in future versions of
  !   the model.  Also, the land model has been run for a suite of test
  !   cases including many of the PILPS (Project for the
  !   Intercomparison of Land Parameterization Schemes) case
  !   studies. These include FIFE (Kansas, USA), Cabauw (Netherlands),
  !   Valdai (Russia), HAPEX (France), and the Amazon (ARME and
  !   ABRACOS). These cases have not been rigorously compared with
  !   observations, but will be thoroughly evaluated in the framework
  !   of the Project for the Intercomparison of Land-surface
  !   Parameterization Schemes (PILPS).
  !
  !
  !
  ! CLM Code PHILOSOPHY:
  !   The CLM is defined as a 1-D land surface model, with all forcings,
  !   parameters, dimensioning, output routines, and coupling performed
  !   by an external driver of the user's design.  It is intended that
  !   the user should be able to accomplish most land surface modeling
  !   tasks without modification of the 1-D CLM code, whose top level
  !   interface begins with clm_main.f90 and whose interaction with
  !   the driver occurs through the clm_module.  An example, highly flexible 
  !   driver is included with the distribution of the 1-D CLM code to aid in
  !   its use.  The naming convention of the "core 1-D CLM code" and the 
  !   external driver subroutines makes this philosophy explicit, as follows:
  !     clm_???.f90 = core CLM 1-D subroutines
  !  	  drv_???.f90 = external CLM driver subroutines, intended to be modified
  !                     for various applications
  !
  !   F90 Conventions: The CLM code fully uses many features of FORTRAN90 that
  !   users should become familiar with, as they make the code far more powerful
  !   than F77 based land models.  The core of this power comes from the use
  !   of a 1-D CLM module that can me dimensioned in the driver however the
  !   user sees fit.  In this distribution, we have dimensioned the CLM module
  !   as a vector.  However, with very little effort, this can be modified
  !   to use a x,y grid, or various levels of higher dimensions (as would
  !   be useful in parameter calibration studies).
  ! 
  !
  ! DESCRIPTION of clm_main.f90:
  !  CLM 1-D model to advance land states at 1 point, 1 timestep into the future.
  !  It is intended that NO USER MODIFICATION at and below this subroutine
  !  be required to run CLM for any application.  The input starting time in the 
  !  CLM is GMT, and the model time is GMT.  
  !
  ! clm_main FLOW DIAGRAM
  !    -> clm_dynvegpar:            ecosystem dynamics: phenology, vegetation, soil carbon 
  !    -> clm_coszen:               cosine solar zenith angle for next time step
  !    -> clm_surfalb:              albedos for next time step 
  !       -> clm_snowalb:           snow albedos: direct beam
  !       -> clm_snowalb:           snow albedos: diffuse
  !       -> clm_soilalb:           soil/lake albedos
  !       -> clm_twostream:         absorbed, reflected, transmitted solar fluxes (vis dir)   
  !       -> clm_twostream:         absorbed, reflected, transmitted solar fluxes (vis dif)  
  !       -> clm_twostream:         absorbed, reflected, transmitted solar fluxes (nir dir)  
  !       -> clm_twostream:         absorbed, reflected, transmitted solar fluxes (nir dif)  
  !    -> clm_surfrad
  !    -> clm_hydro_canopy
  !    -> clm_thermal 
  !         bare soil =>   
  !            -> clm_qsadv
  !            -> clm_obuini
  !            -> clm_obult
  !         non-bare soil =>
  !            -> clm_leaftem
  !               -> clm_qsadv
  !               -> clm_obuini
  !               -> clm_obult 
  !               -> clm_stomata
  !               -> clm_condch
  !               -> clm_condcq    
  !         -> clm_thermalk
  !         -> clm_tridia
  !         -> clm_meltfreeze  
  !      -> clm_hydro_snow
  !      -> clm_compact     
  !      -> clm_combin
  !      -> clm_subdiv   
  !      -> clm_hydro_irrig
  !      -> clm_hydro_wetice
  !      -> clm_lake
  !      -> clm_snowage
  !      -> clm_balchk
  !      -> clm_hydro_soil

  ! CLM_MAIN INPUTS (through clm module)
  !   istep,dtime,latdeg 
  !
  !   Soil information
  !     csol,watsat,sucsat,bsw,tkmg,tksatu,tkdry,hksat,wtfact,trsmx0  
  !
  !   Vegetation information
  !     z0m,displa,dleaf,elai,esai,rootfr,dewmx       
  !
  !   Atmospheric forcing
  !     forc_u,forc_v,forc_t,forc_q,forc_pbot,forc_rho,itypprc,forc_rain,       
  !     forc_snow,forc_hgt_u,forc_hgt_t,forc_hgt_q,                             
  !
  !   Roughness lengths
  !     zlnd,zsno,csoilc
  !
  !   Numerical finite-difference
  !     capr,cnfac,smpmin,ssi,wimp,pondmx 
  !
  ! CLM_MAIN STATES (initialization required, CLM_MAIN updates these) 
  !     snl,dz,z,zi,t_soisno,t_veg,h2ocan,h2osoi_liq,h2osoi_ice,     
  !     snowage, h2osno, snowdp, t_grnd,         
  !     frac_veg_sno,frac_sno,frac_veg_nosno,h2osoi_vol_srf,                         
  !
  ! CLM_MAIN FLUXES (These are the outputs) 
  !     taux,tauy,eflx_sh_tot,qflx_evap_tot,eflx_lh_tot,eflx_sh_veg,qflx_evap_veg,qflx_tran_veg,  
  !     eflx_sh_grnd,qflx_evap_soi,eflx_lwrad_out,eflx_soil_grnd,t_rad,t_ref2m,qflx_surf,qflx_totrnf,    
  !
  !   Net solar radiation
  !     cosz,forc_lwrad,sabv,sabg,solisb,solisd,         
  !
  ! Diagnostic CLM variables
  !  The diagnostic variables are defined in order to allow the user to output any
  !  variable from the CLM for debugging purposes.  The user sets the number of 
  !  diagnostic output variables, and can then set the diagnostic arrays at 
  !  their discretion anywhere in the CLM code.  The number of surface, soil layer,
  !  and snow layer variables are defined by the user in the drv_clmin.dat file.
  !  The surf array handles 1D tile space variables.  The soil and snow arrays
  !  handle 2D soil layer and snow layer variables, respectively.
  ! 
  ! CLM revision:
  !  The clm_* routines have undergone a major revision, where the CLM module is 
  !  passed down into most of the CLM subroutines.  The new design allows for an
  !  easier implementation of coding changes and reduces the runtime.  A rule adhered
  !  to in the new design was:  a variable used in more than 2 subroutines is added
  !  to the CLM module. 
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 and 1-D Revision 
  !=========================================================================
  ! $Id: clm_main.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : tfrz, istsoil, istwet, istice, denice, denh2o
  ! use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
  implicit none

  ! ------------------- arguments -----------------------------------
  type (clm1d), intent(inout) :: clm    !CLM 1-D Module
  real(r8)    , intent(in)    :: day    !needed for zenith angle calc
  real(r8)    , intent(in)    :: gmt    !needed for irrigation schedule @IMF
  ! -----------------------------------------------------------------

  ! ------------------- local ---------------------------------------
  integer j       !loop index
  real(r8) coszen !cosine of zenith angle
  integer,intent(in)  :: clm_forc_veg
  ! -----------------------------------------------------------------

  ! -----------------------------------------------------------------
  ! Ecosystem dynamics: phenology, vegetation, soil carbon, snow frac
  ! -----------------------------------------------------------------

  call clm_dynvegpar (clm,clm_forc_veg)
  ! -----------------------------------------------------------------
  ! Albedos 
  ! -----------------------------------------------------------------

  call clm_coszen (clm,day,coszen)

  call clm_surfalb (clm,coszen)

  ! -----------------------------------------------------------------
  ! Surface Radiation
  ! -----------------------------------------------------------------

  call clm_surfrad (clm)

  ! -----------------------------------------------------------------
  ! Initial set of previous time step variables 
  ! -----------------------------------------------------------------

  clm%h2osno_old = clm%h2osno  ! snow mass at previous time step
  clm%h2ocan_old = clm%h2ocan  ! depth of water on foliage at previous time step
  if (.not.clm%lakpoi) then
     do j = 1,nlevsoi
        clm%h2osoi_liq_old(j)=clm%h2osoi_liq(j)+clm%h2osoi_ice(j)
     enddo
     do j = clm%snl+1, 0       ! ice fraction of snow at previous time step
        clm%frac_iceold(j) = clm%h2osoi_ice(j)/(clm%h2osoi_liq(j)+clm%h2osoi_ice(j)) 
     enddo
  endif

  ! -----------------------------------------------------------------
  ! Energy AND Water balance for non-lake points
  ! -----------------------------------------------------------------

  if (.not. clm%lakpoi) then   

     ! determine beginning water balance for non-lake points

     !clm%begwb = clm%h2ocan + clm%h2osno  ! water balance at previous time step
     !do j = 1, nlevsoi
     !@ Watch the units!
     !  clm%begwb = clm%begwb + (clm%h2osoi_ice(j) + clm%h2osoi_liq(j)) *  denh2o / 1000.0d0
     !	 if (clm%istep ==1 ) then
     !	 endif 
     !  enddo

     !@ Lets do it my way
     !@ Here we add the total water mass of the layers from Parflow to close water balance
     !@ We can use clm(1)%dz(1) because the grids are equidistant and congruent
     !   clm%begwb = 0.0d0 !@only interested in wb below surface
     !   do j = 1, parfl_nlevsoi
     !      clm%begwb =  clm%begwb + clm%pf_vol_liq(j) * clm%dz(1) * 1000.0d0
     !      clm%begwb = clm%begwb + clm%pf_vol_liq(j)/clm%watsat(j) * 0.0001*0.5d0 * clm%pf_press(j)    
     !   enddo
     !@ Why is it handled this way and not as follows: clm%begwb = clm%endwb ????

     clm%begwb = clm%endwb

     ! Apply irrigation
     ! IMF @ NOTE: 
     ! New irrigation scheme gives three options: Spray, Drip, Instant
     ! Irrigation is not applied *before* clm_hydro_canopy to allow intercpetion of spray irrigation
     ! clm_hydro_irrig determines whether to irrigate and how much
     ! ...spray and drip irrigation are then applied in clm_hydro_canopy by adding 
     !    qflx_qirr to the rain rate or throughfall, respectively.
     ! ...instant irrigation (i.e., artificial inflation of soil moisture) is applied 
     !    in ParFlow at the next dt by adding qflx_qirr_inst to pf_flux

     call clm_hydro_irrig (clm,gmt)

     ! Determine canopy interception and precipitation onto ground surface.
     ! Determine the fraction of foliage covered by water and the fraction
     ! of foliage that is dry and transpiring. Initialize snow layer if the 
     ! snow accumulation exceeds 10 mm.

     call clm_hydro_canopy (clm)

     ! Determine thermal processes and surface fluxes

     call clm_thermal (clm)

     ! Determine the change of snow mass and the snow water onto soil

     call clm_hydro_snow (clm)

     ! Determine Soil hydrology 

     if (clm%itypwat == istsoil) call clm_hydro_soil (clm)

     ! Determine compaction rate for snow - combine/divide thin or thick snow elements

     if (clm%snl < 0) then

        ! Natural compaction and metamorphosis. The compaction rate is recalculated every timestep.

        call clm_compact (clm)

        ! Combine thin snow elements

        call clm_combin (clm)

        ! Divide thick snow elements

        call clm_subdiv (clm)

        ! Set zero to the empty node

        if (clm%snl > -nlevsno) then
           clm%snowage = 0.
           do j = -nlevsno+1, clm%snl
              clm%h2osoi_ice(j) = 0.
              clm%h2osoi_liq(j) = 0.
              clm%t_soisno(j) = 0.
              clm%dz(j) = 0.
           enddo
        endif

     endif

     ! Update ground temperature

     clm%t_grnd = clm%t_soisno(clm%snl+1)

     ! Irrigate crops if necessary
     ! IMF @ 
     ! Irrigation now called above, before clm_hydro_canopy, to allow intercpetion of spray irrigation 
     ! if (clm%itypwat == istsoil) call clm_hydro_irrig (clm)

     !@Stefan: Major parts below have been moved to pf_couple.f90 to calculated the mass balance
     !@ Determine volumetric soil water

     !     do j = 1,nlevsoi
     !        clm%h2osoi_vol(j) = clm%h2osoi_liq(j)/(clm%dz(j)*denh2o) &
     !                          + clm%h2osoi_ice(j)/(clm%dz(j)*denice)
     !     end do

     ! Determine ending water balance for non-lake points

     !     clm%endwb=clm%h2ocan+clm%h2osno
     !     do j = 1, nlevsoi
     !        clm%endwb = clm%endwb + clm%h2osoi_ice(j) + clm%h2osoi_liq(j)
     !     enddo

     ! Determine wetland and land ice hydrology (must be placed here since need snow 
     ! updated from clm_combin) and ending water balance

     !     if (clm%itypwat==istwet .or. clm%itypwat==istice) call clm_hydro_wetice (clm)

     ! -----------------------------------------------------------------
     ! Energy AND Water balance for lake points
     ! -----------------------------------------------------------------

     !@Stefan: The "lake" routine is still called from here
  else if (clm%lakpoi) then    

     call clm_lake (clm)

     !     do j = 1,nlevsoi
     !        clm%h2osoi_vol(j) = 1.0
     !     end do

  endif

  ! -----------------------------------------------------------------
  ! Update the snow age
  ! -----------------------------------------------------------------

  !@Stefan: The "snowage" routine is still called from here
  call clm_snowage (clm)

  ! -----------------------------------------------------------------
  ! Check the energy and water balance
  ! -----------------------------------------------------------------
  !
  !  call clm_balchk (clm, clm%istep)
  !@Stefan: End of change
  return
end subroutine clm_main
