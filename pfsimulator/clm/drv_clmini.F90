!#include <misc.h>

subroutine drv_clmini (drv, grid,pf_porosity, tile, clm, istep_pf, clm_forc_veg)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! DESCRIPTION:
!  Setup initial CLM information in tile space.
!  
!  The observational initial values are the first choice.  If there are no
!  observational initial data, arbitrarily set the initial fields, and then
!  spin up the first model year to equilibrium state, and take the 
!  equilibrium state variables as the initial values.
!
!   The arbitrary initial data are created based on the following rules:
!    (1) Foliage temperature is initially set to lowest the atmospheric 
!        model air-temperature.
!    (2) Canopy water storage is set to zero.
!    (3) Soil temperatures are initialized as in bucket type
!        parameterizations using the lowest atmospheric model
!        air-temperature and a climatological deep-ground temperature.
!    (4) Soil moistures are initialized to a percentage of field capacity, 
!        and the percent of liquid water and ice lens are determined by the 
!        layer temperatures.
!    (5) If the depth of snow is known, then subdivide the snow pack 
!        up to five layers based on the following rules: 
!         From top layer to bottom layer
!         minimum thickness:     0.010, 0.015, 0.025, 0.055, 0.115 (m),
!         and maximum thickness: 0.02, 0.05, 0.11, 0.23, and  >0.23 m,
!    (6) The snow layer temperature is set to the surface air temperature.
!        If the air temperature is greater than freezing point, then it is 
!        set to 273.16 K.  If no information on snow is available, then 
!        the snow mass for areas of permanent land ice is initially set to 
!        50000 kg m-2. For other areas, all snow related variables 
!        are set to 0.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!   3 March 2000:     Jon Radakovich; Revision for diagnostic output
!=========================================================================
! $Id: drv_clmini.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use infnan
  use drv_module          ! 1-D Land Model Driver variables
  use drv_gridmodule      ! Grid-space variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! CLM tile variables 
  use clm_varcon, only : denice, denh2o, istwet, istice, istsoil, istslak, istdlak
  use clm_varpar, only : nlevsoi, nlevsno
  implicit none

!=== Arguments ===========================================================

  type (drvdec)  :: drv              
  type (griddec) :: grid(drv%nc,drv%nr)   
  type (tiledec) :: tile
  type (clm1d)   :: clm  
  integer        :: istep_pf

!=== Local Variables =====================================================

  integer  i, j, L           !loop indices
  real(r8) bd                !bulk density of dry soil material [kg/m^3]
  real(r8) tkm               !mineral conductivity
  real(r8) zlak(1:10)   !temporary z; currenly hard coded initialization
  real(r8) dzlak(1:10)  !temporary dz; currenly hard coded initialization
  real(r8) xksat
  real(r8) pf_porosity(nlevsoi)  !porosity from PF, replaces watsat clm var
  integer,intent(in)  :: clm_forc_veg

!=== End Variable List ===================================================

! ========================================================================
! Initialize Time Parameters - additional init done in drv_tick 
! ========================================================================

  clm%istep = istep_pf
  clm%dtime = drv%ts

! ========================================================================
! TIME CONSTANT [1]
!  Define layer structure (here "0" refers to soil surface and 
! "nlevsoi" refers to the bottom of model soil)
! ========================================================================

  if (nlevlak /= nlevsoi) then
     write(6,*)'number of soil levels and number of lake levels must be the same'
     write(6,*)'nlevsoi= ',nlevsoi,' nlevlak= ',nlevlak
     stop
  endif

  if (clm%itypwat == istdlak) then        !assume all lakes are deep lakes

     if (nlevlak > 10) then
        write(6,*) ' must set new values for lake levels > 10'
        stop
     endif

     dzlak(1) = 1.               
     dzlak(2) = 2.               
     dzlak(3) = 3.               
     dzlak(4) = 4.               
     dzlak(5) = 5.
     dzlak(6) = 7.
     dzlak(7) = 7.
     dzlak(8) = 7.
     dzlak(9) = 7.
     dzlak(10)= 7.

     zlak(1) =  0.5
     zlak(2) =  1.5
     zlak(3) =  4.5
     zlak(4) =  8.0
     zlak(5) = 12.5
     zlak(6) = 18.5
     zlak(7) = 25.5
     zlak(8) = 32.5
     zlak(9) = 39.5
     zlak(10)= 46.5

     do j = 1,nlevlak
        clm%z(j) = zlak(j)
        clm%dz(j) = dzlak(j)
     end do
     
  else if (clm%itypwat == istslak) then   !shallow lake (not used)
     
     clm%dz(1:nlevlak) = NaN
     clm%z(1:nlevlak)  = NaN

  else                                    !soil, ice, wetland
     
     do j = 1, nlevsoi
 
        ! IMF: Original CLM formulation
        !      (removed when PF and CLM coupled; scalez no longer read from input files?)
        ! clm%z(j) = tile%scalez*(exp(0.5*(j-0.5))-1.)     !node depths
        
        ! IMF: Replaced by constant DZ from ParFlow...
        !      (z set based on constant ParFlow DZ)
        ! clm%z(j) = drv%dz*(dble(j)-0.5d0)
        
        ! IMF: Replaced by variable DZ from ParFlow...
        !      (dummy values set here, actual values set in clm.F90 initialization)
        !      (z[1] = .5*dz*pf_dz_mult[1])
        !      (z[k] = sum(dz*pf_dz_mult[1],dz*pf_dz_mult[2],...dz*pf_dz_mult[k]) - 0.5*dz*pf_dz_mult[k]) 
        clm%z(j) = drv%dz*(dble(j)-0.5d0)

     enddo
     
     ! IMF: not changed -- same as original CLM 
     clm%dz(1)  = 0.5*(clm%z(1)+clm%z(2))         !thickness b/n two interfaces
     do j = 2,nlevsoi-1
        clm%dz(j)= 0.5*(clm%z(j+1)-clm%z(j-1)) 
     enddo
     clm%dz(nlevsoi)= clm%z(nlevsoi)-clm%z(nlevsoi-1)

     ! IMF: not changed -- same as original CLM
     clm%zi(0)   = 0.                             !interface depths 
     do j = 1, nlevsoi-1
        clm%zi(j)= 0.5*(clm%z(j)+clm%z(j+1))     
     enddo
     clm%zi(nlevsoi) = clm%z(nlevsoi) + 0.5*clm%dz(nlevsoi) 

  endif

! ========================================================================
! TIME CONSTANT [2]
! Initialize root fraction (computing from surface, d is depth in meter):
! Y = 1 -1/2 (exp(-ad)+exp(-bd) under the constraint that
! Y(d =0.1m) = 1-beta^(10 cm) and Y(d=d_obs)=0.99 with beta & d_obs
! given in Zeng et al. (1998).
! ========================================================================

  do j = 1, nlevsoi-1
     clm%rootfr(j) = .5*( exp(-tile%roota*clm%zi(j-1))  &
                        + exp(-tile%rootb*clm%zi(j-1))  &
                        - exp(-tile%roota*clm%zi(j  ))  &
                        - exp(-tile%rootb*clm%zi(j  )) )
  enddo
  clm%rootfr(nlevsoi)=.5*( exp(-tile%roota*clm%zi(nlevsoi-1))&
                         + exp(-tile%rootb*clm%zi(nlevsoi-1)))
  
  ! reset depth variables assigned by user in clmin file 
  do l=1,nlevsoi
     if (grid(tile%col,tile%row)%rootfr /= drv%udef) &
          clm%rootfr(l)=grid(tile%col,tile%row)%rootfr    
  enddo

! ========================================================================
! TIME CONSTANT [3]
! Initialize soil thermal and hydraulic properties
! ========================================================================

! Define the vertical profile of soil thermal and hydraulic properties

  if (clm%itypwat == istsoil) then  ! soil

     do j = 1, nlevsoi
        clm%bsw(j)    = 2.91 + 0.159*tile%clay(j)*100.0
!@      clm%watsat(j) = 0.489 - 0.00126*tile%sand(j)*100.0 Stefan: followed Reed to make it consistent with PILPS
!@        clm%watsat(j) = 0.401d0  !@This varaible is now read in directly in read_array.f90 !@IMF: uncommented b/c used later...must be defined.
       clm%watsat(j) = pf_porosity(j)   !@RMM passed in parflow porosity
        xksat         = 0.0070556 *( 10.**(-0.884+0.0153*tile%sand(j)*100.0)) 

!@== clm%hksat(j)  = xksat * exp(-clm%zi(j)/tile%hkdepth) This is now read in from drv_main from an array in read_array.f90
                
        clm%sucsat(j) = 10. * ( 10.**(1.88-0.0131*tile%sand(j)*100.0) )
        clm%watdry(j) = clm%watsat(j) * (316230./clm%sucsat(j)) ** (-1./clm%bsw(j))
        clm%watopt(j) = clm%watsat(j) * (158490./clm%sucsat(j)) ** (-1./clm%bsw(j))
        tkm           = (8.80*tile%sand(j)*100.0+2.92*tile%clay(j)*100.0) /  &
                        (tile%sand(j)*100.0+tile%clay(j)*100.0)          ! W/(m K)
        bd            = (1.-clm%watsat(j))*2.7e3
        clm%tkmg(j)   = tkm ** (1.- clm%watsat(j))
        clm%tksatu(j) = clm%tkmg(j)*0.57**clm%watsat(j)
        clm%tkdry(j)  = (0.135*bd + 64.7) / (2.7e3 - 0.947*bd)
        clm%csol(j)   = (2.128*tile%sand(j)*100.0+2.385*tile%clay(j)*100.0)/ &
                        (tile%sand(j)*100.0+tile%clay(j)*100.0)*1.e6     ! J/(m3 K)
     enddo

  else                                ! ice/glacier, lakes, wetlands

     do j = 1, nlevsoi
        clm%bsw(j)    = drv%udef
        clm%watsat(j) = 1.
        clm%hksat(j)  = drv%udef
        clm%sucsat(j) = drv%udef
        clm%watdry(j) = 0.
        clm%watopt(j) = 1.
        clm%tkmg(j)   = drv%udef
        clm%tksatu(j) = drv%udef
        clm%tkdry(j)  = drv%udef
        clm%csol(j)   = drv%udef
     enddo

  endif

! ========================================================================
! TIME CONSTANT [4]
! these terms used to be parameters but need to be derived type components
! to be consistent with hybrid code
! ========================================================================

  clm%qe25   =  0.06     ! quantum efficiency at 25c (umol co2 / umol photon)      
  clm%ko25   =  30000.   ! o2 michaelis-menten constant at 25c (pa)                
  clm%kc25   =  30.      ! co2 michaelis-menten constant at 25c (pa)               
  clm%vcmx25 =  33.      ! maximum rate of carboxylation at 25c (umol co2/m**2/s)  
  clm%ako    =  1.2      ! q10 for ko25                                            
  clm%akc    =  2.1      ! q10 for kc25                                            
  clm%avcmx  =  2.4      ! q10 for vcmx25                                          
  clm%bp     =  2000.    ! minimum leaf conductance (umol/m**2/s)                  
  clm%mp     =  9.       ! slope for conductance-to-photosynthesis relationship    
  clm%folnmx =  1.5      ! foliage nitrogen concentration when f(n)=1 (%)          
  clm%folnvt =  2.       ! foliage nitrogen concentration (%)                      
  clm%c3psn  =  1.       ! photosynthetic pathway: 0. = c4, 1. = c3                
  
! ========================================================================
! TIME VARIANT [1]
! Temperatures and snow cover fraction are initialized in CLM
! according to atmospheric temperatures and snow cover.
! ========================================================================

! set water and temperatures to constant values: all points

  clm%h2ocan  = 0.
  clm%snowage = 0. 
  clm%h2osno  = drv%h2osno_ini
  clm%snowdp  = drv%h2osno_ini/250.  !the arbitary snow density = 250 kg/m3
  clm%t_veg   = drv%t_ini
  clm%t_grnd  = drv%t_ini

! For lake points only:

  if (clm%lakpoi) then  !lake 
     if (clm%t_grnd <= 273.16) then
        clm%h2osno = 0.
        clm%snowdp = 0.
     endif
  endif

  clm%acc_errseb = 0.
  clm%acc_errh2o = 0.

! ========================================================================
! TIME VARIANT [2]
! Snow layer number, depth and thickiness 
! ========================================================================

  if (.not. clm%lakpoi) then  !not lake
     if (clm%snowdp < 0.01)then
        clm%snl = 0
        clm%dz(-nlevsno+1:0) = 0.
        clm%z (-nlevsno+1:0) = 0.
        clm%zi(-nlevsno+0:0) = 0.
     else
        if ((clm%snowdp >= 0.01) .AND. (clm%snowdp <= 0.03))then
           clm%snl = -1
           clm%dz(0)  = clm%snowdp
        else if ((clm%snowdp > 0.03) .AND. (clm%snowdp <= 0.04))then
           clm%snl = -2
           clm%dz(-1) = clm%snowdp/2.
           clm%dz( 0) = clm%dz(-1)
        else if ((clm%snowdp > 0.04) .AND. (clm%snowdp <= 0.07))then
           clm%snl = -2
           clm%dz(-1) = 0.02
           clm%dz( 0) = clm%snowdp - clm%dz(-1)
        else if ((clm%snowdp > 0.07) .AND. (clm%snowdp <= 0.12))then
           clm%snl = -3
           clm%dz(-2) = 0.02
           clm%dz(-1) = (clm%snowdp - 0.02)/2.
           clm%dz( 0) = clm%dz(-1)
        else if ((clm%snowdp > 0.12) .AND. (clm%snowdp <= 0.18))then
           clm%snl = -3
           clm%dz(-2) = 0.02
           clm%dz(-1) = 0.05
           clm%dz( 0) = clm%snowdp - clm%dz(-2) - clm%dz(-1)
        else if ((clm%snowdp > 0.18) .AND. (clm%snowdp <= 0.29))then
           clm%snl = -4
           clm%dz(-3) = 0.02
           clm%dz(-2) = 0.05
           clm%dz(-1) = (clm%snowdp - clm%dz(-3) - clm%dz(-2))/2.
           clm%dz( 0) = clm%dz(-1)
        else if ((clm%snowdp > 0.29) .AND. (clm%snowdp <= 0.41))then
           clm%snl = -4
           clm%dz(-3) = 0.02
           clm%dz(-2) = 0.05
           clm%dz(-1) = 0.11
           clm%dz( 0) = clm%snowdp - clm%dz(-3) - clm%dz(-2) - clm%dz(-1)
        else if ((clm%snowdp > 0.41) .AND. (clm%snowdp <= 0.64))then
           clm%snl = -5
           clm%dz(-4) = 0.02
           clm%dz(-3) = 0.05
           clm%dz(-2) = 0.11
           clm%dz(-1) = (clm%snowdp - clm%dz(-4) - clm%dz(-3) - clm%dz(-2))/2.
           clm%dz( 0) = clm%dz(-1)
        else if (clm%snowdp > 0.64)then
           clm%snl = -5
           clm%dz(-4) = 0.02
           clm%dz(-3) = 0.05
           clm%dz(-2) = 0.11
           clm%dz(-1) = 0.23
           clm%dz( 0)=clm%snowdp-clm%dz(-4)-clm%dz(-3)-clm%dz(-2)-clm%dz(-1)
        endif
        do i = 0, clm%snl+1, -1
           clm%z(i) = clm%zi(i) - 0.5*clm%dz(i)
           clm%zi(i-1) = clm%zi(i) - clm%dz(i)
        enddo
     endif
  else   ! lake points
     clm%snl = 0
     clm%dz(-nlevsno+1:0) = 0.
     clm%z (-nlevsno+1:0) = 0.
     clm%zi(-nlevsno+0:0) = 0.
  endif

! ========================================================================
! TIME VARIANT [3]
! Snow/soil temperature
! ========================================================================
  if (.not. clm%lakpoi) then  !not lake
     if (clm%snl < 0) then
        do i = clm%snl+1, 0
           if (drv%t_ini  < 273.16) then
              clm%t_soisno(i) = drv%t_ini
           else
              clm%t_soisno(i) = 273.16 - 1.
           endif
        enddo
     endif
     do i = 1, nlevsoi
        if (clm%itypwat == istice) then
           clm%t_soisno(i) = drv%t_ini
        else if (clm%itypwat == istwet) then
           clm%t_soisno(i) = drv%t_ini
        else
           clm%t_soisno(i) = drv%t_ini
        endif
     end do
  else
     do i = 1, nlevlak
        clm%t_soisno(i) = drv%t_ini
     enddo
  endif

! ========================================================================
! TIME VARIANT [4]
! Snow/soil ice and liquid mass
! ========================================================================

  if (.not. clm%lakpoi) then  !not lake
     if (clm%snl < 0)then
        do i = clm%snl+1, 0
           clm%h2osoi_ice(i) = clm%dz(i)*250.
           clm%h2osoi_liq(i) = 0.
        enddo
     endif
     do i = 1, nlevsoi
        if (clm%t_soisno(i) <= 273.16) then
           clm%h2osoi_ice(i) = clm%dz(i)* drv%sw_ini*clm%watsat(i)*denice
           clm%h2osoi_liq(i) = 0.
           if (clm%itypwat==istwet .or. clm%itypwat==istice) clm%h2osoi_ice(i)=clm%dz(i)*denice
        else
           clm%h2osoi_ice(i) = 0.
           clm%h2osoi_liq(i) = clm%dz(i)* drv%sw_ini*clm%watsat(i)*denh2o
           if (clm%itypwat==istwet .or. clm%itypwat==istice) clm%h2osoi_liq(i)=clm%dz(i)*denh2o
        endif
     enddo
  else    !not used for lake
     do i = -nlevsno+1, nlevlak
        clm%h2osoi_liq(i) = NaN
        clm%h2osoi_ice(i) = NaN
     end do
  endif

! ========================================================================
! TIME VARIANT [5]
! need to set h2osoi_vol (needed by clm_soilalb) -  this is also needed
! upon restart since the albedo calculation is called before h2osoi_vol
! is computed
! ========================================================================

  do l = 1,nlevsoi
     if (.not. clm%lakpoi) then
        clm%h2osoi_vol(l) = clm%h2osoi_liq(l)/(clm%dz(l)*denh2o) &
                          + clm%h2osoi_ice(l)/(clm%dz(l)*denice)
     else
        clm%h2osoi_vol(l) = 1.0
     endif
  end do

! ========================================================================
! TIME VARIANT [6]
! Ecosystem dynamics: phenology, vegetation, soil carbon and snow cover 
! fraction
! ========================================================================

  call clm_dynvegpar (clm,clm_forc_veg)

! ========================================================================
! TIME VARIANT [7]
! Initialize DIAG arrays 
! ========================================================================

!  do i = 1, drv%surfind
!     clm%diagsurf(i) = 0.
!  enddo
!
!  do i = 1, drv%soilind
!     do j = 1, nlevsoi
!        clm%diagsoil(i,j) = 0.
!     enddo
!  enddo
!
!  do i = 1, drv%snowind
!     do j = -nlevsno+1,0
!        clm%diagsnow(i,j) = 0.
!     enddo
!  enddo

  return
end subroutine drv_clmini

