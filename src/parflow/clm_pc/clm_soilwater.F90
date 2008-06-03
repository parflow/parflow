!#include <misc.h>

subroutine clm_soilwater (vol_liq,    eff_porosity,     qinfl,   sdamp,  &
                          dwat,       hk,               dhkdw,   clm,pf_flux )

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
!  Soil moisture is predicted from a 10-layer model (as with soil 
!  temperature), in which the vertical soil moisture transport is governed
!  by infiltration, runoff, gradient diffusion, gravity, and root 
!  extraction through canopy transpiration.  The net water applied to the
!  surface layer is the snowmelt plus precipitation plus the throughfall 
!  of canopy dew minus surface runoff and evaporation. 
!
!  The vertical water flow in an unsaturated porous media is described by
!  Darcy's law, and the hydraulic conductivity and the soil negative 
!  potential vary with soil water content and soil texture based on the work 
!  of Clapp and Hornberger (1978) and Cosby et al. (1984). The equation is
!  integrated over the layer thickness, in which the time rate of change in
!  water mass must equal the net flow across the bounding interface, plus the
!  rate of internal source or sink. The terms of water flow across the layer
!  interfaces are linearly expanded by using first-order Taylor expansion.  
!  The equations result in a tridiagonal system equation. 
!
!  Note: length units here are all millimeter 
!  (in temperature subroutine uses same soil layer 
!  structure required but lengths are m)
!
!  Richards equation:
!
!  d wat      d     d wat d psi
!  ----- = - -- [ k(----- ----- - 1) ] + S
!    dt      dz       dz  d wat
!
!  where: wat = volume of water per volume of soil (mm**3/mm**3)
!  psi = soil matrix potential (mm)
!  dt  = time step (s)
!  z   = depth (mm)
!  dz  = thickness (mm)
!  qin = inflow at top (mm h2o /s) 
!  qout= outflow at bottom (mm h2o /s)
!  s   = source/sink flux (mm h2o /s) 
!  k   = hydraulic conductivity (mm h2o /s)
!
!                        d qin                  d qin
!  qin[n+1] = qin[n] +  --------  d wat(j-1) + --------- d wat(j)
!                        d wat(j-1)             d wat(j)
!                 ==================|================= 
!                                   < qin 
!
!                  d wat(j)/dt * dz = qin[n+1] - qout[n+1] + S(j) 
!
!                                   > qout
!                 ==================|================= 
!                         d qout               d qout
!  qout[n+1] = qout[n] + --------- d wat(j) + --------- d wat(j+1)
!                         d wat(j)             d wat(j+1)
!
!
!  Solution: linearize k and psi about d wat and use tridiagonal 
!  system of equations to solve for d wat, 
!  where for layer j
!
!
!  r_j = a_j [d wat_j-1] + b_j [d wat_j] + c_j [d wat_j+1]
!
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_soilwater.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype
  use clm_varpar, only : nlevsoi
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm	 !CLM 1-D Module

  real(r8), intent(in) ::         &
       qinfl,                     & ! net water input from top [mm h2o/s]
       sdamp,                     & ! extrapolates soiwat dependence of evaporation
       eff_porosity(1 : nlevsoi), & ! effective porosity
       vol_liq(1 : nlevsoi)         ! soil water per unit volume [mm/mm]

  real(r8), intent(out) ::        &
       dwat (1 : nlevsoi),        & ! change of soil water [m3/m3]
       hk   (1 : nlevsoi),        & ! hydraulic conductivity [mm h2o/s]
       dhkdw(1 : nlevsoi)           ! d(hk)/d(vol_liq)

!=== Local Variables =====================================================                  

  integer j                 ! do loop indices

  real(r8) amx(1:nlevsoi),    & ! "a" left off diagonal of tridiagonal matrix
       bmx(1:nlevsoi),    & ! "b" diagonal column for tridiagonal matrix
       cmx(1:nlevsoi),    & ! "c" right off diagonal tridiagonal matrix
       z (1 : nlevsoi),   & ! layer depth [mm]
       dz(1 : nlevsoi),   & ! layer thickness [mm]
       den,               & ! used in calculating qin, qout
       dqidw0,            & ! d(qin)/d(vol_liq(i-1))
       dqidw1,            & ! d(qin)/d(vol_liq(i))
       dqodw1,            & ! d(qout)/d(vol_liq(i))
       dqodw2,            & ! d(qout)/d(vol_liq(i+1))
       dsmpdw(1:nlevsoi), & ! d(smp)/d(vol_liq)
       num,               & ! used in calculating qin, qout
       qin,               & ! flux of water into soil layer [mm h2o/s]
       qout,              & ! flux of water out of soil layer [mm h2o/s]
       rmx(1:nlevsoi),    & ! "r" forcing term of tridiagonal matrix
       s_node,            & ! soil wetness
       s1,                & ! "s" at interface of layer
       s2,                & ! k*s**(2b+2)
       smp(1:nlevsoi)       ! soil matrix potential [mm]

! RMM char edits for linking, 10/02
  integer width_step, i_ST, I_err
  character*7 ST
    real(r8) sat_top_gw, dummy_time,pf_flux(1:nlevsoi)  ! Stefan: fluxes for the soil layers; passed to PF

!=== End Variable List ===================================================
  dwat = 0   !Stefan: initialize 0.
  do j = 1, nlevsoi
     z(j) = clm%z(j)*1.e3
     dz(j) = clm%dz(j)*1.e3
  enddo

! Set zero to hydraulic conductivity if effective porosity 5% in any of 
! two neighbor layers or liquid content (theta) less than 0.001

  do j = 1, nlevsoi
     if (      (eff_porosity(j) < clm%wimp) &
          .OR. (eff_porosity(min(nlevsoi,j+1)) < clm%wimp) &
          .OR. (vol_liq(j) <= 1.e-3))then
        hk(j) = 0.
        dhkdw(j) = 0.
     else
        s1 = 0.5*(vol_liq(j)+vol_liq(min(nlevsoi,j+1))) / &
            (0.5*(clm%watsat(j)+clm%watsat(min(nlevsoi,j+1))))
        s2 = clm%hksat(j)*s1**(2.*clm%bsw(j)+2.)
        hk(j) = s1*s2  
        dhkdw(j) = (2.*clm%bsw(j)+3.)*s2*0.5/clm%watsat(j)
        if(j == nlevsoi) dhkdw(j) = dhkdw(j) * 2.
     endif
  enddo

! Evaluate hydraulic conductivity, soil matric potential,
! d(smp)/d(vol_liq), and d(hk)/d(vol_liq).

!  do j = 1, nlevsoi

!     if (clm%t_soisno(j)>273.16) then

!        s_node = max(vol_liq(j)/clm%watsat(j),0.01)
!        s_node = min(1.,s_node)
!        smp(j) = -clm%sucsat(j)*s_node**(-clm%bsw(j))
!        smp(j) = max(clm%smpmin, smp(j))        ! Limit soil suction
!        dsmpdw(j) = -clm%bsw(j)*smp(j)/(s_node*clm%watsat(j))

!    else

! When ice is present, the matric potential is only related to temperature
! by (Fuchs et al., 1978: Soil Sci. Soc. Amer. J. 42(3):379-385)
! Unit 1 Joule = 1 (kg m2/s2), J/kg /(m/s2) ==> m ==> 1e3 mm 

!        smp(j) = 1.e3 * 0.3336e6/9.80616*(clm%t_soisno(j)-273.16)/clm%t_soisno(j)
!        smp(j) = max(clm%smpmin, smp(j))        ! Limit soil suction
!        dsmpdw(j) = 0.

!     endif
!  enddo

! Set up r, a, b, and c vectors for tridiagonal solution
! Node j=1

!write(999,*) qinfl

!  j      = 1
!  qin    = qinfl
!  den    = (z(j+1)-z(j))
!  num    = (smp(j+1)-smp(j)) - den
!  qout   = -hk(j)*num/den
!  dqodw1 = -(-hk(j)*dsmpdw(j)   + num*dhkdw(j))/den
!  dqodw2 = -( hk(j)*dsmpdw(j+1) + num*dhkdw(j))/den
!  rmx(j) =  qin - qout - clm%qflx_tran_veg*clm%rootfr(j)
!  amx(j) =  0.
!  bmx(j) =  dz(j)*(sdamp+1./clm%dtime) + dqodw1
!  cmx(j) =  dqodw2

! Nodes j=2 to j=nlevsoi-1

!  do j = 2, nlevsoi - 1
!     den    = (z(j) - z(j-1))
!     num    = (smp(j)-smp(j-1)) - den
!     qin    = -hk(j-1)*num/den
!     dqidw0 = -(-hk(j-1)*dsmpdw(j-1) + num*dhkdw(j-1))/den
!     dqidw1 = -( hk(j-1)*dsmpdw(j)   + num*dhkdw(j-1))/den
!     den    = (z(j+1)-z(j))
!     num    = (smp(j+1)-smp(j)) - den
!     qout   = -hk(j)*num/den
!     dqodw1 = -(-hk(j)*dsmpdw(j)   + num*dhkdw(j))/den
!     dqodw2 = -( hk(j)*dsmpdw(j+1) + num*dhkdw(j))/den
!     rmx(j) =  qin - qout - clm%qflx_tran_veg*clm%rootfr(j)
!     amx(j) = -dqidw0
!     bmx(j) =  dz(j)/clm%dtime - dqidw1 + dqodw1
!     cmx(j) =  dqodw2
!  enddo

! Node j=nlevsoi

!  j      = nlevsoi
!  den    = (z(j) - z(j-1))
!  num    = (smp(j)-smp(j-1)) - den
!  qin    = -hk(j-1)*num/den
!  dqidw0 = -(-hk(j-1)*dsmpdw(j-1) + num*dhkdw(j-1))/den
!  dqidw1 = -( hk(j-1)*dsmpdw(j)   + num*dhkdw(j-1))/den
!  qout   =  hk(j)
!  dqodw1 =  dhkdw(j)
!  rmx(j) =  qin - qout - clm%qflx_tran_veg*clm%rootfr(j)
!  amx(j) = -dqidw0
!  bmx(j) =  dz(j)/clm%dtime - dqidw1 + dqodw1
!  cmx(j) =  0.

! Solve for dwat

!  call clm_tridia (nlevsoi ,amx ,bmx ,cmx ,rmx ,dwat)

!  set up fluxes for parflow couple
write(199,*) clm%dtime*dble(clm%istep)
pf_flux(1) = qinfl - clm%qflx_tran_veg*clm%rootr(1)
write(199,*) 1, pf_flux(1)
  do j = 2, nlevsoi
	pf_flux(j) = - clm%qflx_tran_veg*clm%rootr(j)
	write(199,*) j, pf_flux(j)
  end do



!width_step = int(clm%istep/2) + 1
write(ST,"(i7)") clm%istep
width_step = 7 - int(dlog10(dble(clm%istep)))

!write(299,"('set ts(',a,') ',e12.4)"),clm%istep, clm%dtime*dble(clm%istep)

open (299,file='qflux.tcl', blank='NULL')

write(299,"('set ts(',a,') ',f15.4)"),ST(width_step:), clm%dtime*dble(clm%istep)
! need to make 1, j

write(ST,"(i7)") clm%istep-1
if (clm%istep-1 > 0) then
width_step = 7 - int(dlog10(dble(clm%istep-1)))
else
width_step = 7
end if 

  do j = 1, nlevsoi
  if (j < 10) then
    write(299,"('set qflux(',i1,') ',e12.4)"),j, -pf_flux(j)
	else
	write(299,"('set qflux(',i2,') ',e12.4)"),j, -pf_flux(j)
  end if
  end do

write(299,"('set num_ts ',i7)") clm%istep
write(299,"('set ts(',a,') ',f15.4)"),ST(width_step:), clm%dtime*dble(clm%istep-1)

close (299)
! run paflow for equivalent timestep
print*, ' calling PF ts:',clm%istep,' time:',clm%dtime*dble(clm%istep)
  I_err = SYSTEM("tclsh pfstep.tcl")
! determine satn, read back in
  I_err = SYSTEM("tclsh saturation_root_layer.tcl")
write(199,*) 
close(199)

! read in satn from PF
! from last timestep's run of ParFlow, passed back as soil layer satn
  open(499, file='pf_sat_root_layer.txt')
  
  read(499,*) dummy_time, vol_liq(1:nlevsoi)
  do j=1, nlevsoi
   vol_liq(j) = vol_liq(j) * clm%watsat(j)
  end do
  close(499)


! remove density component from flux calc for back pass
!dqidw0 = dqidw0 *den

end subroutine clm_soilwater
