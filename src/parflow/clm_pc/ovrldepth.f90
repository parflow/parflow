subroutine ovrldepth(casc,clm,drv,ntroute,timeroute)
 use precision
 use clmtype
 use casctype
 use drv_module
 implicit none

 type (drvdec):: drv
 type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
 type (casc2D):: casc(drv%nc,drv%nr)

!=== Local variables ==========================
 integer :: coutl,routl,t,c,r
 real(r8):: alpha_outl,hold,qnew
 integer :: ntroute,timeroute
 real(r8):: lineout
 real(r8):: old_depthr,old_surfr,old_infiltr

!=== Initialize local variables
if (timeroute == 1) then
 t = 0
 do r = 1, drv%nr
  do c = 1,drv%nc
   t = t + 1
   if (clm(1)%istep == 1) then
    casc(c,r)%dqov = 0.0d0
    casc(c,r)%h = 0.0d0
    old_depthr = 0.0d0
    old_surfr = 0.0d0
    old_infiltr = 0.0d0
   endif 
   casc(c,r)%tot_vol = 0.0d0
   casc(c,r)%tot_infl = 0.0d0
   casc(c,r)%tot_surf = 0.0d0
   casc(c,r)%qoutov = 0.0d0
   casc(c,r)%sum_volout = 0.0d0
  enddo
 enddo  
endif 

 t = 0
 do r = 1, drv%nr
  do c = 1, drv%nc
   t = t + 1
   
!=== Fraction of water available for suface runoff    
   casc(c,r)%h   = clm(t)%frac * casc(c,r)%depth
   casc(c,r)%surf = casc(c,r)%h / drv%dt

!=== New infiltration depth/rate   
   casc(c,r)%infiltr = (1.0d0 - clm(t)%frac) * casc(c,r)%depth - clm(t)%qflx_evap_grnd * drv%dt
   casc(c,r)%infiltr = casc(c,r)%infiltr / drv%dt
   casc(c,r)%depth = casc(c,r)%depth / drv%dt ! reformulate it as a rate
   
!== Average total water applied at the surface (mm/s)
   casc(c,r)%tot_vol = casc(c,r)%tot_vol + casc(c,r)%depth * drv%dt 
   old_depthr = casc(c,r)%depth
   if (timeroute == ntroute) clm(t)%tot_surf = casc(c,r)%tot_vol / clm(1)%dtime    
   
!=== Average surface flux calculation (mm/s)
   casc(c,r)%tot_surf = casc(c,r)%tot_surf + casc(c,r)%surf * drv%dt
   old_surfr = casc(c,r)%surf
   if (timeroute == ntroute) clm(t)%qflx_surf = casc(c,r)%tot_surf / clm(1)%dtime
                  
!=== Average infiltration calculation (mm/s; what's left from mass balance)
   casc(c,r)%tot_infl = casc(c,r)%tot_infl + casc(c,r)%infiltr * drv%dt
   old_infiltr = casc(c,r)%infiltr
   if (timeroute == ntroute) clm(t)%qflx_infl = casc(c,r)%tot_infl / clm(1)%dtime
       
   enddo !r-loop
 enddo ! c-loop

!=== Outflow along line
lineout = 0.0d0
 do c = 1, drv%nc
  lineout = lineout + dsqrt(0.01d0)/casc(c,drv%nr)%manning * (casc(c,drv%nr)%h)**(5.0d0/3.0d0)/casc(c,drv%nr)%W
  casc(c,r)%h = casc(c,r)%h - dsqrt(0.01d0)/casc(c,drv%nr)%manning * (casc(c,drv%nr)%h)**(5.0d0/3.0d0)  &
                * drv%dt / casc(c,drv%nr)%W
 enddo
 casc(drv%cout,drv%rout)%qoutov = lineout / drv%nc ! This is the average outflow rate along the line

!=== In case of outflow in a single cell
!casc(drv%cout,drv%rout)%qoutov = casc(drv%cout,drv%rout)%h / drv%dt
!if (casc(drv%cout,drv%rout)%qoutov < 1.0d-100) casc(drv%cout,drv%rout)%qoutov = 0.0d0
!casc(drv%cout,drv%rout)%h = 0.0d0                        ! At the outlet water is taken completely out of the domain
!write(2005,*) (clm(1)%istep-1.0d0+float(timeroute)/float(ntroute)) ,casc(drv%cout,drv%rout)%qoutov

end subroutine ovrldepth


