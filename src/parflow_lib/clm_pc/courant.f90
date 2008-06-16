subroutine courant(casc,clm,drv, ntroute)
 use drv_module          ! 1-D Land Model Driver variables
 use precision
 use clmtype
 use casctype
 implicit none

 type (drvdec):: drv
 type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
 type (casc2D):: casc(drv%nc,drv%nr)

!=== Local variables ========================== 
 real(r8) :: ck(drv%nc,drv%nr),crit,cour(drv%nc,drv%nr)
 real(r8) :: alpha
 integer  :: r,c,l,t,ntroute
 
!=== Array is initialized with large values in case of (near) zero velocities at the boundaries 
 ck = 1.0d+30
 cour = 0.0d0  
 if (clm(1)%istep == 1) casc%h=0.0d0

     alpha = sqrt(drv%max_sl) / casc(1,1)%manning

!=== Calculate wave celerity - W ratio ======================================
  t = 0
  do r = 1,drv%nr
   do c = 1, drv%nc
     t = t + 1
     cour(c,r) = (5.0d0/3.0d0)*alpha*(casc(c,r)%h + clm(t)%qflx_surf * clm(1)%dtime)**(2.0d0/3.0d0)
	 if (cour(c,r) == 0.0d0) then
	  ck(c,r) = 1.0d+30
     else
	  ck(c,r) = casc(c,r)%W * 1.0/ cour(c,r)
     endif 
   enddo !c-loop
  enddo !r-loop

!=== Check Courant condition
      crit = drv%ts / minval(ck)
	  ntroute =  int(crit + 1.0d0)
	  write(*,*) "min(cour)",minval(cour)," clm_dt",drv%ts,crit,ntroute 
	  write(*,*)


end subroutine courant
