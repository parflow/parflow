subroutine overland(casc,clm,drv,ntroute)
 use drv_module          ! 1-D Land Model Driver variables
 use precision
 use clmtype
 use casctype
 implicit none

 type (drvdec):: drv
 type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
 type (casc2D):: casc(drv%nc,drv%nr)

 integer :: r,c,l,t
 integer :: timeroute,ntroute 

!=== Loop over modified timestep ==============================
 do timeroute = 1, ntroute

!=== Set new time step based on Courant condition ============= 
 if (timeroute == 1)  drv%dt = clm(1)%dtime / float(ntroute)

!=== Routing loop over cells/tiles: overland routing ==========================   
	print *,"Call routing"
	call routing(casc,clm,drv,timeroute,ntroute)
	
!=== Update water depth applied at the surface, infiltration and evap rates
    call ovrldepth(casc,clm,drv,ntroute,timeroute)	
 
!=== Write 2D flowdepth to a file ======   
     write(2006,*)"dtime:", drv%dt
     do r = drv%nr, 1, -1
      write(2006,'(100(f7.3,1x))') (casc(c,r)%h, c =1, drv%nc)
     enddo
 
 enddo 
!=== end of loop over modified time step =======================================

end subroutine overland 




