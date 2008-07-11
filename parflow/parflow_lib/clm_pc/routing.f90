subroutine routing(casc,clm,drv,timeroute,ntroute)
 use drv_module          ! 1-D Land Model Driver variables
 use precision
 use clmtype
 use casctype
 implicit none

 type (drvdec):: drv
 type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
 type (casc2D):: casc(drv%nc,drv%nr)

!=== Local variables ========================== 
! real(r8) :: sfx(drv%nc,drv%nr),sfy(drv%nc,drv%nr)   ! Check if kinemtaic or diffusive wave is used
 real(r8) :: sox(drv%nc,drv%nr),soy(drv%nc,drv%nr)
 real(r8) :: alpha_x,alpha_y
 real(r8) :: KW(drv%nc,drv%nr),KE(drv%nc,drv%nr),KN(drv%nc,drv%nr),KS(drv%nc,drv%nr)
 integer  :: ax(drv%nc,drv%nr),ay(drv%nc,drv%nr)
 integer  :: r,c,t
 integer  :: routl, coutl
 real :: timeroute,ntroute

!=== Initialize ===============================
 KW = 0.0d0
 KE = 0.0d0
 KN = 0.0d0
 KS = 0.0d0
 coutl = drv%cout
 routl = drv%rout

!=== Assign slope values to  each cell in the x and y direction
!=== These values may come later from some raster file defined with some raster calculator
   do r = 1, drv%nr
   do c = 1, drv%nc
    sox(c,r) = -0.00d0
	soy(c,r) = -0.001d0
   enddo
  enddo

!=== Determine friction slopes at every point in the domaine (NOT used with kinematic wave) 
  do r = 1, drv%nr
   do c = 1, drv%nc
    
	 if (c == 1) then
!      sfx(c,r) = sox(c,r) + (casc(c+1,r)%h - casc(c,r)%h) / (casc(c,r)%W)
     elseif (c == drv%nc) then
!      sfx(c,r) = sox(c,r) + (casc(c,r)%h - casc(c-1,r)%h) / casc(c,r)%W
     else
!      sfx(c,r) = sox(c,r) + (casc(c+1,r)%h - casc(c-1,r)%h) / (2.0d0 * casc(c,r)%W)
     endif

!=== Assign sign for flow in x-direction
	 if (sox(c,r) < 0.0d0) then              !Check whether kinematic or diffusive wave is used
	  ax = -1
     else
	  ax = 1
     endif

 	 if (r == 1) then
!      sfy(c,r) = soy(c,r) + (casc(c,r+1)%h - casc(c,r)%h) / (casc(c,r)%W)
     elseif (r == drv%nr) then
!      sfy(c,r) = soy(c,r) + (casc(c,r)%h - casc(c,r-1)%h) / (casc(c,r)%W)
     else
!      sfy(c,r) = soy(c,r) + (casc(c,r+1)%h - casc(c,r-1)%h) / (2.0d0 * casc(c,r)%W)
     endif

!=== Assign sign for flow in y-direction
     if (soy(c,r) < 0.0d0) then               !Check whether kinematic or diffusive wave is used
	  ay = -1
     else
	  ay = 1
     endif

	enddo
   enddo



!=== Determine fluxes at every point in the domaine (Manning's coeff is homogeneous across domain!)
  do r = 1, drv%nr
   do c = 1, drv%nc

    alpha_x = sqrt(abs(sox(c,r))) / casc(c,r)%manning                           ! Check if kinematic or diffusive wave is used
	 casc(c,r)%qx = (-1.0d0) * ax(c,r) * alpha_x * casc(c,r)%h**(5.0d0/3.0d0)

    alpha_y = sqrt(abs(soy(c,r))) / casc(c,r)%manning                           ! Check if kinematic or diffusive wave is used
     casc(c,r)%qy = (-1.0d0) * ay(c,r) * alpha_y * casc(c,r)%h**(5.0d0/3.0d0)

   enddo
  enddo

!=== Determine upwind coefficients
  do r = 1, drv%nr
   do c = 1, drv%nc
    if (c == 1) then
	 KE(c,r) = max(casc(c,r)%qx,0.0d0) + max(-casc(c+1,r)%qx,0.0d0)
	 KW(c,r) = 0.0d0
    elseif(c == drv%nc) then
     KE(c,r) = 0.0d0
	 KW(c,r) = max(casc(c-1,r)%qx,0.0d0) + max(-casc(c,r)%qx,0.0d0)	
	else 	 
     KE(c,r) = max(casc(c,r)%qx,0.0d0) + max(-casc(c+1,r)%qx,0.0d0)
	 KW(c,r) = max(casc(c-1,r)%qx,0.0d0) + max(-casc(c,r)%qx,0.0d0)
    endif
    
	if (r == 1) then
 	 KN(c,r) = max(casc(c,r)%qy,0.0d0) + max(-casc(c,r+1)%qy,0.0d0)
	 KS(c,r) = 0.0d0
    elseif (r  == drv%nr) then
	 KN(c,r) = 0.0d0
	 KS(c,r) = max(casc(c,r-1)%qy,0.0d0) + max(-casc(c,r)%qy,0.0d0) 
    else
	 KN(c,r) = max(casc(c,r)%qy,0.0d0) + max(-casc(c,r+1)%qy,0.0d0)
	 KS(c,r) = max(casc(c,r-1)%qy,0.0d0) + max(-casc(c,r)%qy,0.0d0)
    endif 

   enddo
  enddo 	 

!=== Determine new flow depth (casc%depth) in the domaine before partioning into runoff and infil
  t = 0
  do r = 1, drv%nr
   do c = 1, drv%nc
    t = t + 1
     
	 casc(c,r)%depth = casc(c,r)%h - (drv%dt / casc(c,r)%W) * ((KE(c,r) - KW(c,r)) + (KN(c,r) - KS(c,r))) &
	                  + clm(t)%qflx_top_soil * drv%dt
	 
   enddo
  enddo   
 
!=== Determine outflow out of the domaine at the outlet (here it is a point-outlet)
!=== This part was moved to the subroutine "ovrldepth" 
! casc(coutl,routl)%qoutov = casc(coutl,routl)%depth * casc(coutl,routl)%W / (1000.0d0**2.0d0 * drv%dt)
! write(2005,*) (clm(1)%istep-1.0d0+timeroute/ntroute),  casc(coutl,routl)%qoutov
! casc(coutl,routl)%depth = 0.0d0

!=== Assign new flow depth to grid
!=== This part was moved to the subroutine "ovrldepth" 
! do r = 1, drv%nr
!  do c = 1, drv%nc
!        casc(c,r)%h = casc(c,r)%hnew
!  enddo
! enddo

 end subroutine routing	