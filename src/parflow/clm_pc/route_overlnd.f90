subroutine route_overlnd(casc,clm,drv)
 use drv_module          ! 1-D Land Model Driver variables
 use precision
 use clmtype
 use casctype
 implicit none

 type (drvdec):: drv
 type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
 type (casc2D):: casc(drv%nc,drv%nr)

!=== Local variables ========================== 
 real(r8) :: sf,so,stordepth,hh,alpha,dqq
 integer  :: r,c,rr,cc,l,t
 real(r8) :: a,exponent,ck
 
 write(2007,*) "timestep:",clm%istep

 t = 0
 do r = 1, drv%nr
  do c = 1, drv%nc
  t = t + 1
   do l=-1, 0
    
	if (((clm(t)%istep/2) * 2 - clm(t)%istep) /= 0 )then
     rr = r - l
	 cc = c + l + 1
     else
	 rr = r + l +1
	 cc = c - l
    endif
   if (rr <= drv%nr .and. cc <= drv%nc) then

!=== Determine friction slope: sf = so - dh/dx ==============================================
	 so = (casc(c,r)%elev - casc(cc,rr)%elev) / casc(c,r)%W
     sf = so - (casc(cc,rr)%h - casc(c,r)%h) / casc(c,r)%W + 1.0e-030
	 if (abs(sf) <= 1.0e-012) sf = 0.0d0

	 hh = casc(c,r)%h
	 stordepth = casc(c,r)%sdep
     alpha = sqrt(abs(sf)) / casc(c,r)%manning

	 if (sf <  0.0d0) then
	  hh = casc(cc,rr)%h
	  stordepth = casc(cc,rr)%sdep
     endif

	 if (hh > stordepth .and. sf /= 0.0d0) then
	  if (sf > 0.d0) a =  1.0d0
	  if (sf < 0.d0) a = -1.0d0
	  dqq = a * casc(c,r)%W * alpha * (hh - stordepth)**(5.0d0/3.0d0)

!=== Calculate change in flow/water volume in cells================= 
	  casc(c,r)%dqov = casc(c,r)%dqov - dqq
	  casc(cc,rr)%dqov = casc(cc,rr)%dqov + dqq
     endif
  endif
  enddo
  enddo
  enddo
	
end subroutine route_overlnd	 

   
   
   
