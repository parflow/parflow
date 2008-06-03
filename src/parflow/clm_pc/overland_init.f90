subroutine overland_init(casc,clm,drv)
 use drv_module          ! 1-D Land Model Driver variables
 use precision
 use clmtype
 use casctype
 implicit none

 type (drvdec):: drv
 type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
 type (casc2D):: casc(drv%nc,drv%nr)

 integer :: r,c


!=== Initialize all overland flow params ============= 
 
  do r = 1, drv%nr
   do c = 1, drv%nc
	 casc(c,r)%qx = 0.
     casc(c,r)%qy = 0.
     casc(c,r)%h = 0.
   enddo
  enddo


end subroutine overland_init




