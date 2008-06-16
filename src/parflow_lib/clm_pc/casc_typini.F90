 subroutine casc_typini(casc,drv)
 use drv_module          ! 1-D Land Model Driver variables
 use precision
 use casctype
 implicit none

 type (drvdec):: drv
 type (casc2D):: casc(drv%nc,drv%nr)

!=== Raster Data =====================================================
 integer :: c,r
 
  
  do r = 1, drv%nr
   do c = 1,drv%nc

!=== Hydrologic Info =================================================
    casc(c,r)%dqov       = 0.0d0              ! change of in/our flow of cell/tile at each time step
    casc(c,r)%h          = 0.0d0	   ! flow depth in cell/tile
    casc(c,r)%hov      = 0.0d0             ! Old flow depth
    casc(c,r)%qoutov     = 0.0d0           ! Overland outflow
    casc(c,r)%sum_volout = 0.0d0        ! Summed outflow
    casc(c,r)%depth      = 0.0d0            ! Water volume/depth appplied to ground surface that is partitioned into surf and infl
    casc(c,r)%tot_vol    = 0.0d0          ! Summed water volume/depth appplied to ground surface that is partitioned into surf and infl
    casc(c,r)%infiltr    = 0.0d0          ! Infiltration rate for each CASC timestep (mm/s)
    casc(c,r)%surf       = 0.0d0             ! Surface runoff rate for each CASC timestep (mm/s)
    casc(c,r)%tot_infl   = 0.0d0
    casc(c,r)%tot_surf   = 0.0d0         ! Summed infiltration and surface depths over CASC timestep
    casc(c,r)%qx         = 0.0d0
    casc(c,r)%qy         = 0.0d0             ! Overland fluxes in the x- and y-direction
   
   enddo
  enddo  

 end subroutine casc_typini