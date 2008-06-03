 module casctype
 use precision
 implicit none

 public casc2D
 
 type casc2D

!=== Raster Data =====================================================
  real(r8) :: elev
  real(r8) :: manning

!=== Hydrologic Info =================================================
  real(r8) :: sdep              ! storage depth (its definition and initialization must be verified!!!)
  real(r8) :: dqov              ! change of in/our flow of cell/tile at each time step
  real(r8) :: h				    ! flow depth in cell/tile
  real(r8) :: W                 ! cell dimensions (m)
  real(r8) :: hov               ! Updated flow depth
  real(r8) :: sovout            ! Slope of watershed overland oulet
  real(r8) :: qoutov            ! Overland outflow
  real(r8) :: sum_volout        ! Summed outflow
  real(r8) :: depth             ! Water volume/depth appplied to ground surface that is partitioned into surf and infl
  real(r8) :: tot_vol           ! Summed water volume/depth appplied to ground surface that is partitioned into surf and infl
  real(r8) :: infiltr           ! Infiltration rate for each CASC timestep (mm/s)
  real(r8) :: surf              ! Surface runoff rate for each CASC timestep (mm/s)
  real(r8) :: tot_infl,tot_surf ! Summed infiltration and surface depths over CASC timestep
  real(r8) :: qx,qy             ! Overland fluxes in the x- and y-direction
  real(r8) :: x_sl,y_sl         ! Topo slopes in the x-y-directions

 end type casc2d
 
 end module casctype