subroutine open_files (clm,drv)
 use clmtype             ! CLM tile variables
 use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
 use precision
 use drv_module          ! 1-D Land Model Driver variables
 implicit none

 type (clm1d) :: clm(nlevsoi)
 type (drvdec):: drv              

  
 open (199,file='balance.txt',status='unknown')
 write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"
 
 open(1995,file="qflx_top_soil.bin", form='binary')  ! @ 2D output file     
 open(1996,file="qflx_infl.bin", form='binary')  ! @ 2D output file     
 open(1997,file="qflx_evap_grnd.bin", form='binary')  ! @ 2D output file     
 open(1998,file="eflx_soil_grnd.bin", form='binary')  ! @ 2D output file     
 open(1999,file="qflx_evap_veg.bin", form='binary')  ! @ 2D output file     
 open(2000,file="soiliqu_2d.bin", form='binary')  ! @ 2D output file     
 open(2001,file="qflx_surf.bin", form='binary')  ! @ 2D output file     
 open(2002,file="qflx_evap_tot.bin", form='binary')   ! @ 2D output file     
 open(2003,file="t_grnd.bin", form='binary') ! @ 2D output file
 open(2004,file="qflx_evap_soi.bin", form='binary')  ! @ 2D output file:soil_liqu
 open(2005,file="qflx_tran_veg.bin", form='binary') ! @ overland outflow at watershed outlet 
 !open(2006,file="fl_depth.dat",form='formatted') ! @  2D output file: overland flow depth
 !open(2007,file="ck.dat",form='formatted') ! @  2D output file: overland flow depth
 open(2008,file="diagnostics.dat",form='formatted') ! @ diagnostics 1D averaged
 write(2008,'(A112)') "clm%istep topsurf topsoil surface evapor infiltr fraction error evapo_tot evap_veg evap_soi tran_veg ice_layer1"
 
 open(2010,file="sat_press.bin",form='binary') ! @ diagnostics 1D averaged
 !write(2010,'(A43)') "tstep nc nr L1 L2 L3 L4 L5 L6 L7 L8 L9 L10"
 
 end subroutine open_files	     