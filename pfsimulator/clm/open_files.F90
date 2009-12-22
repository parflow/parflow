subroutine open_files (clm,drv,rank,ix,iy,ifstep,clm_output_dir,clm_output_dir_length, clm_bin_out_dir)
  use clmtype             ! CLM tile variables
  use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use parflow_config
  implicit none

  type (clm1d) :: clm(nlevsoi)
  type (drvdec):: drv              
  integer :: rank,ix,iy,ifstep  ! RMM added, to include time step for file name
  character*100 RI
  integer :: clm_output_dir_length  ! length of above character string, used in C-Fortran interop
  character (LEN=clm_output_dir_length) :: clm_output_dir  ! character location subdir for all output
  integer :: clm_bin_out_dir       ! boolean if clm binary output goes in dir for each file
  character*5 cistep      ! character for istep to include in file names
  integer :: nz,iz

  nz = 1
  iz = 1

  ! write processor rank and timestep to character for inclusion in FN
  write(RI,*) rank
  write(cistep,'(i5.5)') ifstep

!  print*, "open files" 
!  print*, clm_output_dir
!  print*, clm_bin_out_dir
  ! open (166,file=clm_output_dir//'clm_elog.'//cistep//'.txt.'//trim(adjustl(RI)))

  ! open (199,file=clm_output_dir//'balance.'//cistep//'.txt.'//trim(adjustl(RI)))
  ! write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"

  !print *, "balance file"
  if (clm_bin_out_dir == 0) then
     open(1995,file=clm_output_dir//'qflx_top_soil.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1995) ix,iy,iz,drv%nc,drv%nr,nz

     open(1996,file=clm_output_dir//'qflx_infl.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1996) ix,iy,iz,drv%nc,drv%nr,nz

     open(1997,file=clm_output_dir//'qflx_evap_grnd.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1997) ix,iy,iz,drv%nc,drv%nr,nz

     open(1998,file=clm_output_dir//'eflx_soil_grnd.'//cistep//'.bin.'//trim(adjustl(RI)),  access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1998) ix,iy,iz,drv%nc,drv%nr,nz

     open(1999,file=clm_output_dir//'qflx_evap_veg.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1999) ix,iy,iz,drv%nc,drv%nr,nz

     open(2000,file=clm_output_dir//'eflx_sh_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(2000) ix,iy,iz,drv%nc,drv%nr,nz

     open(2001,file=clm_output_dir//'eflx_lh_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(2001) ix,iy,iz,drv%nc,drv%nr,nz

     open(2002,file=clm_output_dir//'qflx_evap_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)   ! @ 2D output file     
     write(2002) ix,iy,iz,drv%nc,drv%nr,nz

     open(2003,file=clm_output_dir//'t_grnd.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file
     write(2003) ix,iy,iz,drv%nc,drv%nr,nz

     open(2004,file=clm_output_dir//'qflx_evap_soi.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file
     write(2004) ix,iy,iz,drv%nc,drv%nr,nz

     open(2005,file=clm_output_dir//'qflx_tran_veg.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2005) ix,iy,iz,drv%nc,drv%nr,nz

     open(2006,file=clm_output_dir//'eflx_lwrad_out.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2006) ix,iy,iz,drv%nc,drv%nr,nz

     open(2007,file=clm_output_dir//'swe_out.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2007) ix,iy,iz,drv%nc,drv%nr,nz

     open(2009,file=clm_output_dir//'tsoil.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2009) ix,iy,iz,drv%nc,drv%nr,nz

     open(2008,file=clm_output_dir//'diagnostics.'//cistep//'.dat.'//trim(adjustl(RI)),form='formatted') ! @ diagnostics 1D averaged

     write(2008,'(A112)') &
          "clm%istep topsurf topsoil surface evapor infiltr fraction error evapo_tot evap_veg evap_soi tran_veg ice_layer1"

  else
     open(1995,file=clm_output_dir//'qflx_top_soil/qflx_top_soil.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1995) ix,iy,iz,drv%nc,drv%nr,nz

     open(1996,file=clm_output_dir//'qflx_infl/qflx_infl.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1996) ix,iy,iz,drv%nc,drv%nr,nz

     open(1997,file=clm_output_dir//'qflx_evap_grnd/qflx_evap_grnd.'//cistep//'.bin.'//trim(adjustl(RI)), &
access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1997) ix,iy,iz,drv%nc,drv%nr,nz

     open(1998,file=clm_output_dir//'eflx_soil_grnd/eflx_soil_grnd.'//cistep//'.bin.'//trim(adjustl(RI)),  access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1998) ix,iy,iz,drv%nc,drv%nr,nz

     open(1999,file=clm_output_dir//'qflx_evap_veg/qflx_evap_veg.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(1999) ix,iy,iz,drv%nc,drv%nr,nz

     open(2000,file=clm_output_dir//'eflx_sh_tot/eflx_sh_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(2000) ix,iy,iz,drv%nc,drv%nr,nz

     open(2001,file=clm_output_dir//'eflx_lh_tot/eflx_lh_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
     write(2001) ix,iy,iz,drv%nc,drv%nr,nz

     open(2002,file=clm_output_dir//'qflx_evap_tot/qflx_evap_tot.'//cistep//'.bin.'//trim(adjustl(RI)), &
          access=ACCESS, form=FORM)   ! @ 2D output file     
     write(2002) ix,iy,iz,drv%nc,drv%nr,nz

     open(2003,file=clm_output_dir//'t_grnd/t_grnd.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file
     write(2003) ix,iy,iz,drv%nc,drv%nr,nz

     open(2004,file=clm_output_dir//'qflx_evap_soi/qflx_evap_soi.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file
     write(2004) ix,iy,iz,drv%nc,drv%nr,nz

     open(2005,file=clm_output_dir//'qflx_tran_veg/qflx_tran_veg.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2005) ix,iy,iz,drv%nc,drv%nr,nz

     open(2006,file=clm_output_dir//'eflx_lwrad_out/eflx_lwrad_out.'//cistep//'.bin.'//trim(adjustl(RI)), &
          access=ACCESS, form=FORM) ! @ 2D output file 
     write(2006) ix,iy,iz,drv%nc,drv%nr,nz

     open(2007,file=clm_output_dir//'swe_out/swe_out.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2007) ix,iy,iz,drv%nc,drv%nr,nz

     open(2009,file=clm_output_dir//'t_grnd/tsoil.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
     write(2009) ix,iy,iz,drv%nc,drv%nr,nz

     open(2008,file=clm_output_dir//'diag_out/diagnostics.'//cistep//'.dat.'//trim(adjustl(RI)),form='formatted') ! @ diagnostics 1D averaged
     write(2008,'(A112)') &
          "clm%istep topsurf topsoil surface evapor infiltr fraction error evapo_tot evap_veg evap_soi tran_veg ice_layer1"

  end if

end subroutine open_files
