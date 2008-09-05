subroutine open_files (clm,drv,rank,ix,iy,ifstep)
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
 character*5 cistep      ! character for istep to include in file names
 integer :: nz,iz

 nz = 1.0d0
 iz = 1

 ! write processor rank and timestep to character for inclusion in FN
 write(RI,*) rank
 write(cistep,'(i5.5)') ifstep

 print*, "open files" 
 open (6,file='clm_elog.'//cistep//'.txt.'//trim(adjustl(RI)),status='unknown')

 open (199,file='balance.'//cistep//'.txt.'//trim(adjustl(RI)))
 write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"

 open(1995,file='qflx_top_soil.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1995)ix,iy,iz,drv%nc,drv%nr,nz

 open(1996,file='qflx_infl.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1996)ix,iy,iz,drv%nc,drv%nr,nz

 open(1997,file='qflx_evap_grnd.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1997)ix,iy,iz,drv%nc,drv%nr,nz

 open(1998,file='eflx_soil_grnd.'//cistep//'.bin.'//trim(adjustl(RI)),  access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1998)ix,iy,iz,drv%nc,drv%nr,nz

 open(1999,file='qflx_evap_veg.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1999)ix,iy,iz,drv%nc,drv%nr,nz

 open(2000,file='eflx_sh_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(2000)ix,iy,iz,drv%nc,drv%nr,nz

 open(2001,file='eflx_lh_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(2001)ix,iy,iz,drv%nc,drv%nr,nz

 open(2002,file='qflx_evap_tot.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)   ! @ 2D output file     
 write(2002)ix,iy,iz,drv%nc,drv%nr,nz

 open(2003,file='t_grnd.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file
 write(2003)ix,iy,iz,drv%nc,drv%nr,nz

 open(2004,file='qflx_evap_soi.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file
 write(2004)ix,iy,iz,drv%nc,drv%nr,nz

 open(2005,file='qflx_tran_veg.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
 write(2005)ix,iy,iz,drv%nc,drv%nr,nz

 open(2006,file='eflx_lwrad_out.'//cistep//'.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
 write(2006)ix,iy,iz,drv%nc,drv%nr,nz

 open(2008,file='diagnostics.'//cistep//'.dat.'//trim(adjustl(RI)),form='formatted') ! @ diagnostics 1D averaged
 write(2008,'(A112)') "clm%istep topsurf topsoil surface evapor infiltr fraction error evapo_tot evap_veg evap_soi tran_veg ice_layer1"
 
 end subroutine open_files	     
