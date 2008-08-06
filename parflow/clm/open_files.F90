subroutine open_files (clm,drv,rank,ix,iy)
 use clmtype             ! CLM tile variables
 use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
 use precision
 use drv_module          ! 1-D Land Model Driver variables
 use parflow_config
 implicit none

 type (clm1d) :: clm(nlevsoi)
 type (drvdec):: drv              
 integer :: rank,ix,iy
 character*100 RI
 integer :: nz,iz

 nz = 1.0d0
 iz = 1

 write(RI,*) rank
 print*, "open files" 
 open (6,file='clm_elog.txt.'//trim(adjustl(RI)),status='unknown')

 open (199,file='balance.txt.'//trim(adjustl(RI)))
 write(199,'(a59)') "istep error(%) tot_infl_mm tot_tran_veg_mm begwatb endwatb"

 open(1995,file='qflx_top_soil.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1995)ix,iy,iz,drv%nc,drv%nr,nz

 open(1996,file='qflx_infl.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1996)ix,iy,iz,drv%nc,drv%nr,nz

 open(1997,file='qflx_evap_grnd.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1997)ix,iy,iz,drv%nc,drv%nr,nz

 open(1998,file='eflx_soil_grnd.bin.'//trim(adjustl(RI)),  access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1998)ix,iy,iz,drv%nc,drv%nr,nz

 open(1999,file='qflx_evap_veg.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(1999)ix,iy,iz,drv%nc,drv%nr,nz

 open(2000,file='eflx_sh_tot.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(2000)ix,iy,iz,drv%nc,drv%nr,nz

 open(2001,file='eflx_lh_tot.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file     
 write(2001)ix,iy,iz,drv%nc,drv%nr,nz

 open(2002,file='qflx_evap_tot.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)   ! @ 2D output file     
 write(2002)ix,iy,iz,drv%nc,drv%nr,nz

 open(2003,file='t_grnd.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file
 write(2003)ix,iy,iz,drv%nc,drv%nr,nz

 open(2004,file='qflx_evap_soi.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM)  ! @ 2D output file
 write(2004)ix,iy,iz,drv%nc,drv%nr,nz

 open(2005,file='qflx_tran_veg.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
 write(2005)ix,iy,iz,drv%nc,drv%nr,nz

 open(2006,file='eflx_lwrad_out.bin.'//trim(adjustl(RI)), access=ACCESS, form=FORM) ! @ 2D output file 
 write(2006)ix,iy,iz,drv%nc,drv%nr,nz

 open(2008,file='diagnostics.dat.'//trim(adjustl(RI)),form='formatted') ! @ diagnostics 1D averaged
 write(2008,'(A112)') "clm%istep topsurf topsoil surface evapor infiltr fraction error evapo_tot evap_veg evap_soi tran_veg ice_layer1"
 
 end subroutine open_files	     
