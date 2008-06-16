subroutine runoff_infl(clm,drv)

  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  use clm_varpar, only : parfl_nlevsoi
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module

  integer r,c,i,t,ioval,nlogged
  integer active !@ counter for number of active cells (cells with sat(i,j,1)<1.0d)
  real(r8) storage,tot_storage,water,storage_c,applied_h2o(drv%nch)
  real(r8) adj_infl !@ infil rate adjusted for available subsurface storage
  real(r8) ex_rate !@ exchange rate between the surface and subsurface
  real(r8) lamda !@ exchange proportionallity constant
  real(r8) dummy,outflow,sum_out
  character*8 name
  
  open (6000,file="./PF_output/pf_module.out.log",status='old')
  ioval = 0
  sum_out = 0.0
  outflow = 0.0
  do while (ioval==0)
   read(6000,'(A8)',iostat=ioval) name
   if (name=='Overland') then
    read(6000,*) nlogged
    print *,"nlogged",nlogged
    do i=1,nlogged
      read(6000,*)dummy,dummy,dummy,outflow
      sum_out = sum_out + outflow
    enddo
    sum_out = sum_out / (nlogged-1)
    outflow = sum_out /drv%dy
    print *,"OUTFLOW",outflow
   endif
   enddo
   close(6000) 
 
  !@ Calculate available storage for the entire domain and calc applied amount of water for no-ponding cells
  storage = 0.0d0
  tot_storage = 0.0d0
  water = 0.0d0
  t = 0
  do r = 1, drv%nr
   do c = 1, drv%nc
   t = t + 1
    do i = 1, parfl_nlevsoi
     storage = storage + clm(t)%pf_vol_liq(i) * clm(t)%dz(1) * 1000.0d0
     tot_storage = tot_storage + clm(t)%watsat(i)*clm(t)%dz(1) * 1000.0d0
    enddo
     water = water + applied_h2o(t) * clm(1)%dtime
     storage = storage + clm(1)%dtime*dabs(clm(t)%qflx_tran_veg)
   enddo
  enddo
  storage = tot_storage - storage
  print *,""
  print *,"water, storage, %",water,storage,(storage/tot_storage*100)
  
  ! Assign evap/infil rates 
  t = 0
   do r = 1, drv%nr
    do c = 1, drv%nc
     t = t + 1
      clm(t)%qflx_surf = outflow
      clm(t)%qflx_infl = clm(t)%qflx_top_soil - dabs(clm(t)%qflx_evap_grnd)
    enddo
   enddo     
  
  end subroutine runoff_infl
