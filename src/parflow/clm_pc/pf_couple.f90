subroutine pf_couple(drv,clm)

  use drv_module          ! 1-D Land Model Driver variables
  use dfport
  use precision
  use clmtype
  use clm_varpar, only : nlevsoi,parfl_nlevsoi
  use clm_varcon, only : denh2o, denice, istwet, istice
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module

  integer l,r,c,t
  integer width_step,I_err
  integer r_len, c_len, j_len
  character*100 ST,SR,SC,SJ
  real(r8) begwatb,endwatb !@ beginning and ending water balance over ENTIRE domain
  real(r8) tot_infl_mm,tot_tran_veg_mm,tot_drain_mm !@ total mm of h2o from infiltration and transpiration
  real(r8) error !@ mass balance error over entire domain
  real     pf_dt
  
!@ Variable declarations: write *.pfb file
  real*8 x1,y1,z1,dx,dy,dz,value,press
  integer*4 nx,ny,nz,nnx,nny,nnz,is,ns
  integer*4 rx,ry,rz,ix,iy,iz
  integer*4 d1,d2,d3
  integer*4 i,j,k,ni,nj,nk
!  integer inact(drv%nch) !@ counts how many cells starting from the bottom are inactive for each column
  integer*4 soil_lev(drv%nch)  ! counts up from the bottom active soil layer
  
! End of variable declaration 

  open (299,file='tstep.tcl',blank='NULL')
  open (300,file='PF_output/source.pfb',form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='unknown')
  open (301,file='PF_output/in_cond.pfb',form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='unknown') 

  Write(*,*)"========== start the loop over the flux ============="
  
! also: arbitrary cutoff value for evap rate (if rate is too small problems with Parflow solver)
   t = 0
  do r=1,drv%nr     !rows (y)
    do c=1,drv%nc  !columns (x)
	  t = t + 1
      do l = 1, nlevsoi
	    if (l == 1) then
		 clm(t)%pf_flux(l)=(-clm(t)%qflx_tran_veg*clm(t)%rootfr(l)) + clm(t)%qflx_infl
		else  
         clm(t)%pf_flux(l) = - clm(t)%qflx_tran_veg*clm(t)%rootfr(l)
		endif
      enddo
	enddo
  end do

! At the first time step drv%sat_flag_o is equal 0
!  if (clm(1)%istep == 1) drv%sat_flag_o = 0
  
!@ Start: Write misc and timestep information  
  write(299,'(a)')"##set tcl_precision 17"           !@ defines tcl precision
  write(299,'(a,i)')"set sat_flag",drv%sat_flag    !@ this flag defines saturation status of domain
                                                   !@ and is set in runoff_infl.f90.
   
  write(ST,fmt="(i100)") clm(1)%istep
  width_step = 101 - len(trim(adjustl(ST)))
  write(299,"('set ts(',a,') ',f15.4)") ST(width_step:), clm(1)%dtime*dble(clm(1)%istep)
 
  write(299,"('set num_ts ',i7)") clm(1)%istep
  write(ST,fmt="(i100)") clm(1)%istep-1
  if (clm(1)%istep-1 > 0) then
   width_step =101 - len(trim(adjustl(ST)))
  else
   width_step = 100
  endif
  write(299,"('set ts(',a,') ',f15.4)") ST(width_step:), clm(1)%dtime*dble(clm(1)%istep-1)
  
  if (maxval(clm%pond_flag) == 0) then
    write(299,"(a)")"pfset TimeStep.Type                Constant"
    write(299,"(a)")"pfset TimeStep.Value               0.125"
  elseif(maxval(clm%pond_flag) == 1) then
!    pf_dt = 1.0e-7 * 0.2 / maxval(clm%qflx_infl)
    write(299,"(a)")"pfset TimeStep.Type                Constant"
!    write(299,"(a,f)")"pfset TimeStep.InitialStep ", pf_dt
    write(299,"(a)")"pfset TimeStep.Value               0.125"
  endif   

!@ End: Write misc and timestep information

!@ Start: Write distributed source file, source.pfb.
  nx=drv%nc
  ny=drv%nr
  nz=parfl_nlevsoi

  nnx=nx
  nny=ny
  nnz=parfl_nlevsoi

  x1=dfloat(nx)*drv%dx
  y1=dfloat(ny)*drv%dy
  z1=dfloat(nz)*drv%dz

  ix=0
  iy=0
  iz=0
 
  rx=1
  ry=1
  rz=1

  dx=drv%dx
  dy=drv%dy
  dz=drv%dz

  ns=1

  write(300) x1 !X
  write(300) y1 !Y 
  write(300) z1 !Z
  write(301) x1 !X
  write(301) y1 !Y 
  write(301) z1 !Z

  write(300) nx !NX
  write(300) ny !NY
  write(300) nz !NZ 
  write(301) nx !NX
  write(301) ny !NY
  write(301) nz !NZ 

  write(300) dx !DX
  write(300) dy !DY
  write(300) dz !DZ
  write(301) dx !DX
  write(301) dy !DY
  write(301) dz !DZ

  write(300) ns !num_subgrids
  write(301) ns !num_subgrids

!loop over number of sub grids
  do is = 0, (ns-1)
   write(300) ix
   write(300) iy
   write(300) iz 
   write(301) ix
   write(301) iy
   write(301) iz 
   
   write(300) nnx
   write(300) nny
   write(300) nnz
   write(301) nnx
   write(301) nny
   write(301) nnz

   write(300) rx
   write(300) ry
   write(300) rz
   write(301) rx
   write(301) ry
   write(301) rz

!open (1919, file='flux.check.txt')

   do  k=iz, iz + nnz - 1
   t = 0
    do  j=iy, iy + nny - 1
     do  i=ix, ix + nnx - 1
      t = t + 1
      if ((parfl_nlevsoi-k)>nlevsoi) then
       value = 0.0d0                              
       write(300) value
      elseif((parfl_nlevsoi-k)<=nlevsoi) then 
       value = clm(t)%pf_flux(parfl_nlevsoi-k)*86.4d0/dz 
       write(300) value
	  else
       !print *,"Trouble"
       stop 
      endif 
      
      press = clm(t)%pf_press(parfl_nlevsoi-k)/1000.0d0
      write(301)press 
      
     end do
    end do
   end do
   
  end do
  close(300)
  close(301)
!@ End: Write distributed source file, source.pfb.

!@ Run paflow for equivalent timestep
  print*, ' calling PF ts:',clm(1)%istep,' time:',clm(1)%dtime*dble(clm(1)%istep)
  I_err = SYSTEM("tclsh pfstep.tcl")
!@ Read Saturations calculated with Parflow
  write(*,*)"============ Start the readout ================"
  call pfreadout(clm,drv)
  write(*,*)"============ End the readout ================="
!@ Here the couple between CLM and Parflow is effectively completed  

!@ Start: Here we do the mass balance: We look at every tile/cell individually!
!@ Determine volumetric soil water
  begwatb = 0.0d0
  endwatb = 0.0d0
  tot_infl_mm = 0.0d0
  tot_tran_veg_mm = 0.0d0
  tot_drain_mm = 0.0d0
  t = 0

    !@ Start: Loop over domain  
  do r=1,drv%nr     !rows (y)
    do c=1,drv%nc  !columns (x)
	  t = t + 1
      do l = 1, nlevsoi
        clm(t)%h2osoi_vol(l) = clm(t)%h2osoi_liq(l)/(clm(t)%dz(l)*denh2o) &
                             + clm(t)%h2osoi_ice(l)/(clm(t)%dz(l)*denice)
	  enddo
      
    !@ Let's do it my way
    !@ Here we add the total water mass of the layers below CLM soil layers from Parflow to close water balance
    !@ We can use clm(1)%dz(1) because the grids are equidistant and congruent
      clm(t)%endwb=0.0d0 !@only interested in wb below surface
      do l = 1, parfl_nlevsoi
        clm(t)%endwb = clm(t)%endwb + clm(t)%pf_vol_liq(l) * clm(1)%dz(1) * 1000.0d0
        clm(t)%endwb = clm(t)%endwb + clm(t)%pf_vol_liq(l)/clm(t)%watsat(l) * 0.0001*0.2d0 * clm(t)%pf_press(l)    
      enddo
      
    !@ Water balance over the entire domain
       begwatb = begwatb + clm(t)%begwb
       endwatb = endwatb + clm(t)%endwb
       tot_infl_mm = tot_infl_mm + clm(t)%qflx_infl * clm(1)%dtime
       tot_tran_veg_mm = tot_tran_veg_mm + clm(t)%qflx_tran_veg * clm(1)%dtime
 !      tot_drain_mm = tot_drain_mm + (0.0061/86.4d0) * clm(1)%dtime

   ! Determine wetland and land ice hydrology (must be placed here since need snow 
   ! updated from clm_combin) and ending water balance
   !@ Does my new way of doing the wb influence this?! 05/26/2004

        if (clm(t)%itypwat==istwet .or. clm(t)%itypwat==istice) call clm_hydro_wetice (clm(t))

! -----------------------------------------------------------------
! Energy AND Water balance for lake points
! -----------------------------------------------------------------
       
        if (clm(t)%lakpoi) then    
!        call clm_lake (clm)             @Stefan: This subroutine is still called from clm_main; why? 05/26/2004
         do l = 1, nlevsoi
		  clm(t)%h2osoi_vol(l) = 1.0
         enddo  
        endif

! -----------------------------------------------------------------
! Update the snow age
! -----------------------------------------------------------------

!        call clm_snowage (clm)           @Stefan: This subroutine is still called from clm_main

! -----------------------------------------------------------------
! Check the energy and water balance
! -----------------------------------------------------------------
 
        call clm_balchk (clm(t), clm(t)%istep) !@ Stefan: in terms of wb, this call is obsolete;
                                               !@ energy balances are still calculated

    enddo
  enddo
    !@ End: Loop over domain  

  
  error = 0.0d0
  error = endwatb - begwatb - (tot_infl_mm - tot_tran_veg_mm) ! + tot_drain_mm
   
  write(199,'(1i,1x,5(f,1x))') clm(1)%istep,error,tot_infl_mm,tot_tran_veg_mm,begwatb,endwatb
  print *,""
  print *,"Error (mm):",error
!@ End: mass balance  
  
!@ Pass sat_flag to sat_flag_o
! drv%sat_flag_o = drv%sat_flag

close(299)
end subroutine pf_couple

