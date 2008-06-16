subroutine pfreadout(clm,drv)

  use drv_module          ! 1-D Land Model Driver variables
  !use dfport
  use precision
  use clmtype
  use clm_varpar, only : nlevsoi
  use clm_varcon, only : denh2o
  implicit none

  type (drvdec) ,intent(inout) :: drv
  type (clm1d), intent(inout) :: clm(drv%nch)	 !CLM 1-D Module
  real*8  dx2, dy2, dz2
  real*8 sat(drv%nc,drv%nr,parfl_nlevsoi),press(drv%nc,drv%nr,parfl_nlevsoi)
  real*8 dx, dy, dz, x1, y1, z1								
  integer*4 i,j,k, nni, nnj, nnk, ix, iy, iz,			&
            ns,  rx, ry, rz,nx,ny,nz, nnx, nny, nnz,    &
			is,dummy
  integer*4 ijk, namelength, xtent,ytent,ztent
  integer t, counter
      
  open(100,file="PF_output/pf_sat.pfb",form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='old') !@ binary saturation outputfile of Parflow
  open(101,file="PF_output/pf_press.pfb",form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='old') !@ binary saturation outputfile of Parflow                 
  open(200,file="pf_clm_sat.dat",status="unknown")   !@ ASCII outpufile produced by pfreadout for later reference
   
  ! Read in header info

! Start: reading of domain spatial information
  read(100) x1 !X
  read(100) y1 !Y 
  read(100) z1 !Z
  read(101) x1 !X
  read(101) y1 !Y 
  read(101) z1 !Z
!  print *,"1",x1,y1,z1

  read(100) nx !NX
  read(100) ny !NY
  read(100) nz !NZ
  read(101) nx !NX
  read(101) ny !NY
  read(101) nz !NZ
!  print *,"n",nx,ny,nz

  read(100) dx !DX
  read(100) dy !DY
  read(100) dz !DZ
  read(101) dx !DX
  read(101) dy !DY
  read(101) dz !DZ
!  print *,"d",dx,dy,dz

  dx2 = dx
  dy2 = dy
  dz2 = dz
  read(100) ns !num_subgrids
  read(101) ns !num_subgrids
!  print *,"ns",ns
! End: reading of domain spatial information

! Start: loop over number of sub grids
  do is = 0, (ns-1)

! Start: reading of sub-grid spatial information
   read(100) ix
   read(100) iy
   read(100) iz
   read(101) ix
   read(101) iy
   read(101) iz
!   print *,"i",ix,iy,iz
   
   read(100) nnx
   read(100) nny
   read(100) nnz
   read(101) nnx
   read(101) nny
   read(101) nnz
!   print *,"nn",nnx,nny,nnz

   read(100) rx
   read(100) ry
   read(100) rz
   read(101) rx
   read(101) ry
   read(101) rz
!   print *,"r",rx,ry,rz

! End: reading of sub-grid spatial information

! Start: read in saturation data from each individual subgrid
  do  k=iz +1 , iz + nnz
   do  j=iy +1 , iy + nny
    do  i=ix +1 , ix + nnx
     read(100) sat(i,j,k)
     read(101) press(i,j,k)
	 write(200,*)i,j,k,sat(i,j,k)
    end do
   end do
  end do
! End: read in saturation data from each individual subgrid

  end do
! End: loop over number of sub grids
print*, "+++++++++++++++ about to loop and copy sats back ino CLM ++++++++++++++"
! Start: assign saturation data from PF to tiles/layers of CLM
  t = 0
  do j=1,drv%nr     !rows (y)
   do i=1,drv%nc  !columns (x)
   	t = t + 1
   	counter = -1
	do k=1,parfl_nlevsoi
!	 write(200,*)i,j,k,sat(i,j,k)

!      if(clm(t)%planar_mask == 1.0d0) then
 !        counter  = counter + 1
	 clm(t)%pf_vol_liq(k) = sat(i,j,nz-k+1) * clm(t)%watsat(k)
	 clm(t)%pf_press(k) = press(i,j,nz-k+1) * 1000.d0
	 if (k <= nlevsoi) clm(t)%h2osoi_liq(k) = clm(t)%pf_vol_liq(k)*clm(t)%dz(k)*denh2o 
 !  	 if (clm(t)%topo_mask(k) == 1) clm(t)%h2osoi_liq(nlevsoi-counter) =  &
 !  	                                clm(t)%pf_vol_liq(k)*clm(t)%dz(1)*denh2o
 !     endif 

    enddo
     write(2010) (sat(i,j,nz-k+1),k=1,parfl_nlevsoi)
     write(2010) (press(i,j,nz-k+1),k=1,parfl_nlevsoi)   
   enddo
  enddo
    
  close(100)
  close(200)
  close(101)
  end subroutine pfreadout

 
