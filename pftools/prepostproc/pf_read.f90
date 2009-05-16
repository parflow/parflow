  program pf_read
  implicit none
  real*8  dx2, dy2, dz2
  real*8,allocatable :: sat(:,:,:)
  real*8 ri, rj, rk1, rk2, headsum, rsum, junk,  &
         ksum, kavg,f, dx, dy, dz, x1, y1, z1								
  integer*4 i,j,k, nni, nnj, nnk, ix, iy, iz,			&
            ns,  rx, ry, rz,nx,ny,nz, nnx, nny, nnz,    &
			is,dummy
  integer*4 ijk, namelength, xtent,ytent,ztent
  integer t,counter
  character*100 fname
  
  write(*,*)"File Name:"
  read(*,'(a)')fname
  
  open(100,file=trim(adjustl(fname)),form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='old')         !binary outputfile of Parflow
  open(200,file="test.dat",status="unknown")
   
  ! Read in header info

! Start: reading of domain spatial information
  read(100) x1 !X
print *,"x1",x1
  read(100) y1 !Y 
print *,"y1",y1
  read(100) z1 !Z
print *,"z1",z1
print *,""

  read(100) nx !NX
print *,"nx",nx
  read(100) ny !NY
print *,"ny",ny
  read(100) nz !NZ
print *,"nz",nz
print *,""
  allocate (sat(nx,ny,nz))

  read(100) dx !DX
print *,"dx",dx
  read(100) dy !DY
print *,"dy",dy
  read(100) dz !DZ
print *,"dz",dz
print *,""

  dx2 = dx
  dy2 = dy
  dz2 = dz
  read(100) ns !num_subgrids
  print *,"ns",ns
print *,""
! End: reading of domain spatial information

! Start: loop over number of sub grids
  do is = 0, (ns-1)
print *,"Doing subgrid",is

! Start: reading of sub-grid spatial information
   read(100) ix
print *,"ix",ix
   read(100) iy
print *,"iy",iy
   read(100) iz
print *,"iz",iz
print *,""   

   read(100) nnx
print *,"nnx",nnx 
  read(100) nny
print *,"nny",nny 
   read(100) nnz
print *,"nnz",nnz 
print *,""
   read(100) rx
   read(100) ry
   read(100) rz
print *,"r",rx,ry,rz

! End: reading of sub-grid spatial information

! Start: read in saturation data from each individual subgrid
  do  k=iz +1 , iz + nnz
   do  j=iy +1 , iy + nny
    do  i=ix +1 , ix + nnx
     read(100) sat(i,j,k)
    end do
   end do
  end do
! End: read in saturation data from each individual subgrid

! End: read in saturation data from each individual subgrid

  end do
! End: loop over number of sub grids

!Proceed with writing vtk file


counter=0  
  do  k=1,nnz
   do  j=1,ny
    do  i=1,nx
	  write(200,*)i,j,k,sat(i,j,k)
     if (sat(i,j,k)==0)counter=counter+1

    end do
   end do
  end do
print *,"Counter:",counter
  close(100)
  close(200)

  end
