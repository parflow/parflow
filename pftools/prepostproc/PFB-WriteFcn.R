writepfb=function(to.write,filename,dx,dy,dz){
# This is a function to take array inputs and write them as pfbs
# to.write - is the array to be written. This must be a 3D array with dimensions nx, ny, nz
#  values should be input to the array such that [1,1,1] is the lower left corner of the domain and [nx,ny,nz] is the upper right
# file name - is the name for the file to be written out
# dx,dy and dz are the grid cell sizes
# the function can be called like: writepfb(to.write=arraytowrite,filename="pfbtest.pfb",dx=dx,dy=dy,dz=dz)

  fp=file(filename,"wb") 
  
  #write in X0, Y0, Z0 of domain
  X2=as.vector(0)
  Y2=as.vector(0)
  Z2=as.vector(0)
  writeBin(X2,fp,double(),endian="big")
  writeBin(Y2,fp,double(),endian="big")
  writeBin(Z2,fp,double(),endian="big")
  
  #write in global nx,ny,nz of domain
  nx=dim(to.write)[1]
  ny=dim(to.write)[2]
  nz=dim(to.write)[3]
  if(is.na(nz)){nz=as.integer(1)}
  nx2=as.vector(nx)
  ny2=as.vector(ny)
  nz2=as.vector(nz)
  writeBin(nx2,fp,integer(),endian="big")
  writeBin(ny2,fp,integer(),endian="big")
  writeBin(nz2,fp,integer(),endian="big")
  
  #write the dx,dy,dz
  dx2=as.vector(dx)
  dy2=as.vector(dy)
  dz2=as.vector(dz)
  writeBin(dx2,fp,double(),endian="big")
  writeBin(dy2,fp,double(),endian="big")
  writeBin(dz2,fp,double(),endian="big")
  
  ##write for local grids if there is any
  #number of subgrid
  is=as.vector(as.integer(1))
  writeBin(is,fp,integer(),endian="big")
  
  #local starting point ix,iy, iz
  ix=as.vector(as.integer(0))
  iy=as.vector(as.integer(0))
  iz=as.vector(as.integer(0))
  writeBin(ix,fp,integer(),endian="big")
  writeBin(iy,fp,integer(),endian="big")
  writeBin(iz,fp,integer(),endian="big")
  
  #local nnx,nny,nnz
  nnx=nx2
  nny=ny2
  nnz=nz2
  writeBin(nnx,fp,integer(),endian="big")
  writeBin(nny,fp,integer(),endian="big")
  writeBin(nnz,fp,integer(),endian="big")
  
  #local refinement
  rx=as.vector(as.integer(1))
  ry=as.vector(as.integer(1))
  rz=as.vector(as.integer(1))
  writeBin(rx,fp,integer(),endian="big")
  writeBin(ry,fp,integer(),endian="big")
  writeBin(rz,fp,integer(),endian="big")
  
  # aa=as.vector(as.double(a))
  aa=as.vector(to.write)
  writeBin(aa,fp,double(),endian="big")
  
  close(fp)
}

