readpfb=function(filename, verbose){

# code to read parflow binary in R
# RMM 10-19-13

to.read = file(filename,"rb")
on.exit(close(to.read))
if(verbose){print(to.read)}
#read in X0, Y0, Z0 of domain
X = readBin(to.read,double(),endian="big")
Y = readBin(to.read,double(),endian="big")
Z = readBin(to.read,double(),endian="big")

if(verbose){
print(X)
print(Y)
print(Z)
}

#read in global nx, ny, nz of domain
nx = readBin(to.read,integer(),endian="big")
ny = readBin(to.read,integer(),endian="big")
nz = readBin(to.read,integer(),endian="big")

# set up a blank array the size of the domain
Data = array(0,dim=c(nx,ny,nz))

if(verbose){
print(nx)
print(ny)
print(nz)
}

#read in dx dy dz
dx = readBin(to.read,double(),endian="big")
dy = readBin(to.read,double(),endian="big")
dz = readBin(to.read,double(),endian="big")

if(verbose){
print(dx)
print(dy)
print(dz)
}
# read in number of subgrids
is = readBin(to.read,integer(),endian="big")

if(verbose){print(is)}
#loop over each subgrid to grab data

for (i in 1:is ) {
#read in local starting points ix, iy, iz of this subgrid
ix = readBin(to.read,integer(),endian="big")
iy = readBin(to.read,integer(),endian="big")
iz = readBin(to.read,integer(),endian="big")

if(verbose){
print(ix)
print(iy)
print(iz)
}

#read in locall nx, ny, nz of this subgrid
nnx = readBin(to.read,integer(),endian="big")
nny = readBin(to.read,integer(),endian="big")
nnz = readBin(to.read,integer(),endian="big")

if(verbose){
print(nnx)
print(nny)
print(nnz)
}

#read in local refinement of this subgridr
rx = readBin(to.read,integer(),endian="big")
ry = readBin(to.read,integer(),endian="big")
rz = readBin(to.read,integer(),endian="big")

if(verbose){
print(rx)
print(ry)
print(rz)
}

for (k in (iz+1):(iz+nnz))  {
   for (j in (iy+1):(iy+nny))  {	
	for (i in (ix+1):(ix+nnx))  {
	Data[i,j,k] = readBin(to.read,double(),endian="big")
    }
   }
  }
}
return(Data)
}