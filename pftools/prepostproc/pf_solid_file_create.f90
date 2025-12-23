program pf_sol_gen

!
!  written by Reed Maxwell (maxwell5@llnl.gov)
!  10-30-04
!
real, allocatable:: dem(:,:), points(:,:),bottom_elev(:,:)
integer, allocatable:: triangles(:,:), patches(:,:), num_tri_patches(:)

real datum, x0, y0, dx, dy, xmax, ymax,x
integer nx_dem, ny_dem, num_points, num_tri, num_patches, i, j, k, ii, jj, kk, &
num_solid
character*40 dem_filename, pfsol_filename

! code creates a triangulated pf solid file
! it reads in a DEM file for the shape of the top surface, but assumes that
! the rest of the domain is rectangularly shaped (and is the same size as the
! extents of the DEM)
!
!NOTES:
! we assume the spatial discretization is the SAME for bottom and top of domain
! we assume rectangular sides
! we assume one (1) solid
! we assume six (6) patches
!
! input block, should be read in and made more general

nx_dem = 128   ! num dem pnts in x
ny_dem = 88    ! num dem pnts in y
x0 = 0.0       ! origin coord, y
y0 = 0.0       ! origin coord, x
dx = 1000.0    ! dx
dy = 1000.0    ! dy
xmax = float(nx_dem)*dx + x0
ymax = float(ny_dem)*dy + y0

!
num_solid = 1
! calculate total number of points
num_points =  2*nx_dem*ny_dem 
! currently we are gridding 2 triangles per box of four nodes, top and bottom
num_triangles = 4*(nx_dem-1)*(ny_dem-1)  + 4*(nx_dem-1) + 4*(ny_dem-1)
! currently hard wired for 6 patches
! we also allocate the max size for patches- num_triangles
num_patches = 6

allocate(DEM(nx_dem,ny_dem),points(num_points,3),triangles(num_triangles,3),  &
patches(num_patches,num_triangles),num_tri_patches(num_triangles),bottom_elev(nx_dem,ny_dem) )

bottom_elev = 250.  ! elevation of the bottom of the domain, init to 0.
dem = 0.            ! elevation of the top of the domain, init to 0.

dem_filename = 'corr.lw.1km.dem.txt'

!pfsol_filename = 'lw.1km.big.pfsol'
pfsol_filename = 'lw.1km.big.flatb.pfsol'

!Part for Little Washita watershed
open(10, file = trim(dem_filename))
read(10,*)
do j = 1, ny_dem
do i =1, nx_dem
read(10,*) dem(i,j)
!bottom_elev(i,j) = dem(i,j) - 20.

end do
end do
close(10)

!constant bottom, dem can be read from file
!open(10,file=trim(dem_filename))
!write(10,*)nx_dem,ny_dem
!do j = 1, ny_dem
!do i =1, nx_dem
! x = float(i-1)*dx+dx/2.0
! y = float(j-1)*dy+dy/2.0
! dem(i,j) = 15.24*sin(0.004581*x)+0.05*x+305. !-2.*cos(0.25*y)+0.001*y
! write(10,*) dem(i,j)
!end do
!end do
!close(10)

! loop over points and assign, jj = point counter
! first the DEM/upper surface
! we assume points are in the center of the digital elevation tile 

jj = 1
do i = 1, nx_dem
do j = 1, ny_dem
points(jj,1) = float(i-1)*dx + dx/2. + x0
points(jj,2) = float(j-1)*dy + dy/2. + y0
points(jj,3) = dem(i,j)
jj = jj + 1
end do
end do

! then the lower surface; currently a flat surface - i.e. datum
do i = 1, nx_dem
do j = 1, ny_dem
points(jj,1) = float(i-1)*dx + dx/2. + x0
points(jj,2) = float(j-1)*dy + dy/2. + y0
points(jj,3) = bottom_elev(i,j)
jj = jj + 1
end do
end do

print*, 'points: ',jj - 1, num_points

! loop over triangles and assign, jj = triangle counter; we also do the patches at the same time
! note: point looping starts at zero but we will start at 1, then sub later
! first the DEM/upper surface
! notes:
! patch(1,:) = upper surface (z = zmax)
! patch(2,:) = lower surface (z = z0)
! patch(3,:) = x = x0
! patch(4,:) = x = xmax
! patch(5,:) = y = y0
! patch(6,:) = y = ymax
! jj = triangle counter
! ii = patch counter; ii restarts w/ every face
!
! NOTE: the order in which the triangle points vertices are specified is critical -
! they must be specified so the normal vector points out of the domain always (ie RHR
!  counterclockwise for the top face, etc)
!
jj = 1
ii = 1
do i = 1, (nx_dem-1)
do j = 1, (ny_dem-1)
triangles(jj,1) = (i-1)*ny_dem + j 
triangles(jj,2) = (i)*ny_dem + j  + 1
triangles(jj,3) = (i-1)*ny_dem + j  +1
patches(1,ii) = jj
jj = jj + 1
ii = ii + 1
triangles(jj,1) = (i-1)*ny_dem + j 
triangles(jj,2) = (i)*ny_dem + j 
triangles(jj,3) = (i)*ny_dem + j +1
patches(1,ii) = jj
jj = jj + 1
ii = ii + 1
end do
end do

num_tri_patches(1) = ii - 1
ii = 1

! then the lower surface; 
do i = 1, (nx_dem-1)
do j = 1, (ny_dem-1)
triangles(jj,1) = (i-1)*ny_dem + j + nx_dem*ny_dem
triangles(jj,2) = (i-1)*ny_dem + j  +1 + nx_dem*ny_dem
triangles(jj,3) = (i)*ny_dem + j + 1 + nx_dem*ny_dem
patches(2,ii) = jj
jj = jj + 1
ii = ii + 1

triangles(jj,1) = (i-1)*ny_dem + j + nx_dem*ny_dem
triangles(jj,2) = (i)*ny_dem + j +1 + nx_dem*ny_dem
triangles(jj,3) = (i)*ny_dem + j + nx_dem*ny_dem
patches(2,ii) = jj
jj = jj + 1
ii = ii + 1

end do
end do

num_tri_patches(2) = ii - 1
ii = 1

! then x=x0 face; 
do j = 1, (ny_dem-1)
triangles(jj,1) =  j + nx_dem*ny_dem
triangles(jj,2) =  j 
triangles(jj,3) =  j +1
patches(3,ii) = jj
jj = jj + 1
ii = ii + 1

triangles(jj,1) =  j + nx_dem*ny_dem
triangles(jj,2) =  j + 1 
triangles(jj,3) =  j + 1 + nx_dem*ny_dem
patches(3,ii) = jj
jj = jj + 1
ii = ii + 1
end do

num_tri_patches(3) = ii - 1
ii = 1

! now the x=xmax face; 
do j = 1, (ny_dem-1)
triangles(jj,1) =  (nx_dem-1)*ny_dem + j + nx_dem*ny_dem
triangles(jj,2) =  (nx_dem-1)*ny_dem + j  +1
triangles(jj,3) =  j + (nx_dem-1)*ny_dem 
patches(4,ii) = jj
jj = jj + 1
ii = ii + 1

triangles(jj,1) =  j + nx_dem*ny_dem + (nx_dem-1)*ny_dem 
triangles(jj,2) =  j + 1 + (nx_dem-1)*ny_dem + nx_dem*ny_dem
triangles(jj,3) =  j + 1 + (nx_dem-1)*ny_dem 
patches(4,ii) = jj
jj = jj + 1
ii = ii + 1

end do

num_tri_patches(4) = ii - 1
ii = 1


! and the y=y0 face; 
do i = 1, (nx_dem-1)
triangles(jj,1) =  (i-1)*ny_dem + 1 + nx_dem*ny_dem
!triangles(jj,2) =  (i)*ny_dem + 1 
triangles(jj,2) =  ny_dem + (i-1)*ny_dem + 1 
triangles(jj,3) =  (i-1)*ny_dem + 1 
patches(5,ii) = jj
jj = jj + 1
ii = ii + 1

triangles(jj,1) =  (i-1)*ny_dem + 1 + nx_dem*ny_dem
!triangles(jj,2) =  1 +(i)*ny_dem + nx_dem*ny_dem
triangles(jj,2) =  (i-1)*ny_dem + 1 + nx_dem*ny_dem + ny_dem
triangles(jj,3) =  1 + ny_dem + (i-1)*ny_dem 
patches(5,ii) = jj
jj = jj + 1
ii = ii + 1

end do

num_tri_patches(5) = ii - 1
ii = 1


! and the y=ymax face; 
do i = 1, (nx_dem-1)
triangles(jj,1) =  (i-1)*ny_dem + ny_dem + nx_dem*ny_dem
triangles(jj,2) =  (i-1)*ny_dem + ny_dem 
triangles(jj,3) =  (i-1)*ny_dem + 2*ny_dem
patches(6,ii) = jj
jj = jj + 1
ii = ii + 1

triangles(jj,1) =  (i-1)*ny_dem + ny_dem + nx_dem*ny_dem
triangles(jj,2) =  2*ny_dem +(i-1)*ny_dem  
triangles(jj,3) =  2*ny_dem +(i-1)*ny_dem + nx_dem*ny_dem
patches(6,ii) = jj
jj = jj + 1
ii = ii + 1

end do

num_tri_patches(6) = ii - 1
ii = 1

print*, 'triangles: ',jj - 1, num_triangles


open(20,file= trim(pfsol_filename))
! write version
write(20,'(i1)') 1
! write num vertices/points
write(20,'(i8)') num_points
! write points
do i = 1, num_points
write(20,'(3(f15.4,2x))') points(i,1), points(i,2), points(i,3)
end do
! currently we are assuming 1 solid 
write(20,'(i4)') num_solid
do k = 1, num_solid

! write num trianglesw
write(20,'(i8)') num_triangles
! write triangles
do i = 1, num_triangles
write(20,'(3(i8,2x))') triangles(i,1) -1, triangles(i,2) -1, triangles(i,3) -1
end do ! num_triangles

!write number of patches
write(20,'(i3)') num_patches
do i = 1, num_patches
! write num triangles in each patch
write(20,'(1x,i8)') num_tri_patches(i)
do j = 1, num_tri_patches(i)
write(20,'(i8)') patches(i,j) - 1
!write(20,'(<num_tri_patches(i)>(i8,2x))') patches(i,1:num_tri_patches(i)) - 1
!write(20,*) patches(i,1:num_tri_patches(i)) - 1
end do  ! j, num_tri_patch
end do  ! i, num_patch

end do ! k, num solids
close (20)
print *,"PFSOL File finished"
! write dots to check w/ chunk
open(30,file= 'chunk_check.pt.txt')
! write points
write(30,'(5(f15.4,2x))') points(1,1), points(1,2), points(1,3),1,0.
do i = 2, num_points
write(30,'(5(f15.4,2x))') points(i,1), points(i,2), points(i,3),1,1.
end do
close(30)

! write 3 line triangles segments to check w/ chunk
open(30,file= 'chunk_check.tri.txt')
! write triangles
ii = 1
do i =  1  , num_triangles
if (i>nx_dem*ny_dem) ii = 2
if (i>2*nx_dem*ny_dem) ii = 3
if (i>2*nx_dem*ny_dem+ 2*(nx_dem-1)) ii = 4
if (i>2*nx_dem*ny_dem+ 4*(nx_dem-1)) ii = 5
if (i>2*nx_dem*ny_dem+ 4*(nx_dem-1)+2*(ny_dem-1)) ii = 6

write(30,'(5(f15.4,2x))') points(triangles(i,1),1), points(triangles(i,1),2), points(triangles(i,1),3),ii,0.
write(30,'(5(f15.4,2x))') points(triangles(i,2),1), points(triangles(i,2),2), points(triangles(i,2),3),ii,1.
write(30,'(5(f15.4,2x))') points(triangles(i,3),1), points(triangles(i,3),2), points(triangles(i,3),3),ii,1.
write(30,'(5(f15.4,2x))') points(triangles(i,1),1), points(triangles(i,1),2), points(triangles(i,1),3),ii,1.
end do
close(30)

end program pf_sol_gen


