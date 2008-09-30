program colortab

integer  r, g, b

open (10,file='rainbow256.ctb')

write(10,*) 255*6

! black to blue
g = 0
b = 0
r = 0
do b =  0, 255
write(10,*) r, g, b
end do

! blue to violet
g = 0
b = 255
r = 0
do g = 1, 255
write(10,*) r, g, b
end do

!violet to green
g = 255
b = 255
r = 0
do b = 254, 0, -1
write(10,*) r, g, b
end do

! green to yellow
g = 255
b = 0
r = 0
do r = 1, 255
write(10,*) r, g, b
end do

!yellow to red
g = 255
b = 0
r = 255
do g = 254, 0, -1
write(10,*) r, g, b
end do

!red to white
g = 255
b = 0
r = 255
do g =  1,255
b = g
write(10,*) r, g, b
end do


end program colortab