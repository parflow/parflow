% Matlab script to read parflow binary files
% Jehan Rihani, UCB
%
clear all
% open file
[fid,message] = fopen('PARFLOW.out.FILE.pfb','r','ieee-be'); % (filename,permission,format) 

% Read domain spatial information
x1 = fread(fid,1,'double');    %Lower X
y1= fread(fid,1,'double');    %Lower Y
z1 = fread(fid,1,'double');    %Lower Z

nx = fread(fid,1,'int32');  % NX
ny = fread(fid,1,'int32');  % NY
nz = fread(fid,1,'int32');  % NZ

dx = fread(fid,1,'double');
dy = fread(fid,1,'double');
dz = fread(fid,1,'double');

ns = fread(fid,1,'int32');   % num_subgrids
% Loop over number of subgrids
for is = 1:ns;  %number of subgrids

% Read subgrid spatial information
   ix = fread(fid,1,'int32');
   iy = fread(fid,1,'int32');   
   iz = fread(fid,1,'int32');   

   nnx = fread(fid,1,'int32');  % nx
   nny = fread(fid,1,'int32');  % ny   
   nnz = fread(fid,1,'int32');  % nz 

   rx = fread(fid,1,'int32');
   ry = fread(fid,1,'int32');   
   rz = fread(fid,1,'int32');   
 
% Read Pressure data from each subgrid
for k=(iz+1):(iz+nnz);
    for j=(iy+1):(iy+nny);
        for i=(ix+1):(ix+nnx);
            satur(i,j,k) = fread(fid,1,'double');
        end   % i
    end   %j
end   %k

end %is


