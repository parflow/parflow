!#include <misc.h>

subroutine drv_restart (rw, drv, tile, clm, rank, istep_pf)

  !=========================================================================
  !
  !  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
  !  L                        M  available land surface process model.  
  !  M --COMMON LAND MODEL--  C  	
  !  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
  !  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
  !
  !=========================================================================
  ! DESCRIPTION:
  !  This program reads and writes restart files for CLM.  This
  !   includes all relevant water/energy storages, tile information,
  !   and time information.  It also rectifies changes in the tile space.  
  !
  ! REVISION HISTORY:
  !  22  Oct 1999: Jon Radakovich and Paul Houser; Initial code
  !=========================================================================
  ! RESTART FILE FORMAT(fortran sequential binary):
  !  yr,mo,da,hr,mn,ss,vclass,nch !Restart time,Veg class,no.tiles, no.soil lay 
  !  tile(nch)%col        !Grid Col of Tile   
  !  tile(nch)%row        !Grid Row of Tile
  !  tile(nch)%fgrd       !Fraction of Grid covered by tile
  !  tile(nch)%vegt       !Vegetation Type of Tile
  !  clm(nch)%states      !Model States in Tile Space
  !=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use clm_varpar, only : nlevsoi, nlevsno
  use clm_varcon, only : denh2o, denice
  implicit none

  !=== Arguments ===========================================================  

  integer, intent(in)    :: rw         ! 1=read restart, 2=write restart
  integer, intent(inout) :: istep_pf   ! istep counter, incremented in PF
  type (drvdec)  :: drv              
  type (tiledec) :: tile(drv%nch)
  type (clm1d)   :: clm (drv%nch)

  !=== Local Variables =====================================================

  integer :: c,t,l,n       ! Loop counters
  integer :: found         ! Counting variable

  !=== Temporary tile space transfer files (different than in DRV_module)

  integer :: yr,mo,da,hr,mn,ss        ! Time variables
  integer :: vclass,nc,nr,nch
  integer, pointer  :: col(:)         ! Column
  integer, pointer  :: row(:)         ! Row
  integer, pointer  :: vegt(:)        ! Tile veg type
  real(r8), pointer :: fgrd(:)        ! Grid Fraction of Tile
  real(r8), pointer :: t_grnd(:)      ! CLM Soil Surface Temperature [K]
  real(r8), pointer :: t_veg(:)       ! CLM Leaf Temperature [K]
  real(r8), pointer :: h2osno(:)      ! CLM Snow Cover, Water Equivalent [mm]
  real(r8), pointer :: snowage(:)     ! CLM Non-dimensional snow age [-] 
  real(r8), pointer :: snowdp(:)      ! CLM Snow Depth [m] 
  real(r8), pointer :: h2ocan(:)      ! CLM Depth of Water on Foliage [mm]

  real(r8), pointer :: frac_sno(:)            ! CLM Fractional Snow Cover [-]
  real(r8), pointer :: elai(:)                ! CLM Leaf Area Index
  real(r8), pointer :: esai(:)                ! CLM Stem Area Index

  integer, pointer  :: snl(:)       ! CLM Actual number of snow layers
  integer           :: istep        ! number of time step
  real(r8), pointer :: xerr(:)      ! accumulation of water balance error
  real(r8), pointer :: zerr(:)      ! accumulation of energy balnce error

  real(r8), pointer :: dz(:,:)           ! CLM Layer Depth [m]
  real(r8), pointer :: z(:,:)            ! CLM Layer Thickness [m]
  real(r8), pointer :: zi(:,:)           ! CLM Interface Level Below a "z" Level [m]
  real(r8), pointer :: t_soisno(:,:)     ! CLM Soil + Snow Layer Temperature [K]
  real(r8), pointer :: h2osoi_liq(:,:)   ! CLM Average Soil Water Content [kg/m2]
  real(r8), pointer :: h2osoi_ice(:,:)   ! CLM Average Ice Content [kg/m2]

  real(r8), pointer :: tmptileot(:) ! Temporary Transfer Array   
  real(r8), pointer :: tmptileow(:) ! Temporary Transfer Array 
  real(r8), pointer :: tmptileoi(:) ! Temporary Transfer Array
  real(r8), pointer :: tmptileoa(:) ! Temporary Transfer Array
  real(r8) :: tmptilent(drv%nch)    ! Temporary Transfer Array   
  real(r8) :: tmptilenw(drv%nch)    ! Temporary Transfer Array
  real(r8) :: tmptileni(drv%nch)    ! Temporary Transfer Array

  real(r8) :: g_t_grnd(drv%nc,drv%nr)         ! CLM Soil Surface Temperature [K]
  real(r8) :: g_t_veg(drv%nc,drv%nr)          ! CLM Leaf Temperature [K] 
  real(r8) :: g_h2osno(drv%nc,drv%nr)         ! CLM Snow Cover, Water Equivalent [mm] 
  real(r8) :: g_snowage(drv%nc,drv%nr)        ! CLM Non-dimensional snow age [-] 
  real(r8) :: g_snowdp(drv%nc,drv%nr)         ! CLM Snow Depth [m] 
  real(r8) :: g_h2ocan(drv%nc,drv%nr)         ! CLM Depth of Water on Foliage [mm]
  real(r8) :: g_frac_sno(drv%nc,drv%nr)       ! CLM Fractional Snow Cover [-]
  real(r8) :: g_elai(drv%nc,drv%nr)           ! CLM Leaf + Stem Area Index
  real(r8) :: g_esai(drv%nc,drv%nr)           ! CLM Leaf + Stem Area Index

  real(r8) :: g_dz(drv%nc,drv%nr,-nlevsno+1:nlevsoi)    ! CLM Layer Depth [m]
  real(r8) :: g_z (drv%nc,drv%nr,-nlevsno+1:nlevsoi)    ! CLM Layer Thickness [m]
  real(r8) :: g_zi(drv%nc,drv%nr,-nlevsno:nlevsoi)      ! CLM Interface Level Below a "z" Level [m]
  real(r8) :: g_t_soisno  (drv%nc,drv%nr,nlevsoi)       ! CLM Soil + Snow Layer Temperature [K]
  real(r8) :: g_h2osoi_liq(drv%nc,drv%nr,nlevsoi)       ! CLM Average Soil Water Content [kg/m2]
  real(r8) :: g_h2osoi_ice(drv%nc,drv%nr,-nlevsno+1:nlevsoi) ! CLM Average Ice Content [kg/m2]

  integer :: rank
  character*100 RI
  
  ! IMF -- for adding time stamp to restart files
  integer :: tstamp
  character*100 TS

  !=== End Variable Definition =============================================

  write(RI,*)  rank

  !=== Read Active Archive File ============================================

  if((rw.eq.1.and.drv%clm_ic.eq.1).or.(rw.eq.1.and.drv%startcode.eq.1))then

     ! IMF -- Read restart from PREVIOUS timestep 
     !        (i.e., start from end point of previous timestep)
     tstamp = istep_pf - 1
     write(TS,'(I5.5)') tstamp
     open(40,file=trim(adjustl(drv%rstf))//trim(adjustl(TS))//'.'//trim(adjustl(RI)),form='unformatted')
     ! open(40,file=trim(adjustl(drv%rstf))//trim(adjustl(RI)),form='unformatted')

     read(40)     yr,mo,da,hr,mn,ss,vclass,nc,nr,nch  !Time, veg class, no. tiles
     !write(999,*) yr,mo,da,hr,mn,ss,vclass,nc,nr,nch  !Time, veg class, no. tiles

     allocate (col(nch),row(nch),fgrd(nch),vegt(nch))
     allocate (t_grnd(nch),t_veg(nch),h2osno(nch),snowage(nch),         &
          snowdp(nch),h2ocan(nch),frac_sno(nch))
     allocate (elai(nch), esai(nch), snl(nch),xerr(nch),zerr(nch))
     allocate (dz(nch,-nlevsno+1:nlevsoi),        &
          z(nch,-nlevsno+1:nlevsoi),         &
          zi(nch,-nlevsno:nlevsoi),          &           
          t_soisno(nch,-nlevsno+1:nlevsoi),  &
          tmptileot(nch),                    &
          h2osoi_liq(nch,-nlevsno+1:nlevsoi),&
          tmptileow(nch),                    &
          h2osoi_ice(nch,-nlevsno+1:nlevsoi),&
          tmptileoi(nch),tmptileoa(nch))

     read(40) col                  !Grid Col of Tile   
     read(40) row                  !Grid Row of Tile
     read(40) fgrd                 !Fraction of Grid covered by tile
     read(40) vegt                 !Vegetation Type of Tile
     read(40) t_grnd               !CLM Soil Surface Temperature [K] 
     read(40) t_veg                !CLM Leaf Temperature [K] 
     read(40) h2osno               !CLM Snow Cover, Water Equivalent [mm] 
     read(40) snowage              !CLM Non-dimensional snow age [-] 
     read(40) snowdp               !CLM Snow Depth [m]
     read(40) h2ocan               !CLM Depth of Water on Foliage [mm]
     read(40) frac_sno             !CLM Fractional Snow Cover [-]
     read(40) elai                 !CLM Leaf Area Index
     read(40) esai                 !CLM Stem Area Index
     read(40) snl                  !CLM Actual number of snow layers
     read(40) xerr                 !CLM Accumulation of water balance error
     read(40) zerr                 !CLM Accumulation of energy balnce error
     read(40) istep                !CLM Number of time step

     do l = -nlevsno+1,nlevsoi
        read(40) tmptileoa  !CLM Layer Depth [m]
        do t = 1,drv%nch
           dz(t,l) = tmptileoa(t) 
        enddo
     enddo
     do l = -nlevsno+1,nlevsoi
        read(40) tmptileoa  !CLM Layer Thickness [m]
        do t = 1,drv%nch
           z(t,l) = tmptileoa(t) 
        enddo
     enddo
     do l = -nlevsno,nlevsoi
        read(40) tmptileoa  !CLM Interface Level Below a "z" Level [m]
        do t = 1,drv%nch
           zi(t,l) = tmptileoa(t) 
        enddo
     enddo

     do l = -nlevsno+1,nlevsoi
        read(40) tmptileot  !CLM Soil + Snow Layer Temperature [K]
        do t = 1,drv%nch
           t_soisno(t,l) = tmptileot(t) 
        enddo
     enddo
     do l = -nlevsno+1,nlevsoi
        read(40) tmptileow  !Average Soil Water Content [kg/m2]
        do t = 1,drv%nch
           h2osoi_liq(t,l) = tmptileow(t)
        enddo
     enddo
     do l = -nlevsno+1,nlevsoi
        read(40) tmptileoi  !CLM Average Ice Content [kg/m2]
        do t = 1,drv%nch
           h2osoi_ice(t,l) = tmptileoi(t)
        enddo
     enddo

     close(40)
     if(rank.eq.0)then
        write(*,*)'CLM Restart File Read: ',drv%rstf
     endif

     !=== Establish Model Restart Time  

     if(drv%startcode.eq.1)then
        drv%yr = yr
        drv%mo = mo 
        drv%da = da
        drv%hr = hr
        drv%mn = mn
        drv%ss = ss
        call drv_date2time(drv%time,drv%doy,drv%day,drv%gmt,yr,mo,da,hr,mn,ss) 
        drv%ctime = drv%time !@ assign restart time "ctime"
        if(rank.eq.0)then
           write(*,*)'CLM Restart File Time Used: ',drv%rstf
        endif
     endif

     !=== Rectify Restart Tile Space to DRV Tile Space =====================

     if(drv%clm_ic.eq.1)then

        ! Check for Vegetation Class Conflict 

        if(vclass.ne.drv%vclass)then
           write(*,*)drv%rstf,' Vegetation class conflict - CLM HALTED'
           stop
        endif

        ! Check for Grid Space Conflict 

        if(nc.ne.drv%nc.or.nr.ne.drv%nr)then
           write(*,*)drv%rstf,'Grid space mismatch - CLM HALTED'
           stop
        endif

        ! Transfer Restart tile space to DRV tile space

        if(nch.ne.drv%nch)then
           if(rank.eq.0)then
              write(*,*)'Restart Tile Space Mismatch-Transfer in Progress'
           endif

           !  Start by finding grid averages

           call drv_t2gr(t_grnd         ,g_t_grnd         ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(t_veg          ,g_t_veg          ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(h2osno         ,g_h2osno         ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(snowage        ,g_snowage        ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(snowdp         ,g_snowdp         ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(h2ocan         ,g_h2ocan         ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(frac_sno       ,g_frac_sno       ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(elai           ,g_elai           ,drv%nc,drv%nr,nch,fgrd,col,row)
           call drv_t2gr(esai           ,g_esai           ,drv%nc,drv%nr,nch,fgrd,col,row)

           do l = -nlevsno+1,nlevsoi
              call drv_t2gr(dz(:,l),g_dz(:,:,l),drv%nc,drv%nr,nch,fgrd,col,row)
           enddo
           do l = -nlevsno+1,nlevsoi
              call drv_t2gr(z(:,l),g_z(:,:,l),drv%nc,drv%nr,nch,fgrd,col,row)
           enddo
           do l = -nlevsno,nlevsoi
              call drv_t2gr(zi(:,l),g_zi(:,:,l),drv%nc,drv%nr,nch,fgrd,col,row)
           enddo

           do l = -nlevsno+1,nlevsoi
              call drv_t2gr(t_soisno(:,l),g_t_soisno(:,:,l),drv%nc,drv%nr,nch,fgrd,col,row)
           enddo

           do l = -nlevsno+1,nlevsoi
              call drv_t2gr(h2osoi_liq(:,l),g_h2osoi_liq(:,:,l),drv%nc,drv%nr,nch,fgrd,col,row)
           enddo

           do l = -nlevsno+1,nlevsoi
              call drv_t2gr(h2osoi_ice(:,l),g_h2osoi_ice(:,:,l),drv%nc,drv%nr,nch,fgrd,col,row)
           enddo

           ! Perform state transfer

           c = 0
           do 555 t = 1,drv%nch 

              if(amod(float(t),10000.0).eq.0.0)then
                 if(rank.eq.0)then
                    write(*,23)'  Transferred ', &
                         100.0*float(t)/float(drv%nch),' Percent of Tiles'
                 endif
              endif

23            format(a14,f5.2,a17)
              found = 0
              do n = 1,nch
                 if ( tile(t)%vegt.eq.vegt(n) .and.   &
                      tile(t)%col .eq. col(n) .and.   &
                      tile(t)%row .eq. row(n) )then
                    clm(t)%t_grnd = t_grnd(n)
                    clm(t)%t_veg = t_veg(n)
                    clm(t)%h2osno = h2osno(n)
                    clm(t)%snowage = snowage(n)
                    clm(t)%snowdp = snowdp(n)
                    clm(t)%h2ocan = h2ocan(n)
                    clm(t)%frac_sno = frac_sno(n)
                    clm(t)%elai = elai(n)
                    clm(t)%esai = esai(n)

                    do l = -nlevsno+1,nlevsoi
                       clm(t)%dz(l) = dz(n,l)
                    enddo
                    do l = -nlevsno+1,nlevsoi
                       clm(t)%z(l) = z(n,l)
                    enddo
                    do l = -nlevsno,nlevsoi
                       clm(t)%zi(l) = zi(n,l)
                    enddo
                    do l = -nlevsno+1,nlevsoi
                       clm(t)%t_soisno(l) = t_soisno(n,l)
                    enddo
                    do l = -nlevsno+1,nlevsoi
                       clm(t)%h2osoi_liq(l) = h2osoi_liq(n,l)
                    enddo
                    do l = -nlevsno+1,nlevsoi
                       clm(t)%h2osoi_ice(l) = h2osoi_ice(n,l)
                    enddo

                    found = 1
                    goto 555 
                 endif
              enddo

              if(found.eq.0)then        
                 clm(t)%t_grnd = g_t_grnd(tile(t)%col,tile(t)%row)
                 clm(t)%t_veg = g_t_veg(tile(t)%col,tile(t)%row)
                 clm(t)%h2osno = g_h2osno(tile(t)%col,tile(t)%row)
                 clm(t)%snowage = g_snowage(tile(t)%col,tile(t)%row)
                 clm(t)%snowdp = g_snowdp(tile(t)%col,tile(t)%row)
                 clm(t)%h2ocan = g_h2ocan(tile(t)%col,tile(t)%row)
                 clm(t)%frac_sno = g_frac_sno(tile(t)%col,tile(t)%row)
                 clm(t)%elai = g_elai(tile(t)%col,tile(t)%row)
                 clm(t)%esai = g_esai(tile(t)%col,tile(t)%row)
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%dz(l) = g_dz(tile(t)%col,tile(t)%row,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%z(l) = g_z(tile(t)%col,tile(t)%row,l)
                 enddo
                 do l = -nlevsno,nlevsoi
                    clm(t)%zi(l) = g_zi(tile(t)%col,tile(t)%row,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%t_soisno(l) = g_t_soisno(tile(t)%col,tile(t)%row,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%h2osoi_liq(l) = g_h2osoi_liq(tile(t)%col,tile(t)%row,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%h2osoi_ice(l) = g_h2osoi_ice(tile(t)%col,tile(t)%row,l)
                 enddo
                 c = 0
              endif
555           continue
              if(rank.eq.0)then
                 write(*,*)'Tile Space Transfer Complete'
                 write(*,*)'CLM Restart NCH:',nch,'Current NCH:',drv%nch
                 write(*,*) c, ' Tiles not found in old CLM restart'        
                 write(*,*)
              endif

           else  !The number of tiles is a match

              do t = 1,drv%nch
                 clm(t)%t_grnd = t_grnd(t)
                 clm(t)%t_veg = t_veg(t)
                 clm(t)%h2osno = h2osno(t)
                 clm(t)%snowage = snowage(t)
                 clm(t)%snowdp = snowdp(t)
                 clm(t)%h2ocan = h2ocan(t)
                 clm(t)%frac_sno = frac_sno(t)
                 clm(t)%elai = elai(t)
                 clm(t)%esai = esai(t)
                 clm(t)%snl=snl(t)
                 clm(t)%acc_errh2o=xerr(t)
                 clm(t)%acc_errseb=zerr(t)

                 do l = -nlevsno+1,nlevsoi
                    clm(t)%dz(l) = dz(t,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%z(l) = z(t,l)
                 enddo
                 do l = -nlevsno,nlevsoi
                    clm(t)%zi(l) = zi(t,l)
                 enddo

                 do l = -nlevsno+1,nlevsoi
                    clm(t)%t_soisno(l) = t_soisno(t,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%h2osoi_liq(l) = h2osoi_liq(t,l)
                 enddo
                 do l = -nlevsno+1,nlevsoi
                    clm(t)%h2osoi_ice(l) = h2osoi_ice(t,l)
                 enddo
              enddo
           endif

        endif

        ! Determine h2osoi_vol(1) - needed for soil albedo calculation

        do t = 1,drv%nch
           clm(t)%h2osoi_vol(1) = clm(t)%h2osoi_liq(1)/(clm(t)%dz(1)*denh2o) &
                + clm(t)%h2osoi_ice(1)/(clm(t)%dz(1)*denice)
        end do

     endif !RW option 1

     ! === Set starttime to CLMin Stime when STARTCODE = 2 
     if(rw.eq.1.and.drv%startcode.eq.2)then 
        drv%yr = drv%syr
        drv%mo = drv%smo 
        drv%da = drv%sda
        drv%hr = drv%shr
        drv%mn = drv%smn
        drv%ss = drv%sss
        call drv_date2time(drv%time,drv%doy,drv%day,drv%gmt, &
             drv%yr,drv%mo,drv%da,drv%hr,drv%mn,drv%ss) 
        if(rank.eq.0)then
           write(*,*)'Using drv_clmin.dat start time ',drv%time
        endif
     endif

     if(rw.eq.1)then
        if(rank.eq.0)then
           write(*,*)'CLM Start Time: ',drv%yr,drv%mo,drv%da,drv%hr,drv%mn,drv%ss
           write(*,*)
        endif
     endif



     !=== Restart Writing (2 file are written - active and archive)

     if(rw.eq.2)then

        if(rank.eq.0)then
           write(*,*)'Write CLM Active Restart: istep_pf = ',istep_pf
        endif

        ! IMF -- add time stamp (istep) to restart file
        !        NOTE: READ from istep-1 (previous time)
        !              WRITE to istep (current time)
        tstamp = istep_pf 
        write(TS,'(I5.5)') tstamp
        open(40,file=trim(adjustl(drv%rstf))//trim(adjustl(TS))//'.'//trim(adjustl(RI)),form='unformatted')
        !open(40,file=trim(adjustl(drv%rstf))//trim(adjustl(RI)),form='unformatted') !Active archive restart

        write(40) drv%yr,drv%mo,drv%da,drv%hr,drv%mn,drv%ss,&
             drv%vclass,drv%nc,drv%nr,drv%nch  !Veg class, no tiles       
        write(40) tile%col                  !Grid Col of Tile   
        write(40) tile%row                  !Grid Row of Tile
        write(40) tile%fgrd                 !Fraction of Grid covered by tile
        write(40) tile%vegt                 !Vegetation Type of Tile
        write(40) clm%t_grnd                !CLM Soil Surface Temperature [K]
        write(40) clm%t_veg                 !CLM Leaf Temperature [K]
        write(40) clm%h2osno                !CLM Snow Cover, Water Equivalent [mm]
        write(40) clm%snowage               !CLM Non-dimensional snow age [-]
        write(40) clm%snowdp                !CLM Snow Depth [m]
        write(40) clm%h2ocan                !CLM Depth of Water on Foliage [mm]
        write(40) clm%frac_sno              !CLM Fractional Snow Cover [-]
        write(40) clm%elai                  !CLM Leaf Area Index
        write(40) clm%esai                  !CLM Stem Area Index
        write(40) clm%snl                   !CLM Actual number of snow layers
        write(40) clm%acc_errh2o            !CLM Accumulation of water balance error
        write(40) clm%acc_errseb            !CLM Accumulation of energy balance error
        write(40) istep_pf

        do l = -nlevsno+1,nlevsoi
           do t = 1,drv%nch
              tmptilent(t) = clm(t)%dz(l)  
           enddo
           write(40) tmptilent     !CLM Layer Depth [m]
        enddo
        do l = -nlevsno+1,nlevsoi
           do t = 1,drv%nch
              tmptilent(t) = clm(t)%z(l)  
           enddo
           write(40) tmptilent     !CLM Layer Thickness [m]
        enddo
        do l = -nlevsno,nlevsoi
           do t = 1,drv%nch
              tmptilent(t) = clm(t)%zi(l)  
           enddo
           write(40) tmptilent     !CLM Interface Level Below a "z" Level [m]
        enddo

        do l = -nlevsno+1,nlevsoi
           do t = 1,drv%nch
              tmptilent(t) = clm(t)%t_soisno(l)  
           enddo
           write(40) tmptilent     !CLM Soil + Snow Layer Temperature [K] 
        enddo
        do l = -nlevsno+1,nlevsoi
           do t = 1,drv%nch
              tmptilenw(t) = clm(t)%h2osoi_liq(l)
           enddo
           write(40) tmptilenw     !CLM Average Soil Water Content [kg/m2]
        enddo
        do l = -nlevsno+1,nlevsoi
           do t = 1,drv%nch
              tmptileni(t) = clm(t)%h2osoi_ice(l)
           enddo
           write(40) tmptileni     !CLM Average Ice Content [kg/m2] 
        enddo

        close(40)

        if(rank.eq.0)then
           write(*,*)'CLM Active Restart Written: ',drv%rstf
        endif
     endif

     return
   end subroutine drv_restart


