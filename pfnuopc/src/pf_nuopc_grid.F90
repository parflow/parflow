#include "pf_nuopc_macros.h"

module parflow_nuopc_grid

  use ESMF
  use NUOPC
  use parflow_nuopc_flags
  use iso_c_binding, only: c_int, c_float

  implicit none

  private

  public distgrid_create
  public grid_create
  public grid_write

  !-----------------------------------------------------------------------------
  contains
  !-----------------------------------------------------------------------------

  function distgrid_create(vm, nx, ny, rc) result(pfdistgrid)
    type(ESMF_VM), intent(in)         :: vm
    integer, intent(out)              :: nx
    integer, intent(out)              :: ny
    integer, intent(out)              :: rc
    ! return
    type(ESMF_DistGrid) :: pfdistgrid
    ! local Variables
    integer                        :: petCount
    integer(c_int)                 :: ierr
    integer(c_int)                 :: lclbnds(4)
    integer, allocatable           :: gblbnds(:)
    integer, allocatable           :: deBlockList(:,:,:)
    integer                        :: i

    rc = ESMF_SUCCESS

    call ESMF_VMGet(vm, petCount=petCount, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! grid distribution
    ! call parflox c interface
    ! void cplparflowlcldecomp_(int *sg,
    !                           int *lowerx, int *upperx,
    !                           int *lowery, int *uppery,
    !                           int *ierror)
    call cplparflowlcldecomp(0, lclbnds(1), lclbnds(2), &
      lclbnds(3), lclbnds(4), ierr)
    if (ierr .ne. 0) then
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="cplparflowlcldecomp failed.", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return
    endif
    lclbnds = lclbnds + 1
    allocate(gblbnds(petCount*4))
    call ESMF_VMAllGather(vm, sendData=lclbnds(1:4), &
      recvData=gblbnds(:), count=4, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    allocate(deBlockList(2,2,petCount))
    do i=0, petCount-1
      deBlockList(1,1,i+1) = gblbnds((i*4)+1)
      deBlockList(1,2,i+1) = gblbnds((i*4)+2)
      deBlockList(2,1,i+1) = gblbnds((i*4)+3)
      deBlockList(2,2,i+1) = gblbnds((i*4)+4)
    enddo
    deallocate(gblbnds)
    nx=maxval(deBlockList(1,2,:))
    ny=maxval(deBlockList(2,2,:))
    pfdistgrid = ESMF_DistGridCreate( &
      minIndex=(/1, 1/), &
      maxIndex=(/nx, ny/), &
      deBlockList=deBlockList, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    deallocate(deBlockList)

  end function

  !-----------------------------------------------------------------------------

  function grid_create(pfdistgrid, gname, ctype, cfname, gnx, gny, rc) &
  result(pfgrid)
    type(ESMF_DistGrid), intent(in)   :: pfdistgrid
    character(*), intent(in)          :: gname    ! grid name
    type(grid_coord_flag), intent(in) :: ctype    ! coord type
    character(*), intent(in)          :: cfname   ! coord file name
    integer, intent(in)               :: gnx, gny ! global nx ny
    integer, intent(out)              :: rc
    ! return
    type(ESMF_Grid) :: pfgrid
    ! local Variables
    integer(c_int)                 :: ierr
    integer(c_int), allocatable    :: lclmask(:,:)
    real(c_float), allocatable     :: lclctrx(:,:)
    real(c_float), allocatable     :: lclctry(:,:)
    real(c_float), allocatable     :: lcledgx(:,:)
    real(c_float), allocatable     :: lcledgy(:,:)
    integer                        :: tlb(2), tub(2)
    integer(ESMF_KIND_I4), pointer :: maskPtr(:,:)
    real(ESMF_KIND_R4), pointer    :: ctrxPtr(:,:)
    real(ESMF_KIND_R4), pointer    :: ctryPtr(:,:)
    real(ESMF_KIND_R4), pointer    :: edgxPtr(:,:)
    real(ESMF_KIND_R4), pointer    :: edgyPtr(:,:)

    rc = ESMF_SUCCESS

    if (ctype .eq. GRD_COORD_CARTESIAN) then
      ! create grid
      pfgrid = ESMF_GridCreate(name=trim(gname), &
        distgrid=pfdistgrid, &
        gridAlign=(/-1,-1/), &
        coordSys=ESMF_COORDSYS_CART, &
        coordTypeKind=ESMF_TYPEKIND_R4, &
        indexflag=ESMF_INDEX_GLOBAL, &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! add mask
      call ESMF_GridAddItem(pfgrid, itemflag=ESMF_GRIDITEM_MASK, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetItem(pfgrid, itemflag=ESMF_GRIDITEM_MASK, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, farrayPtr=maskPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      allocate(lclmask(tlb(1):tub(1),tlb(2):tub(2)))
      ! call parflox c interface
      ! void cplparflowlclmask_(int *sg,
      !                         int *localmask,
      !                         int *ierror)
      call cplparflowlclmask(0, lclmask, ierr)
      if (ierr .ne. 0) then
        call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
          msg="cplparflowlclmask failed.", &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return
      endif
      maskPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lclmask(tlb(1):tub(1),tlb(2):tub(2))
      deallocate(lclmask)

      ! add center coordinates
      call ESMF_GridAddCoord(pfgrid, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetCoord(pfgrid, coordDim=1, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, farrayPtr=ctrxPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetCoord(pfgrid, coordDim=2, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, farrayPtr=ctryPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      allocate(lclctrx(tlb(1):tub(1),tlb(2):tub(2)))
      allocate(lclctry(tlb(1):tub(1),tlb(2):tub(2)))
      ! call parflox c interface
      ! void cplparflowlclxyctr_(int   *sg,
      !                          float *localx,
      !                          float *localy,
      !                          int   *ierror)
      call cplparflowlclxyctr(0, lclctrx, lclctry, ierr)
      if (ierr .ne. 0) then
        call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
          msg="cplparflowlclxyctr failed.", &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return
      endif
      ctrxPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lclctrx(tlb(1):tub(1),tlb(2):tub(2))
      ctryPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lclctry(tlb(1):tub(1),tlb(2):tub(2))
      deallocate(lclctrx)
      deallocate(lclctry)

      ! add corner (edge) coordinates
      call ESMF_GridAddCoord(pfgrid, &
        staggerLoc=ESMF_STAGGERLOC_CORNER, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetCoord(pfgrid, coordDim=1, &
        staggerLoc=ESMF_STAGGERLOC_CORNER, farrayPtr=edgxPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetCoord(pfgrid, coordDim=2, &
        staggerLoc=ESMF_STAGGERLOC_CORNER, farrayPtr=edgyPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      allocate(lcledgx(tlb(1):tub(1),tlb(2):tub(2)))
      allocate(lcledgy(tlb(1):tub(1),tlb(2):tub(2)))
      ! call parflox c interface
      ! void cplparflowlclxyedg_(int   *sg,
      !                          float *localx,
      !                          float *localy,
      !                          int   *ierror)
      call cplparflowlclxyedg(0, lcledgx, lcledgy, ierr)
      if (ierr .ne. 0) then
        call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
          msg="cplparflowlclxyedg failed.", &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return
      endif
      edgxPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lcledgx(tlb(1):tub(1),tlb(2):tub(2))
      edgyPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lcledgy(tlb(1):tub(1),tlb(2):tub(2))
      deallocate(lcledgx)
      deallocate(lcledgy)
    elseif (ctype .eq. GRD_COORD_CLMVEGTF) then
      ! create grid
      pfgrid = ESMF_GridCreate(name=trim(gname), &
        distgrid=pfdistgrid, &
        gridAlign=(/-1,-1/), &
        coordSys=ESMF_COORDSYS_SPH_DEG, &
        coordTypeKind=ESMF_TYPEKIND_R4, &
        indexflag=ESMF_INDEX_GLOBAL, &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! add mask
      call ESMF_GridAddItem(pfgrid, itemflag=ESMF_GRIDITEM_MASK, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetItem(pfgrid, itemflag=ESMF_GRIDITEM_MASK, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, farrayPtr=maskPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      allocate(lclmask(tlb(1):tub(1),tlb(2):tub(2)))
      ! call parflox c interface
      ! void cplparflowlclmask_(int *sg,
      !                         int *localmask,
      !                         int *ierror)
      call cplparflowlclmask(0, lclmask, ierr)
      if (ierr .ne. 0) then
        call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
          msg="cplparflowlclmask failed.", &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return
      endif
      maskPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lclmask(tlb(1):tub(1),tlb(2):tub(2))
      deallocate(lclmask)

      ! add center coordinates
      call ESMF_GridAddCoord(pfgrid, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetCoord(pfgrid, coordDim=1, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, farrayPtr=ctrxPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_GridGetCoord(pfgrid, coordDim=2, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, farrayPtr=ctryPtr, &
        totalLBound=tlb, totalUBound=tub, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_DistGridGet(pfdistgrid, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      allocate(lclctrx(tlb(1):tub(1),tlb(2):tub(2)))
      allocate(lclctry(tlb(1):tub(1),tlb(2):tub(2)))
      call read_vegtf(cfname, lbnd=tlb, ubnd=tub, rcnt=(gnx*gny), &
        lat=lclctry, lon=lclctrx, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      ctrxPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lclctrx(tlb(1):tub(1),tlb(2):tub(2))
      ctryPtr(tlb(1):tub(1),tlb(2):tub(2)) = &
        lclctry(tlb(1):tub(1),tlb(2):tub(2))
      deallocate(lclctrx)
      deallocate(lclctry)
    else
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="Unsupported coordinate type", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out      LogSetError
    endif

  end function

  !-----------------------------------------------------------------------------

  subroutine grid_write(grid, fileName, overwrite, status, timeslice, iofmt, &
  relaxedflag, rc)
    type(ESMF_Grid), intent(in)                      :: grid
    character(len=*), intent(in), optional           :: fileName
    logical, intent(in), optional                    :: overwrite
    type(ESMF_FileStatus_Flag), intent(in), optional :: status
    integer, intent(in), optional                    :: timeslice
    type(ESMF_IOFmt_Flag), intent(in), optional      :: iofmt
    logical, intent(in), optional                    :: relaxedflag
    integer, intent(out)                             :: rc
    ! local variables

    logical                 :: ioCapable
    logical                 :: doItFlag
    character(len=64)       :: lfileName
    character(len=64)       :: gridName
    type(ESMF_Array)        :: array
    type(ESMF_ArrayBundle)  :: arraybundle
    logical                 :: isPresent
    integer                 :: dimCount
    integer                 :: dimIndex
    integer                 :: coordDimMax
    integer                 :: stat
    logical                 :: hasCorners

    rc = ESMF_SUCCESS

    ioCapable = (ESMF_IO_PIO_PRESENT .and. &
      (ESMF_IO_NETCDF_PRESENT .or. ESMF_IO_PNETCDF_PRESENT))

    doItFlag = .true. ! default
    if (present(relaxedFlag)) then
      doItFlag = .not.relaxedflag .or. (relaxedflag.and.ioCapable)
    endif

    if (doItFlag) then

      if (present(fileName)) then
        lfileName = trim(fileName)
      else
        call ESMF_GridGet(grid, name=gridName, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        lfileName = trim(gridName)//".nc"
      endif

      arraybundle = ESMF_ArrayBundleCreate(rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! -- centers --

      call ESMF_GridGetCoord(grid, staggerLoc=ESMF_STAGGERLOC_CENTER, &
        isPresent=isPresent, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (isPresent) then
        call ESMF_GridGetCoord(grid, coordDim=1, &
          staggerLoc=ESMF_STAGGERLOC_CENTER, array=array, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArraySet(array, name="lon_center", rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArrayBundleAdd(arraybundle,(/array/),rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_GridGetCoord(grid, coordDim=2, &
          staggerLoc=ESMF_STAGGERLOC_CENTER, array=array, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArraySet(array, name="lat_center", rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArrayBundleAdd(arraybundle,(/array/),rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      endif

      ! -- corners --

      call ESMF_GridGetCoord(grid, staggerLoc=ESMF_STAGGERLOC_CORNER, &
        isPresent=hasCorners, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (hasCorners) then
        call ESMF_GridGetCoord(grid, coordDim=1, &
          staggerLoc=ESMF_STAGGERLOC_CORNER, array=array, rc=rc)
        if (.not. ESMF_STDERRORCHECK(rc)) then
          call ESMF_ArraySet(array, name="lon_corner", rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
          call ESMF_ArrayBundleAdd(arraybundle,(/array/),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
        call ESMF_GridGetCoord(grid, coordDim=2, &
          staggerLoc=ESMF_STAGGERLOC_CORNER, array=array, rc=rc)
        if (.not. ESMF_STDERRORCHECK(rc)) then
          call ESMF_ArraySet(array, name="lat_corner", rc=rc)
          if (ESMF_LogFoundError(rcToCheck=rc, PASSTHRU)) return  ! bail out
          call ESMF_ArrayBundleAdd(arraybundle,(/array/),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      endif

      ! -- mask --

      call ESMF_GridGetItem(grid, itemflag=ESMF_GRIDITEM_MASK, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, isPresent=isPresent, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (isPresent) then
        call ESMF_GridGetItem(grid, staggerLoc=ESMF_STAGGERLOC_CENTER, &
          itemflag=ESMF_GRIDITEM_MASK, array=array, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArraySet(array, name="mask", rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArrayBundleAdd(arraybundle,(/array/),rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      endif

      ! -- area --

      call ESMF_GridGetItem(grid, itemflag=ESMF_GRIDITEM_AREA, &
        staggerLoc=ESMF_STAGGERLOC_CENTER, isPresent=isPresent, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (isPresent) then
        call ESMF_GridGetItem(grid, staggerLoc=ESMF_STAGGERLOC_CENTER, &
          itemflag=ESMF_GRIDITEM_AREA, array=array, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArraySet(array, name="area", rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_ArrayBundleAdd(arraybundle,(/array/),rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      endif

      call ESMF_ArrayBundleWrite(arraybundle, &
        fileName=trim(lfileName),rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      call ESMF_ArrayBundleDestroy(arraybundle,rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif
  end subroutine

  !-----------------------------------------------------------------------------

  subroutine read_vegtf(fname, lbnd, ubnd, rcnt, lat, lon, rc)
    character(len=*), intent(in)    :: fname
    integer, intent(in)             :: lbnd(2)  ! local lower bnd
    integer, intent(in)             :: ubnd(2)  ! local upper bnd
    integer, intent(in)             :: rcnt     ! global nx*ny
    real(ESMF_KIND_R4), intent(out) :: lat(lbnd(1):ubnd(1),lbnd(2):ubnd(2))
    real(ESMF_KIND_R4), intent(out) :: lon(lbnd(1):ubnd(1),lbnd(2):ubnd(2))
    integer, intent(out)            :: rc

    type :: recfmt
      integer  :: x
      integer  :: y
      real*8   :: lat
      real*8   :: lon
      real*4   :: sand
      real*4   :: clay
      integer  :: color
      integer  :: i01, i02, i03, i04, i05, i06
      integer  :: i07, i08, i09, i10, i11, i12
      integer  :: i13, i14, i15, i16, i17, i18
    end type recfmt

    type(recfmt) :: recdat
    integer :: funit
    integer :: ierr
    integer :: i

    lat = -9999
    lon = -9999
    rc = ESMF_RC_FILE_READ

    call ESMF_UtilIOUnitGet(funit, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    open(funit, file=fname, status='old', access='sequential', &
      action="read", form="formatted", iostat=ierr)
    if (ierr .ne. 0) then
      call ESMF_LogSetError(ESMF_RC_FILE_OPEN, &
        msg="Failed to open: "//trim(fname), &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out
    endif

    ! skip two line header
    read(funit,*)
    read(funit,*)
    ! loop over all records
    do i=1, rcnt
      read(funit, *, end=1, iostat=ierr) recdat
      if (ierr .ne. 0) then
        call ESMF_LogSetError(ESMF_RC_FILE_READ, &
          msg="File read failed, check format: "//trim(fname), &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return  ! bail out
      endif
        if (recdat%x .ge. lbnd(1) .AND. recdat%x .le. ubnd(1) .AND. &
            recdat%y .ge. lbnd(2) .AND. recdat%y .le. ubnd(2)) then
        lat(recdat%x,recdat%y) = real(recdat%lat,ESMF_KIND_R4)
        lon(recdat%x,recdat%y) = real(recdat%lon,ESMF_KIND_R4)
      endif
      rc = ESMF_SUCCESS
    enddo
1   close(funit,iostat=ierr)
    if (ierr .ne. 0) then
      call ESMF_LogSetError(ESMF_RC_FILE_CLOSE, &
        msg="Failed to close: "//trim(fname), &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out
    endif
  end subroutine

  !-----------------------------------------------------------------------------

end module parflow_nuopc_grid
