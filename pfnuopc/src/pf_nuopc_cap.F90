!NUOPC:MEDIATION_LAYER:PHYSICS
!
#include "pf_nuopc_macros.h"

module parflow_nuopc
  use ESMF
  use NUOPC
  use NUOPC_Model, &
    model_routine_SS           => SetServices, &
    model_label_SetClock       => label_SetClock, &
    model_label_DataInitialize => label_DataInitialize, &
    model_label_CheckImport    => label_CheckImport, &
    model_label_Advance        => label_Advance, &
    model_label_Finalize       => label_Finalize
  use parflow_nuopc_fields
  use parflow_nuopc_grid
  use parflow_nuopc_flags
  use iso_c_binding, only: c_null_char, c_int, c_double, c_float

  implicit none

  private

  public SetVM, SetServices

  character(LEN=*), parameter :: label_InternalState = 'InternalState'

  type type_InternalStateStruct
    logical                :: initialized        = .false.
    type(ESMF_FieldBundle) :: pf_fields
    logical                :: prep_run           = .false.
    character(len=64)      :: prep_util          = "none"
    character(len=64)      :: prep_filename      = "none"
    character(len=64)      :: pfidb_filename     = "none"
    logical                :: multi_instance     = .false.
    logical                :: realize_all_export = .false.
    logical                :: realize_all_import = .false.
    type(field_init_flag)  :: init_export        = FLD_INIT_ZERO
    type(field_init_flag)  :: init_import        = FLD_INIT_ZERO
    type(field_check_flag) :: check_import       = FLD_CHECK_CURRT
    type(geom_src_flag)    :: geom_src           = GEOM_PROVIDE
    type(grid_coord_flag)  :: ctype              = GRD_COORD_CLMVEGTF
    character(len=64)      :: coord_filename     = "none"
    integer                :: nx                 = 0
    integer                :: ny                 = 0
    integer                :: nz                 = 0
    integer                :: cplnz              = 0
    real,allocatable       :: cpldz(:)
    character(len=16)      :: transfer_offer     = "cannot provide"
    character(len=64)      :: input_dir          = "."
    character(len=64)      :: output_dir         = "."
    type(ESMF_Time)        :: pf_epoch
  end type

  type type_InternalState
    type(type_InternalStateStruct), pointer :: wrap
  end type

  interface
    integer function local_chdir(path) bind(C, name="chdir")
      use iso_c_binding
      character(c_char) :: path(*)
    end function
  end interface

!EOP

!------------------------------------------------------------------
  contains
!------------------------------------------------------------------

  subroutine SetServices(gcomp, rc)
    type(ESMF_GridComp)  :: gcomp
    integer, intent(out) :: rc

    ! local variables
    type(type_InternalState)   :: is
    integer                    :: stat

    rc = ESMF_SUCCESS

    ! allocate memory for this internal state and set it in the component
    allocate(is%wrap, stat=stat)
    if (ESMF_LogFoundAllocError(statusToCheck=stat, &
      msg='Allocation of internal state memory failed.', &
      line=__LINE__, file=__FILE__, rcToReturn=rc)) return  ! bail out
    call ESMF_UserCompSetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! the NUOPC model component will register the generic methods
    call NUOPC_CompDerive(gcomp, model_routine_SS, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! switching to IPD versions
    call ESMF_GridCompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
      userRoutine=InitializeP0, phase=0, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! set entry point for methods that require specific implementation
    call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
      phaseLabelList=(/"IPDv03p1"/), userRoutine=InitializeP1, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
      phaseLabelList=(/"IPDv03p3"/), userRoutine=InitializeP3, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
!    call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
!      phaseLabelList=(/"IPDv03p4"/), userRoutine=InitializeP4, rc=rc)
!    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
!    call NUOPC_CompSetEntryPoint(gcomp, ESMF_METHOD_INITIALIZE, &
!      phaseLabelList=(/"IPDv03p5"/), userRoutine=InitializeP5, rc=rc)
!    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! attach specializing method(s)
    call NUOPC_CompSpecialize(gcomp, specLabel=model_label_SetClock, &
      specRoutine=SetClock, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call NUOPC_CompSpecialize(gcomp, specLabel=model_label_DataInitialize, &
       specRoutine=DataInitialize, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call ESMF_MethodRemove(gcomp, label=model_label_CheckImport, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call NUOPC_CompSpecialize(gcomp, specLabel=model_label_CheckImport, &
       specRoutine=CheckImport, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call NUOPC_CompSpecialize(gcomp, speclabel=model_label_Advance, &
      specRoutine=ModelAdvance, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call NUOPC_CompSpecialize(gcomp, specLabel=model_label_Finalize, &
      specRoutine=ModelFinalize, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

  end subroutine

!..................................................................

  subroutine InitializeP0(gcomp, importState, exportState, clock, rc)
    type(ESMF_GridComp)   :: gcomp
    type(ESMF_State)      :: importState, exportState
    type(ESMF_Clock)      :: clock
    integer, intent(out)  :: rc

    ! local variables
    character(32)              :: cname
    character(*), parameter    :: rname="InitializeP0"
    integer                    :: verbosity, diagnostic
    type(type_InternalState)   :: is
    integer                    :: stat

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! Switch to IPDv03 by filtering all other phaseMap entries
    call NUOPC_CompFilterPhaseMap(gcomp, ESMF_METHOD_INITIALIZE, &
      acceptStringList=(/"IPDv03p"/), rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call PF_AttributeRead(rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! prepare diagnostics folder
    if (btest(diagnostic,16)) then
      call ESMF_UtilIOMkDir(pathName=trim(is%wrap%output_dir), &
        relaxedFlag=.true., rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

    contains ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    subroutine PF_AttributeRead(rc)
      integer, intent(out)  :: rc

      ! local variables
      logical                    :: configIsPresent
      logical                    :: attrIsPresent
      type(ESMF_Config)          :: config
      type(NUOPC_FreeFormat)     :: attrFF
      character(len=64)          :: attval
      character(ESMF_MAXSTR)     :: logMsg
      integer                    :: stat
      integer                    :: oldidx
      integer                    :: newidx
      integer                    :: i

      ! check gcomp for config
      call ESMF_GridCompGet(gcomp, configIsPresent=configIsPresent, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! read and ingest free format component attributes
      if (configIsPresent) then
        call ESMF_GridCompGet(gcomp, config=config, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        attrFF = NUOPC_FreeFormatCreate(config, &
          label=trim(cname)//"_attributes::", relaxedflag=.true., rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call NUOPC_CompAttributeIngest(gcomp, attrFF, addFlag=.true., rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call NUOPC_FreeFormatDestroy(attrFF, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      endif

      ! multiple instances
      call ESMF_AttributeGet(gcomp, name="multi_instance_gwr", &
        value=is%wrap%multi_instance, defaultvalue=.false., &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! realize all import fields
      call ESMF_AttributeGet(gcomp, name="realize_all_import", &
        value=is%wrap%realize_all_import, defaultvalue=.false., &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! realize all export fields
      call ESMF_AttributeGet(gcomp, name="realize_all_export", &
        value=is%wrap%realize_all_export, defaultvalue=.false., &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! set preprocessing information
      call ESMF_AttributeGet(gcomp, name="prep_filename", &
        isPresent=attrIsPresent, &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (attrIsPresent) then
        is%wrap%prep_run=.true.
        call ESMF_AttributeGet(gcomp, name="prep_filename", &
          value=is%wrap%prep_filename, defaultvalue="config_file", &
          convention="NUOPC", purpose="Instance", rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_AttributeGet(gcomp, name="prep_util", &
          value=is%wrap%prep_util, defaultvalue="python3", &
          convention="NUOPC", purpose="Instance", rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      else
        is%wrap%prep_run=.false.
      endif

      ! set configuration file name
      call ESMF_AttributeGet(gcomp, name="filename", &
        value=is%wrap%pfidb_filename, defaultvalue="config_file", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! import data initialization type
      call ESMF_AttributeGet(gcomp, name="initialize_import", &
        value=attval, defaultvalue="FLD_INIT_FILLV", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      is%wrap%init_import = attval

      ! export data initialization type
      call ESMF_AttributeGet(gcomp, name="initialize_export", &
        value=attval, defaultvalue="FLD_INIT_MODEL", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      is%wrap%init_export = attval

      ! import check type
      call ESMF_AttributeGet(gcomp, name="check_import", &
        value=attval, defaultvalue="FLD_CHECK_CURRT", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      is%wrap%check_import = attval

      ! geom source type
      ! ParFlow must provide geom
      is%wrap%geom_src = GEOM_PROVIDE

      ! grid coord type
      call ESMF_AttributeGet(gcomp, name="coord_type", &
        value=attval, defaultvalue="GRD_COORD_CLMVEGTF", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      is%wrap%ctype = attval

      ! set coordinates file name
      call ESMF_AttributeGet(gcomp, name="coord_filename", &
        value=is%wrap%coord_filename, defaultvalue="drv_vegm.alluv.dat", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! number of soil layers
      call ESMF_AttributeGet(gcomp, name="number_of_soil_layers", &
        value=is%wrap%cplnz, defaultvalue=4, &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! thickness of soil layers
      allocate(is%wrap%cpldz(is%wrap%cplnz), stat=stat)
      if (ESMF_LogFoundAllocError(statusToCheck=stat, &
        msg='Allocation of soil thickness memory failed.', &
        line=__LINE__, file=__FILE__, rcToReturn=rc)) return  ! bail out
      is%wrap%cpldz = 0.0
      call ESMF_AttributeGet(gcomp, name="thickness_of_soil_layers", &
        value=attval, defaultvalue="0.1,0.3,0.6,1.0,1.0,1.0,1.0,1.0", &
        convention="NUOPC", purpose="Instance", rc=rc)
      oldidx=1
      do i=1, is%wrap%cplnz
        attval=adjustl(attval(oldidx:))
        if (len_trim(attval).lt.1) then
          call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
            msg="missing thickness_of_soil_layers", &
            line=__LINE__, file=__FILE__, rcToReturn=rc)
          return  ! bail out
        endif
        newidx=index(attval,",")
        if (newidx.eq.0) then
          newidx=len(attval)
        else
          newidx=newidx-1
        endif
        is%wrap%cpldz(i) = ESMF_UtilString2Real(attval(:newidx), rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        oldidx=newidx+2
      enddo
      if (ANY(is%wrap%cpldz.le.0)) then
        call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
          msg="thickness_of_soil_layers must be positive value", &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return  ! bail out
      endif

      ! set component input directory
      call ESMF_AttributeGet(gcomp, name="input_directory", &
        value=is%wrap%input_dir, defaultvalue=trim(cname)//"_INPUT", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      ! Get component output directory
      call ESMF_AttributeGet(gcomp, name="output_directory", &
        value=is%wrap%output_dir, defaultvalue=trim(cname)//"_OUTPUT", &
        convention="NUOPC", purpose="Instance", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      if (btest(verbosity,16)) then
        call ESMF_LogWrite(trim(cname)//": Settings",ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,I0))") trim(cname)//': ', &
          '  Verbosity                = ',verbosity
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,I0))") trim(cname)//': ', &
          '  Diagnostic               = ',diagnostic
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        attval = is%wrap%init_export
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Initialize Export        = ',trim(attval)
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        attval = is%wrap%init_import
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Initialize Import        = ',trim(attval)
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        attval = is%wrap%geom_src
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Geom Source              = ',trim(attval)
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        attval = is%wrap%ctype
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Coordinate Type          = ',trim(attval)
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Coodinates Filename      = ',is%wrap%coord_filename
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        attval = is%wrap%check_import
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Check Import             = ',trim(attval)
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,L1))") trim(cname)//': ', &
          '  Multiple Instances       = ',is%wrap%multi_instance
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,L1))") trim(cname)//': ', &
          '  Realze All Imports       = ',is%wrap%realize_all_import
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,L1))") trim(cname)//': ', &
          '  Realze All Exports       = ',is%wrap%realize_all_export
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,L1))") trim(cname)//': ', &
          '  Run Preprocessor         = ',is%wrap%prep_run
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Preprocessor Utility     = ',is%wrap%prep_util
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Preprocessor Filename    = ',is%wrap%prep_filename
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Config Filename          = ',is%wrap%pfidb_filename
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,I0))") trim(cname)//': ', &
          '  Number of Soil Layers    = ',is%wrap%cplnz
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (attval, "(I0)") is%wrap%cplnz
        write (logMsg, "(A,(A,"//trim(attval)//"(1X,F4.1)))") &
          trim(cname)//': ','  Thickness of Soil Layers = ',is%wrap%cpldz
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Input Directory         = ',is%wrap%input_dir
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Output Directory         = ',is%wrap%output_dir
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      endif

    end subroutine

  end subroutine

!..................................................................

  subroutine InitializeP1(gcomp, importState, exportState, clock, rc)
    type(ESMF_GridComp)     :: gcomp
    type(ESMF_State)        :: importState, exportState
    type(ESMF_Clock)        :: clock
    integer,intent(out)     :: rc

    ! LOCAL VARIABLES
    character(32)              :: cname
    character(*), parameter    :: rname="InitializeP1"
    integer                    :: verbosity, diagnostic
    type(type_InternalState)   :: is
    integer                    :: stat

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! change directory for multiple instances
    if (is%wrap%multi_instance) then
      if (btest(verbosity,16)) then
        call ESMF_LogWrite(trim(cname)//": Change working directory", &
          ESMF_LOGMSG_INFO)
      endif
      call change_directory(trim(cname),rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

    ! determine if component will provide or accept geom
    if (is%wrap%geom_src .eq. GEOM_PROVIDE) then
      is%wrap%transfer_offer="will provide"
    else
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="Unsupported geom type", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out      LogSetError
    end if

    call field_advertise(fieldList=pf_nuopc_fld_list, &
      importState=importState, &
      exportState=exportState, &
      transferOffer=trim(is%wrap%transfer_offer), &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    if (btest(verbosity,16)) then
      call field_advertise_log(pf_nuopc_fld_list, trim(cname), rc=rc)
    end if

    contains

    !..................................................................

    subroutine change_directory(path,rc)
      use iso_c_binding
      character(*), intent(in) :: path
      integer, intent(out)     :: rc

      rc = local_chdir(path//c_null_char)
      if (rc /= 0) then
        call ESMF_LogSetError(rcToCheck=ESMF_RC_ARG_BAD,   &
          msg="change_directory failed changing to "//trim(path), &
          CONTEXT, rcToReturn=rc)
      else
        rc = ESMF_SUCCESS
      endif

    end subroutine

  end subroutine

!..................................................................

  subroutine InitializeP3(gcomp, importState, exportState, clock, rc)
    type(ESMF_GridComp)  :: gcomp
    type(ESMF_State)     :: importState, exportState
    type(ESMF_Clock)     :: clock
    integer, intent(out) :: rc
    ! local Variables
    character(32)                  :: cname
    character(*), parameter        :: rname="InitializeP3"
    integer                        :: verbosity, diagnostic
    type(type_InternalState)       :: is
    type(ESMF_VM)                  :: vm
    integer                        :: comm
    integer                        :: petCount
    logical                        :: file_exists
    character(len=64)              :: pfidb
    integer                        :: ext
    type(ESMF_DistGrid)            :: pfdistgrid
    type(ESMF_Grid)                :: pfgrid
    character(ESMF_MAXSTR)         :: logMsg
    integer                        :: pfsubgridcnt
    integer                        :: pfnumprocs
    integer                        :: ierr

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! call preprocessor
    if (is%wrap%prep_run) then
      if (btest(verbosity,16)) then
        call ESMF_LogWrite(trim(cname)//": "//rname,ESMF_LOGMSG_INFO)
        write (logMsg, "(A,A)") trim(cname)//': ', &
          '  Calling ParFlow Preprocessor'
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        write (logMsg, "(A,(A,A))") trim(cname)//': ', &
          '  Preprocessor Filename = ',trim(is%wrap%prep_filename)
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
        call ESMF_LogFlush()
      end if
      inquire(file=trim(is%wrap%prep_filename), exist=file_exists)
      if (.not. file_exists) then
        call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
          msg="configuraiton file missing: "//trim(is%wrap%prep_filename), &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return
      end if
      call pf_preprocessor(util=is%wrap%prep_util, &
        filename=is%wrap%prep_filename, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    end if

    if (btest(verbosity,16)) then
      call ESMF_LogWrite(trim(cname)//": "//rname,ESMF_LOGMSG_INFO)
      write (logMsg, "(A,A)") trim(cname)//': ', &
        '  Calling cplparflowinit'
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,A))") trim(cname)//': ', &
        '  Config Filename = ',trim(is%wrap%pfidb_filename)
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      call ESMF_LogFlush()
    end if

    call ESMF_GridCompGet(gcomp, vm=vm, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call ESMF_VMGet(vm, petCount=petCount, mpiCommunicator=comm, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    inquire(file=trim(is%wrap%pfidb_filename), exist=file_exists)
    if (.not. file_exists) then
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="configuraiton file missing: "//trim(is%wrap%pfidb_filename), &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return
    end if
    pfidb = ESMF_UtilStringLowerCase(is%wrap%pfidb_filename, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    ext = index(pfidb, substring=".pfidb", back=.true.)
    if (ext .le. 0) ext = len(is%wrap%pfidb_filename) + 1
    ! call parflow c interface
    ! void cplparflowinit_(int  *fcom,
    !                      char *input_file,
    !                      int  *numprocs,
    !                      int  *subgridcount,
    !                      int  *nz,
    !                      int  *ierror)
    call cplparflowinit(comm, &
      trim(is%wrap%pfidb_filename(:ext-1))//c_null_char, &
      pfnumprocs, pfsubgridcnt, is%wrap%nz, ierr)
    if (ierr .ne. 0) then
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="cplparflowinit failed.", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return
    elseif (pfnumprocs .ne. petCount) then
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="pfnumprocs does not equal component petcount.", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return
    elseif (pfsubgridcnt .ne. 1) then
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="Unsupported subgrid count", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out
    endif

    pfdistgrid = distgrid_create(vm, is%wrap%nx, is%wrap%ny, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    pfgrid = grid_create(pfdistgrid, trim(cname)//"-Grid", is%wrap%ctype, &
      is%wrap%coord_filename, is%wrap%nx, is%wrap%ny, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! write grid to NetCDF file
    if (btest(diagnostic,16)) then
      call grid_write(pfgrid, trim(is%wrap%output_dir)// &
        "/diagnostic_"//trim(cname)//"_"//rname//"_grid.nc", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

    call field_init_internal(internalFB=is%wrap%pf_fields, &
      grid=pfgrid, nz=is%wrap%nz, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call field_realize(fieldList=pf_nuopc_fld_list, &
      importState=importState, exportState=exportState, &
      grid=pfgrid, num_soil_layers=is%wrap%cplnz, &
      realizeAllImport=is%wrap%realize_all_import, &
      realizeAllExport=is%wrap%realize_all_export, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call field_fill_state(importState, &
      fill_type=FLD_INIT_FILLV, &
      fillValue=ESMF_DEFAULT_VALUE, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call field_fill_state(exportState, &
      fill_type=FLD_INIT_FILLV, &
      fillValue=ESMF_DEFAULT_VALUE, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    if (btest(verbosity,16)) then
      call field_realize_log(pf_nuopc_fld_list, trim(cname), rc=rc)
    end if

    contains ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    subroutine pf_preprocessor(util,filename,rc)
      character(*), intent(in) :: util
      character(*), intent(in) :: filename
      integer, intent(out)     :: rc

      ! local variables
      type(ESMF_VM) :: vm
      integer       :: localPet

      call ESMF_GridCompGet(gcomp, vm=vm, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      call ESMF_VMGet(vm, localPet=localPet, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      if (localPet == 0) then
        call execute_command_line(trim(util)//" "//trim(filename), exitstat=rc)
        if (rc /= 0) then
          call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
            msg="", &
            line=__LINE__, file=__FILE__, rcToReturn=rc)
          return  ! bail out
        end if
      endif
      ! wait for preprocessing script to finish
      call ESMF_VMBarrier(vm, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    end subroutine

  end subroutine

!..................................................................

  subroutine InitializeP4(gcomp, importState, exportState, clock, rc)
    type(ESMF_GridComp)  :: gcomp
    type(ESMF_State)     :: importState, exportState
    type(ESMF_Clock)     :: clock
    integer, intent(out) :: rc

    ! local variables
    type(ESMF_Field)              :: field
    type(ESMF_Grid)               :: grid
    integer                       :: localDeCount
    character(80)                 :: name
    character(160)                :: msgString

    type(ESMF_DistGrid)           :: distgrid
    integer                       :: dimCount, tileCount, arbDimCount
    integer, allocatable          :: minIndexPTile(:,:), maxIndexPTile(:,:)
    integer                       :: connectionCount
    type(ESMF_DistGridConnection), allocatable :: connectionList(:)
    character(ESMF_MAXSTR)        :: transferAction
    logical                       :: regDecompFlag

    rc = ESMF_SUCCESS

  end subroutine

!..................................................................

  subroutine InitializeP5(gcomp, importState, exportState, clock, rc)
    type(ESMF_GridComp)  :: gcomp
    type(ESMF_State)     :: importState, exportState
    type(ESMF_Clock)     :: clock
    integer, intent(out) :: rc

    ! local variables
    type(ESMF_Field)              :: field
    type(ESMF_Grid)               :: grid
    integer                       :: localDeCount
    character(80)                 :: name
    character(160)                :: msgString

    type(ESMF_DistGrid)           :: distgrid
    integer                       :: dimCount, tileCount, arbDimCount
    integer, allocatable          :: minIndexPTile(:,:), maxIndexPTile(:,:)
    integer                       :: connectionCount
    type(ESMF_DistGridConnection), allocatable :: connectionList(:)
    character(ESMF_MAXSTR)        :: transferAction
    logical                       :: regDecompFlag

    rc = ESMF_SUCCESS

  end subroutine

!..................................................................

  subroutine SetClock(gcomp, rc)
    type(ESMF_GridComp)  :: gcomp
    integer, intent(out) :: rc
    ! local variables
    character(32)            :: cname
    character(*), parameter  :: rname="SetClock"
    integer                  :: verbosity, diagnostic
    type(type_InternalState) :: is

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call ESMF_TimeSet(is%wrap%pf_epoch, &
      yy=1900, mm=1, dd=1, &
       h=0,     m=0,  s=0, &
      calkindflag=ESMF_CALKIND_GREGORIAN, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

  end subroutine

!..................................................................

  subroutine DataInitialize(gcomp, rc)
    type(ESMF_GridComp)  :: gcomp
    integer, intent(out) :: rc

    ! local variables
    character(32)            :: cname
    character(*), parameter  :: rname="DataInitialize"
    integer                  :: verbosity, diagnostic
    type(type_InternalState) :: is
    integer                  :: stat
    type(ESMF_Clock)         :: modelClock
    type(ESMF_Time)          :: currTime
    type(ESMF_Time)          :: invalidTime
    character(len=32)        :: currTimeStr
    type(ESMF_State)         :: importState
    type(ESMF_State)         :: exportState
    character(len=32)        :: initTypeStr
    logical                  :: importInit
    logical                  :: exportInit
    integer(c_int)           :: totalLWidth(2,1)
    integer(c_int)           :: totalUWidth(2,1)
    integer(c_int)           :: ierr
    character(ESMF_MAXSTR)   :: logMsg

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query the component for its clock and importState
    call NUOPC_ModelGet(gcomp, &
      modelClock=modelClock, &
      importState=importState, &
      exportState=exportState, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! get the current time out of the clock
    call ESMF_ClockGet(modelClock, currTime=currTime, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call ESMF_TimeGet(currTime, timeString=currTimeStr, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! set up invalid time
    call ESMF_TimeSet(invalidTime, yy=99999999, mm=01, dd=01, &
      h=00, m=00, s=00, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    if ( is%wrap%init_import .eq. FLD_INIT_IMPORT ) then
      call ESMF_ClockGet(modelClock, currTime=currTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      importInit = NUOPC_IsAtTime(importState, time=currTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (importInit) then
        call ESMF_LogWrite( &
          trim(cname)//': '//rname//' Initialize-Data-Dependency SATISFIED!!!', &
          ESMF_LOGMSG_INFO)
      else
        call ESMF_LogWrite( &
          trim(cname)//': '//rname//' Initialize-Data-Dependency NOT YET SATISFIED!!!', &
          ESMF_LOGMSG_INFO)
      endif
    elseif ( is%wrap%init_import .eq. FLD_INIT_FILLV ) then
      call field_fill_state(importState, &
        fill_type=is%wrap%init_import, &
        fillValue=ESMF_DEFAULT_VALUE, &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      importInit = .true.
    elseif ( is%wrap%init_import .eq. FLD_INIT_DEFAULT ) then
      call field_fill_state(importState, &
        fill_type=is%wrap%init_import, &
        fieldList=pf_nuopc_fld_list, &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      importInit = .true.
    else
      initTypeStr = is%wrap%init_import
      call ESMF_LogSetError(ESMF_FAILURE, &
        msg="Import data initialize routine unknown "//trim(initTypeStr), &
        line=__LINE__,file=__FILE__,rcToReturn=rc)
      return  ! bail out
      importInit = .FALSE.
    endif

    if ( is%wrap%init_export .eq. FLD_INIT_ZERO ) then
      call field_fill_state(exportState, &
        fill_type=is%wrap%init_export, &
        rc=rc)
      call NUOPC_SetTimestamp(exportState, time=invalidTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      exportInit = .TRUE.
    elseif ( is%wrap%init_export .eq. FLD_INIT_DEFAULT ) then
      call field_fill_state(exportState, &
        fill_type=is%wrap%init_export, &
        fieldList=pf_nuopc_fld_list, &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call NUOPC_SetTimestamp(exportState, time=invalidTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      exportInit = .TRUE.
    elseif ( is%wrap%init_export .eq. FLD_INIT_FILLV ) then
      call field_fill_state(exportState, &
        fill_type=is%wrap%init_export, &
        fillValue=ESMF_DEFAULT_VALUE, &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call NUOPC_SetTimestamp(exportState, time=invalidTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      exportInit = .TRUE.
    elseif ( is%wrap%init_export .eq. FLD_INIT_FILE ) then
      call field_fill_state(exportState, &
        fill_type=is%wrap%init_export, &
        filePrefix=trim(is%wrap%input_dir)//"/restart_"//trim(cname)// &
          "_exp_"//trim(currTimeStr), &
        rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
      call NUOPC_SetTimestamp(exportState, time=currTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      exportInit = .TRUE.
    elseif ( is%wrap%init_export .eq. FLD_INIT_MODEL ) then
      ! check internal fields
      if(.not.associated(pf_pressure%ptr)) then
        call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_pressure missing", &
          line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
      endif
      if(.not.associated(pf_porosity%ptr)) then
        call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_porosity missing", &
          line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
      endif
      if(.not.associated(pf_saturation%ptr)) then
        call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_saturation missing", &
          line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
      endif
      if(.not.associated(pf_specific%ptr)) then
        call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_specific missing", &
           line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
      endif
      if(.not.associated(pf_zmult%ptr)) then
        call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_zmult missing", &
          line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
      endif

      totalLWidth = 0
      totalUWidth = 0

      if (btest(verbosity,16)) then
        call ESMF_LogWrite(trim(cname)//": "//rname,ESMF_LOGMSG_INFO)
        write (logMsg, "(A,A)") trim(cname)//': ', &
          '  Calling cplparflowexport'
        call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      endif
      ! call parflow c interface
      ! field dimensions (i,num_soil_layers,j)
      ! void cplparflowexport_(float *exp_pressure, float *exp_porosity,
      !   float *exp_saturation, float *exp_specific, float *exp_zmulit,
      !   int *num_soil_layers, int *num_cpl_layers
      !   int *ghost_size_i_lower, int *ghost_size_j_lower,
      !   int *ghost_size_i_upper, int *ghost_size_j_upper,
      !   ierror)
      call cplparflowexport(pf_pressure%ptr, &
        pf_porosity%ptr, pf_saturation%ptr, &
        pf_specific%ptr, pf_zmult%ptr, &
        is%wrap%nz, is%wrap%cplnz, &
        totalLWidth(1,1), totalLWidth(2,1), &
        totalUWidth(1,1), totalUWidth(2,1), &
        ierr)
      if (ierr .ne. 0) then
        call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
          msg="cplparflowexport failed.", &
          line=__LINE__, file=__FILE__, rcToReturn=rc)
        return
      endif

      call field_prep_export(exportState, is%wrap%nz, is%wrap%cplnz, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call NUOPC_SetTimestamp(exportState, time=currTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      exportInit = .TRUE.
    else
      initTypeStr = is%wrap%init_export
      call ESMF_LogSetError(ESMF_FAILURE, &
        msg="Export data initialize routine unknown "//trim(initTypeStr), &
        line=__LINE__,file=__FILE__,rcToReturn=rc)
      return  ! bail out
      exportInit = .FALSE.
    endif

    ! set InitializeDataComplete Attribute to "true", indicating to the
    ! generic code that all inter-model data dependencies are satisfied
    if (importInit .and. exportInit) then
      call NUOPC_CompAttributeSet(gcomp, name="InitializeDataComplete", value="true", rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

  end subroutine

!..................................................................

  subroutine CheckImport(gcomp, rc)
    type(ESMF_GridComp) :: gcomp
    integer,intent(out) :: rc

    ! local variables
    character(32)               :: cname
    character(*), parameter     :: rname="CheckImport"
    integer                     :: verbosity, diagnostic
    type(type_InternalState)    :: is
    type(ESMF_State)            :: importState
    type(ESMF_Clock)            :: modelClock
    type(ESMF_Time)             :: modelCurrTime
    type(ESMF_Time)             :: modelNextTime
    logical                     :: checkTime

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query the component for its clock and importState
    call NUOPC_ModelGet(gcomp, &
      modelClock=modelClock, &
      importState=importState, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    if ( is%wrap%check_import .eq. FLD_CHECK_CURRT ) then
      call ESMF_ClockGet(modelClock, currTime=modelCurrTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      checkTime = NUOPC_IsAtTime(importState, modelCurrTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    else if ( is%wrap%check_import .eq. FLD_CHECK_NEXTT ) then
      call ESMF_ClockGetNextTime(modelClock, nextTime=modelNextTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      checkTime = NUOPC_IsAtTime(importState, modelNextTime, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    else if ( is%wrap%check_import .eq. FLD_CHECK_NONE ) then
      checkTime = .true.
    else
      call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
        msg="Unsupported check_import in CheckImport", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out
    end if

    if (.not.checkTime) then
      call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
        msg="Import Timestamp: Import fields not at correct time. "// &
            "Try check_import=FLD_CHECK_NEXTT for sequential run.", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return  ! bail out
    endif

  end subroutine

!..................................................................

  subroutine ModelAdvance(gcomp, rc)
    type(ESMF_GridComp)  :: gcomp
    integer, intent(out) :: rc

    ! local variables
    character(32)            :: cname
    character(*), parameter  :: rname="ModelAdvance"
    integer                  :: verbosity, diagnostic
    character(len=64)        :: value
    type(type_InternalState) :: is
    type(ESMF_Clock)         :: modelClock
    type(ESMF_State)         :: importState, exportState
    type(ESMF_Time)          :: startTime, currTime
    character(len=32)        :: currTimeStr
    type(ESMF_TimeInterval)  :: elapsedTime, timeStep
    real(c_double)           :: pf_dt, pf_time
    type(forcing_flag)       :: forcType
    integer(c_int)           :: totalLWidth(2,1)
    integer(c_int)           :: totalUWidth(2,1)
    integer(c_int)           :: ierr
    character(ESMF_MAXSTR)   :: logMsg

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query the component for its clock, importState, and exportState
    call NUOPC_ModelGet(gcomp, &
      modelClock=modelClock, &
      importState=importState, &
      exportState=exportState, &
      rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! convert elapsed time and time step to hours
    call ESMF_ClockGet(modelClock, &
      startTime=startTime, currTime=currTime, &
      timeStep=timeStep, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    elapsedTime = currTime - startTime
    call ESMF_TimeIntervalGet(elapsedTime, h_r8=pf_time, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call ESMF_TimeIntervalGet(timeStep, h_r8=pf_dt, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    call ESMF_TimeGet(currTime, timeString=currTimeStr, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! check internal fields
    if(.not.associated(pf_flux%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_flux missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif
    if(.not.associated(pf_pressure%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_pressure missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif
    if(.not.associated(pf_porosity%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_porosity missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif
    if(.not.associated(pf_saturation%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_saturation missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif
    if(.not.associated(pf_specific%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_specific missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif
    if(.not.associated(pf_zmult%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_zmult missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif

    ! prepare import data
    call field_prep_import(importState, is%wrap%nz, is%wrap%cplnz, &
      is%wrap%cpldz, forcType, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    totalLWidth = 0
    totalUWidth = 0

    if (btest(verbosity,16)) then
      call ESMF_LogWrite(trim(cname)//": "//rname,ESMF_LOGMSG_INFO)
      write (logMsg, "(A,A)") trim(cname)//': ', &
        '  Calling cplparflowadvance'
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      value = forcType
      write (logMsg, "(A,(A,A))") trim(cname)//': ', &
        '  Forcing Type                  = ',trim(value)
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,F0.3))") trim(cname)//': ', &
        '  Current Time(h)               = ',real(pf_time)
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,F0.3))") trim(cname)//': ', &
        '  Time Step(h)                  = ',real(pf_dt)
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,I0))") trim(cname)//': ', &
        '  Number of Soil Layers         = ',int(is%wrap%nz)
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,I0))") trim(cname)//': ', &
        '  Number of Coupled Soil Layers = ',int(is%wrap%cplnz)
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,I0,1X,I0))") trim(cname)//': ', &
        '  Halo Sizes I                  = ',int(totalLWidth(1,1)), &
        int(totalUWidth(1,1))
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      write (logMsg, "(A,(A,I0,1X,I0))") trim(cname)//': ', &
        '  Halo Sizes J                  = ',int(totalLWidth(2,1)), &
        int(totalUWidth(2,1))
      call ESMF_LogWrite(trim(logMsg),ESMF_LOGMSG_INFO)
      call ESMF_LogFlush()
    end if

    ! write out internal fields
    if (btest(diagnostic,16)) then
      call ESMF_FieldBundleWrite(is%wrap%pf_fields, &
        fileName=trim(is%wrap%output_dir)//"/diagnostic_"//trim(cname)// &
        "_enter_internalFB_"//trim(currTimeStr)//".nc", &
        overwrite=.true., status=ESMF_FILESTATUS_REPLACE, timeslice=1, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

    ! call parflow c interface
    ! field dimensions (i,num_soil_layers,j)
    ! void cplparflowadvance_(double *current_time, double *dt,
    !   float *imp_flux,     float *exp_pressure,
    !   float *exp_porosity, float *exp_saturation,
    !   float *exp_specific, float *exp_zmulit
    !   int *num_soil_layers, int *num_cpl_layers
    !   int *ghost_size_i_lower, int *ghost_size_j_lower,
    !   int *ghost_size_i_upper, int *ghost_size_j_upper,
    !   ierror)
    call cplparflowadvance(pf_time, pf_dt, &
      pf_flux%ptr,     pf_pressure%ptr, &
      pf_porosity%ptr, pf_saturation%ptr, &
      pf_specific%ptr, pf_zmult%ptr, &
      is%wrap%nz, is%wrap%cplnz, &
      totalLWidth(1,1), totalLWidth(2,1), &
      totalUWidth(1,1), totalUWidth(2,1), &
      ierr)
    if (ierr .ne. 0) then
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="cplparflowadvance failed.", &
        line=__LINE__, file=__FILE__, rcToReturn=rc)
      return
    endif

    ! write out internal fields
    if (btest(diagnostic,16)) then
      call ESMF_FieldBundleWrite(is%wrap%pf_fields, &
        fileName=trim(is%wrap%output_dir)//"/diagnostic_"//trim(cname)// &
        "_exit_internalFB_"//trim(currTimeStr)//".nc", &
        overwrite=.true., status=ESMF_FILESTATUS_REPLACE, timeslice=1, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

    ! prepare export data
    call field_prep_export(exportState, is%wrap%nz, is%wrap%cplnz, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

  end subroutine

!..................................................................

  subroutine ModelFinalize(gcomp, rc)
    type(ESMF_GridComp)  :: gcomp
    integer, intent(out) :: rc

    ! local variables
    character(32)              :: cname
    character(*), parameter    :: rname="ModelFinalize"
    integer                    :: verbosity, diagnostic
    type(type_InternalState)   :: is
    integer                    :: stat

    rc = ESMF_SUCCESS

    ! query component for name, verbosity, and diagnostic values
    call NUOPC_CompGet(gcomp, name=cname, verbosity=verbosity, &
      diagnostic=diagnostic, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    ! query component for its internal State
    nullify(is%wrap)
    call ESMF_UserCompGetInternalState(gcomp, label_InternalState, is, rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    call field_fin_internal(internalFB=is%wrap%pf_fields, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out

    deallocate(is%wrap%cpldz, stat=stat)
    if (ESMF_LogFoundDeallocError(statusToCheck=stat, &
      msg='Deallocation of cpldz memory failed.', &
      line=__LINE__, file=__FILE__, rcToReturn=rc)) return  ! bail out

    deallocate(is%wrap, stat=stat)
    if (ESMF_LogFoundDeallocError(statusToCheck=stat, &
      msg='Deallocation of internal state memory failed.', &
      line=__LINE__, file=__FILE__, rcToReturn=rc)) return  ! bail out

  end subroutine

!------------------------------------------------------------------

end module parflow_nuopc

#ifdef SHARED_OBJECT

! External access to SetVM
subroutine SetVM(gcomp, rc)
  use ESMF
  use , only: parflow_nuopc => SetVM
  type(ESMF_GridComp) :: gcomp
  integer, intent(out) :: rc
  call SetVMModule(gcomp, rc)
  if (ESMF_STDERRORCHECK(rc)) return  ! bail out

end subroutine

! External access to SetServices
subroutine SetServices(gcomp, rc)
  use ESMF
  use parflow_nuopc, only: SetServicesModule => SetServices
  type(ESMF_GridComp) :: gcomp
  integer, intent(out) :: rc
  call SetServicesModule(gcomp, rc)
  if (ESMF_STDERRORCHECK(rc)) return  ! bail out

end subroutine

#endif
