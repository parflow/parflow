#include "pf_nuopc_macros.h"

module parflow_nuopc_fields

  use ESMF
  use NUOPC
  use parflow_nuopc_flags
  use iso_c_binding, only: c_null_char, c_int, c_double, c_float

  implicit none

  private

  type pf_fld_type
    character(len=64)           :: fname      = "dummy" ! state name
    character(len=64)           :: units      = "-"     ! units
    type(ESMF_Field), pointer   :: efld       => null()
    real(ESMF_KIND_R4), pointer :: ptr(:,:,:) => null()
  endtype pf_fld_type

  ! internal fields
  type(pf_fld_type) :: pf_flux = &
    pf_fld_type(fname="PF_FLUX      ", units="1 h-1")
  type(pf_fld_type) :: pf_porosity = &
    pf_fld_type(fname="PF_POROSITY  ", units="-")
  type(pf_fld_type) :: pf_pressure = &
    pf_fld_type(fname="PF_PRESSURE  ", units="m")
  type(pf_fld_type) :: pf_saturation = &
    pf_fld_type(fname="PF_SATURATION", units="-")
  type(pf_fld_type) :: pf_specific = &
    pf_fld_type(fname="PF_SPECIFIC  ", units="m3")
  type(pf_fld_type) :: pf_zmult = &
    pf_fld_type(fname="PF_ZMULT     ", units="m")

  type pf_nuopc_fld_type
    sequence
    character(len=64)           :: sd_name    = "dummy" ! standard name
    character(len=64)           :: st_name    = "dummy" ! state name
    character(len=64)           :: units      = "-"     ! units
    logical*8                   :: layers     = .FALSE. ! layered field
    logical                     :: ad_import  = .FALSE. ! advertise import
    logical                     :: ad_export  = .FALSE. ! advertise export
    logical                     :: rl_import  = .FALSE. ! realize import
    logical                     :: rl_export  = .FALSE. ! realize export
    real(ESMF_KIND_R8)          :: vl_default = ESMF_DEFAULT_VALUE ! default value
  end type pf_nuopc_fld_type

  ! external field list
  type(pf_nuopc_fld_type),target,dimension(22) :: pf_nuopc_fld_list = (/     &
    pf_nuopc_fld_type("total_water_flux                        ", &
      "FLUX      ", "kg m-2 s-1",  .TRUE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("total_water_flux_layer_1                ", &
      "FLUX1     ", "kg m-2 s-1", .FALSE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("total_water_flux_layer_2                ", &
      "FLUX2     ", "kg m-2 s-1", .FALSE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("total_water_flux_layer_3                ", &
      "FLUX3     ", "kg m-2 s-1", .FALSE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("total_water_flux_layer_4                ", &
      "FLUX4     ", "kg m-2 s-1", .FALSE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("precip_drip                             ", &
      "PCPDRP    ", "kg m-2 s-1", .FALSE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("bare_soil_evaporation                   ", &
      "EDIR      ", "W m-2     ", .FALSE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("vegetation_transpiration                ", &
      "ET        ", "W m-2     ",  .TRUE.,  .TRUE., .FALSE.), &
    pf_nuopc_fld_type("porosity                                ", &
      "POROSITY  ", "-         ",  .TRUE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("pressure                                ", &
      "PRESSURE  ", "m         ",  .TRUE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("saturation                              ", &
      "SATURATION", "-         ",  .TRUE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("ground_water_storage                    ", &
      "GWS       ", "-         ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("soil_moisture_fraction                  ", &
      "SMOIS     ", "-         ",  .TRUE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("soil_moisture_fraction_layer_1          ", &
      "SMOIS1     ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("soil_moisture_fraction_layer_2          ", &
      "SMOIS2     ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("soil_moisture_fraction_layer_3          ", &
      "SMOIS3     ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("soil_moisture_fraction_layer_4          ", &
      "SMOIS4     ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("liquid_fraction_of_soil_moisture        ", &
      "SH2O       ", "-        ",  .TRUE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("liquid_fraction_of_soil_moisture_layer_1", &
      "SH2O1      ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("liquid_fraction_of_soil_moisture_layer_2", &
      "SH2O2      ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("liquid_fraction_of_soil_moisture_layer_3", &
      "SH2O3      ", "-        ", .FALSE., .FALSE.,  .TRUE.), &
    pf_nuopc_fld_type("liquid_fraction_of_soil_moisture_layer_4", &
      "SH2O4      ", "-        ", .FALSE., .FALSE.,  .TRUE.) /)

  integer(ESMF_KIND_I4), pointer :: fld_mask(:,:) => null()

  public pf_flux
  public pf_porosity
  public pf_pressure
  public pf_saturation
  public pf_specific
  public pf_zmult
  public pf_nuopc_fld_list
  public field_init_internal
  public field_fin_internal
  public field_advertise
  public field_realize
  public field_advertise_log
  public field_realize_log
  public field_fill_state
  public field_prep_import
  public field_prep_export

  !-----------------------------------------------------------------------------
  contains
  !-----------------------------------------------------------------------------

  subroutine field_dictionary_add(fieldList, rc)
    type(pf_nuopc_fld_type), intent(in) :: fieldList(:)
    integer, intent(out) :: rc
    ! local variables
    integer :: n
    logical :: isPresent

    rc = ESMF_SUCCESS

    do n=lbound(fieldList,1),ubound(fieldList,1)
      isPresent = NUOPC_FieldDictionaryHasEntry( &
        fieldList(n)%sd_name, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if (.not.isPresent) then
        call NUOPC_FieldDictionaryAddEntry( &
          StandardName=trim(fieldList(n)%sd_name), &
          canonicalUnits=trim(fieldList(n)%units), &
          rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      end if
    end do

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_init_internal(internalFB, grid, nz, rc)
    type(ESMF_FieldBundle), intent(inout) :: internalFB
    type(ESMF_Grid), intent(in)           :: grid
    integer, intent(in)                   :: nz
    integer, intent(out)                  :: rc
    ! local variables
    logical :: isCreated

    rc = ESMF_SUCCESS

    ! create internal fields
    if (associated(pf_flux%efld)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_CREATE, msg="pf_flux exists", &
        line=__LINE__,file=__FILE__,rcToReturn=rc); return  ! bail out
    else
      allocate(pf_flux%efld)
      pf_flux%efld=field_create_layers(grid=grid, layers=nz, &
        name=pf_flux%fname, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(pf_flux%efld, farrayPtr=pf_flux%ptr, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldFill(pf_flux%efld, dataFillScheme="const", &
        const1=ESMF_DEFAULT_VALUE, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
    endif
    if (associated(pf_porosity%efld)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_CREATE, msg="pf_porosity exists", &
        line=__LINE__,file=__FILE__,rcToReturn=rc); return  ! bail out
    else
      allocate(pf_porosity%efld)
      pf_porosity%efld=field_create_layers(grid=grid, layers=nz, &
        name=pf_porosity%fname, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(pf_porosity%efld, farrayPtr=pf_porosity%ptr, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldFill(pf_porosity%efld, dataFillScheme="const", &
        const1=ESMF_DEFAULT_VALUE, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
    endif
    if (associated(pf_pressure%efld)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_CREATE, msg="pf_pressure exists", &
        line=__LINE__,file=__FILE__,rcToReturn=rc); return  ! bail out
    else
      allocate(pf_pressure%efld)
      pf_pressure%efld=field_create_layers(grid=grid, layers=nz, &
        name=pf_pressure%fname, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(pf_pressure%efld, farrayPtr=pf_pressure%ptr, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldFill(pf_pressure%efld, dataFillScheme="const", &
        const1=ESMF_DEFAULT_VALUE, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
    endif
    if (associated(pf_saturation%efld)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_CREATE, msg="pf_saturation exists", &
        line=__LINE__,file=__FILE__,rcToReturn=rc); return  ! bail out
    else
      allocate(pf_saturation%efld)
      pf_saturation%efld=field_create_layers(grid=grid, layers=nz, &
        name=pf_saturation%fname, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(pf_saturation%efld, farrayPtr=pf_saturation%ptr, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldFill(pf_saturation%efld, dataFillScheme="const", &
        const1=ESMF_DEFAULT_VALUE, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
    endif
    if (associated(pf_specific%efld)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_CREATE, msg="pf_specific exists", &
        line=__LINE__,file=__FILE__,rcToReturn=rc); return  ! bail out
    else
      allocate(pf_specific%efld)
      pf_specific%efld=field_create_layers(grid=grid, layers=nz, &
        name=pf_specific%fname, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(pf_specific%efld, farrayPtr=pf_specific%ptr, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldFill(pf_specific%efld, dataFillScheme="const", &
        const1=ESMF_DEFAULT_VALUE, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
    endif
    if (associated(pf_zmult%efld)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_CREATE, msg="pf_zmult exists", &
        line=__LINE__,file=__FILE__,rcToReturn=rc); return  ! bail out
    else
      allocate(pf_zmult%efld)
      pf_zmult%efld=field_create_layers(grid=grid, layers=nz, &
        name=pf_zmult%fname, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(pf_zmult%efld, farrayPtr=pf_zmult%ptr, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldFill(pf_zmult%efld, dataFillScheme="const", &
        const1=ESMF_DEFAULT_VALUE, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return
    endif

    ! store mask
    call ESMF_GridGetItem(grid, itemflag=ESMF_GRIDITEM_MASK, &
      farrayPtr=fld_mask, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return

    ! add fields to internal field bundle
    isCreated = ESMF_FieldBundleIsCreated(internalFB, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    if (.not. isCreated) then
      internalFB = ESMF_FieldBundleCreate(name="PF_INTERNAL", &
        fieldList=(/ pf_flux%efld, pf_porosity%efld, pf_pressure%efld, &
        pf_saturation%efld, pf_specific%efld, pf_zmult%efld /), rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    else
      call ESMF_FieldBundleAdd(internalFB, fieldList=(/ pf_flux%efld, &
        pf_porosity%efld, pf_pressure%efld, pf_saturation%efld, &
        pf_specific%efld, pf_zmult%efld /), rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_fin_internal(internalFB, rc)
    type(ESMF_FieldBundle), intent(inout) :: internalFB
    integer, intent(out)                  :: rc
    ! local variables
    logical :: isCreated

    rc = ESMF_SUCCESS

    ! destroy internal fields
    if (associated(pf_flux%efld)) then
      call ESMF_FieldDestroy(pf_flux%efld, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      deallocate(pf_flux%efld)
    endif
    if (associated(pf_porosity%efld)) then
      call ESMF_FieldDestroy(pf_porosity%efld, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      deallocate(pf_porosity%efld)
    endif
    if (associated(pf_pressure%efld)) then
      call ESMF_FieldDestroy(pf_pressure%efld, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      deallocate(pf_pressure%efld)
    endif
    if (associated(pf_saturation%efld)) then
      call ESMF_FieldDestroy(pf_saturation%efld, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      deallocate(pf_saturation%efld)
    endif
    if (associated(pf_specific%efld)) then
      call ESMF_FieldDestroy(pf_specific%efld, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      deallocate(pf_specific%efld)
    endif
    if (associated(pf_zmult%efld)) then
      call ESMF_FieldDestroy(pf_zmult%efld, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      deallocate(pf_zmult%efld)
    endif

    ! destory internal field bundle
    isCreated = ESMF_FieldBundleIsCreated(internalFB, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    if (isCreated) then
      call ESMF_FieldBundleDestroy(internalFB, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    endif

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_realize(fieldList, importState, exportState, grid, &
  num_soil_layers, realizeAllImport, realizeAllExport, rc)
    type(pf_nuopc_fld_type), intent(inout) :: fieldList(:)
    type(ESMF_State), intent(inout)        :: importState
    type(ESMF_State), intent(inout)        :: exportState
    type(ESMF_Grid), intent(in)            :: grid
    integer, intent(in)                    :: num_soil_layers
    logical, intent(in)                    :: realizeAllImport
    logical, intent(in)                    :: realizeAllExport
    integer, intent(out)                   :: rc
    ! local variables
    integer :: n
    logical :: realizeImport
    logical :: realizeExport
    type(ESMF_Field) :: field_import
    type(ESMF_Field) :: field_export
    real(ESMF_KIND_FIELD), pointer :: ptr_import(:,:,:)

    rc = ESMF_SUCCESS

    do n=lbound(fieldList,1),ubound(fieldList,1)

      ! check realize import
      if (fieldList(n)%ad_import) then
        if (realizeAllImport) then
          realizeImport = .true.
        else
          realizeImport = NUOPC_IsConnected(importState, &
            fieldName=trim(fieldList(n)%st_name),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      else
        realizeImport = .false.
      end if
      ! check realize export
      if (fieldList(n)%ad_export) then
        if (realizeAllExport) then
          realizeExport = .true.
        else
          realizeExport = NUOPC_IsConnected(exportState, &
            fieldName=trim(fieldList(n)%st_name),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      else
        realizeExport = .false.
      end if
      ! create import field
      if ( realizeImport ) then
        if (fieldList(n)%layers) then
          field_import=field_create_layers(grid=grid, &
            layers=num_soil_layers, &
            name=fieldList(n)%st_name, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        else
          field_import=field_create(grid=grid, &
            name=fieldList(n)%st_name, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
        call NUOPC_Realize(importState, field=field_import, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        fieldList(n)%rl_import = .true.
      else
        call ESMF_StateRemove(importState, (/fieldList(n)%st_name/), &
          relaxedflag=.true., rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        fieldList(n)%rl_import = .false.
      end if
      ! create export field
      if( realizeExport ) then
        if (fieldList(n)%layers) then
          field_export=field_create_layers(grid=grid, &
            layers=num_soil_layers, &
            name=fieldList(n)%st_name, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        else
          field_export=field_create(grid=grid, &
            name=fieldList(n)%st_name, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
        call NUOPC_Realize(exportState, field=field_export, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        fieldList(n)%rl_export = .true.
      else
        call ESMF_StateRemove(exportState, (/fieldList(n)%st_name/), &
          relaxedflag=.true., rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        fieldList(n)%rl_export = .false.
      end if

    end do

  end subroutine

  !-----------------------------------------------------------------------------

  function field_create(grid, name, rc) result(field)
    type(ESMF_Grid), intent(in)         :: grid
    character(*), intent(in)            :: name
    integer, intent(out)                :: rc
    ! return value
    type(ESMF_Field)                    :: field
    ! local variables

    rc = ESMF_SUCCESS

    field = ESMF_FieldCreate(grid=grid, &
      typekind=ESMF_TYPEKIND_FIELD, &
      name=name, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
  end function

  !-----------------------------------------------------------------------------

  function field_create_layers(grid, layers, name, rc) result(field)
    type(ESMF_Grid), intent(in)         :: grid
    integer, intent(in)                 :: layers
    character(*), intent(in)            :: name
    integer, intent(out)                :: rc
    ! return value
    type(ESMF_Field)                    :: field
    ! local variables

    rc = ESMF_SUCCESS

    field = ESMF_FieldCreate(grid=grid, &
      typekind=ESMF_TYPEKIND_FIELD, &
      gridToFieldMap=(/1,3/), &
      ungriddedLBound=(/1/), &
      ungriddedUBound=(/layers/), &
      name=name, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
  end function

  !-----------------------------------------------------------------------------

  subroutine field_advertise(fieldList, importState, exportState, &
  transferOffer, rc)
    type(pf_nuopc_fld_type), intent(in) :: fieldList(:)
    type(ESMF_State), intent(inout)     :: importState
    type(ESMF_State), intent(inout)     :: exportState
    character(*), intent(in),optional   :: transferOffer
    integer, intent(out)                :: rc
    ! local variables
    integer :: n

    rc = ESMF_SUCCESS

    do n=lbound(fieldList,1),ubound(fieldList,1)
      if (fieldList(n)%ad_import) then
        call NUOPC_Advertise(importState, &
          StandardName=fieldList(n)%sd_name, &
          Units=fieldList(n)%units, &
          TransferOfferGeomObject=transferOffer, &
          name=fieldList(n)%st_name, &
          rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      end if
      if (fieldList(n)%ad_export) then
        call NUOPC_Advertise(exportState, &
          StandardName=fieldList(n)%sd_name, &
          Units=fieldList(n)%units, &
          TransferOfferGeomObject=transferOffer, &
          name=fieldList(n)%st_name, &
          rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      end if
    end do

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_advertise_log(fieldList, cname, rc)
    type(pf_nuopc_fld_type), intent(in) :: fieldList(:)
    character(*), intent(in)            :: cname
    integer, intent(out)                :: rc
    ! local variables
    integer                    :: cntImp
    integer                    :: cntExp
    integer                    :: n
    character(32)              :: label
    character(ESMF_MAXSTR)     :: logMsg

    rc = ESMF_SUCCESS

    label = trim(cname)

    ! count advertised import and export fields
    cntImp = 0
    cntExp = 0
    do n = lbound(fieldList,1), ubound(fieldList,1)
      if (fieldList(n)%ad_import) cntImp = cntImp + 1
      if (fieldList(n)%ad_export) cntExp = cntExp + 1
    enddo

    ! log advertised import fields
    write(logMsg,'(a,a,i0,a)') trim(label)//': ', &
      'List of advertised import fields(',cntImp,'):'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    write(logMsg,'(a,a5,a,a16,a,a)') trim(label)//': ', &
      'index',' ','name',' ','standardName'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    cntImp = 0
    do n=lbound(fieldList,1), ubound(fieldList,1)
      if (.NOT.fieldList(n)%ad_import) cycle
      cntImp = cntImp + 1
      write(logMsg,'(a,i5,a,a16,a,a)') trim(label)//': ', &
        cntImp,' ',trim(fieldList(n)%st_name), &
        ' ',trim(fieldList(n)%sd_name)
      call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    enddo

    ! log advertised export fields
    write(logMsg,'(a,a,i0,a)') trim(label)//': ', &
      'List of advertised export fields(',cntExp,'):'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    write(logMsg,'(a,a5,a,a16,a,a)') trim(label)//': ', &
      'index',' ','name',' ','standardName'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    cntExp = 0
    do n=lbound(fieldList,1), ubound(fieldList,1)
      if (.NOT.fieldList(n)%ad_export) cycle
      cntExp = cntExp + 1
      write(logMsg,'(a,i5,a,a16,a,a)') trim(label)//': ', &
        cntExp,' ',trim(fieldList(n)%st_name), &
        ' ',trim(fieldList(n)%sd_name)
      call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    enddo

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_realize_log(fieldList, cname, rc)
    type(pf_nuopc_fld_type), intent(in) :: fieldList(:)
    character(*), intent(in)            :: cname
    integer, intent(out)                :: rc
    ! local variables
    integer                    :: cntImp
    integer                    :: cntExp
    integer                    :: n
    character(32)              :: label
    character(ESMF_MAXSTR)     :: logMsg

    rc = ESMF_SUCCESS

    label = trim(cname)

    ! count realized import and export fields
    cntImp = 0
    cntExp = 0
    do n = lbound(fieldList,1), ubound(fieldList,1)
      if (fieldList(n)%rl_import) cntImp = cntImp + 1
      if (fieldList(n)%rl_export) cntExp = cntExp + 1
    enddo

    ! log realized import fields
    write(logMsg,'(a,a,i0,a)') trim(label)//': ', &
      'List of realized import fields(',cntImp,'):'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    write(logMsg,'(a,a5,a,a16,a,a)') trim(label)//': ', &
      'index',' ','name',' ','standardName'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    cntImp = 0
    do n=lbound(fieldList,1), ubound(fieldList,1)
      if (.NOT.fieldList(n)%rl_import) cycle
      cntImp = cntImp + 1
      write(logMsg,'(a,i5,a,a16,a,a)') trim(label)//': ', &
        cntImp,' ',trim(fieldList(n)%st_name), &
        ' ',trim(fieldList(n)%sd_name)
      call ESMF_LogWrite(trim(LogMsg), ESMF_LOGMSG_INFO)
    enddo

    ! log realized export fields
    write(logMsg,'(a,a,i0,a)') trim(label)//': ', &
      'List of realized export fields(',cntExp,'):'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    write(logMsg,'(a,a5,a,a16,a,a)') trim(label)//': ', &
      'index',' ','name',' ','standardName'
    call ESMF_LogWrite(trim(logMsg), ESMF_LOGMSG_INFO)
    cntExp = 0
    do n=lbound(fieldList,1), ubound(fieldList,1)
      if (.NOT.fieldList(n)%rl_export) cycle
      cntExp = cntExp + 1
      write(logMsg,'(a,i5,a,a16,a,a)') trim(label)//': ', &
        cntExp,' ',trim(fieldList(n)%st_name), &
        ' ',trim(fieldList(n)%sd_name)
      call ESMF_LogWrite(trim(LogMsg), ESMF_LOGMSG_INFO)
    enddo

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_find_standardname(fieldList, standardName, location, &
  defaultValue, rc)
    type(pf_nuopc_fld_type), intent(in)     :: fieldList(:)
    character(len=64), intent(in)           :: standardName
    integer, intent(out), optional          :: location
    real(ESMF_KIND_R8),intent(out),optional :: defaultValue
    integer, intent(out)                    :: rc
    ! local variables
    integer :: n

    rc = ESMF_RC_NOT_FOUND

    if (present(location)) location = lbound(fieldList,1) - 1
    if (present(defaultValue)) defaultValue = ESMF_DEFAULT_VALUE

    do n=lbound(fieldList,1),ubound(fieldList,1)
      if (fieldList(n)%sd_name .eq. standardName) then
        if (present(location)) location = n
        if (present(defaultValue)) defaultValue = fieldList(n)%vl_default
        rc = ESMF_SUCCESS
        return
      end if
    end do

    if (ESMF_LogFoundError(rcToCheck=rc, &
      msg="Field not found in fieldList "//trim(standardName), &
      line=__LINE__, &
      file=__FILE__)) &
      return  ! bail out

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_find_statename(fieldList, stateName, location, &
  defaultValue, rc)
    type(pf_nuopc_fld_type), intent(in)     :: fieldList(:)
    character(len=64), intent(in)           :: stateName
    integer, intent(out), optional          :: location
    real(ESMF_KIND_R8),intent(out),optional :: defaultValue
    integer, intent(out)                    :: rc
    ! local variables
    integer :: n

    rc = ESMF_RC_NOT_FOUND

    if (present(location)) location = lbound(fieldList,1) - 1
    if (present(defaultValue)) defaultValue = ESMF_DEFAULT_VALUE

    do n=lbound(fieldList,1),ubound(fieldList,1)
      if (fieldList(n)%st_name .eq. stateName) then
        if (present(location)) location = n
        if (present(defaultValue)) defaultValue = fieldList(n)%vl_default
        rc = ESMF_SUCCESS
        return
      end if
    end do

    if (ESMF_LogFoundError(rcToCheck=rc, &
      msg="Field not found in fieldList "//trim(stateName), &
      line=__LINE__, &
      file=__FILE__)) &
      return  ! bail out

  end subroutine

  !-----------------------------------------------------------------------------

  subroutine field_fill_state(state, fill_type, fieldList, fillValue, &
  filePrefix, rc)
    type(ESMF_State), intent(inout)               :: state
    type(field_init_flag), intent(in)             :: fill_type
    type(pf_nuopc_fld_type), intent(in), optional :: fieldList(:)
    real(ESMF_KIND_R8), intent(in), optional      :: fillValue
    character(len=*), intent(in), optional        :: filePrefix
    integer, intent(out)                          :: rc
    ! local variables
    integer                                :: n
    integer                                :: itemCount
    character(len=64),allocatable          :: itemNameList(:)
    type(ESMF_StateItem_Flag), allocatable :: itemTypeList(:)
    type(ESMF_Field)                       :: field
    character(len=64)                      :: fldName
    real(ESMF_KIND_R8)                     :: defaultValue
    integer                                :: stat

    rc = ESMF_SUCCESS

    call ESMF_StateGet(state,itemCount=itemCount, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return ! bail out

    allocate(itemNameList(itemCount), stat=stat)
    if (ESMF_LogFoundAllocError(statusToCheck=stat, &
      msg="Allocation of state item name memory failed.", &
      line=__LINE__, file=__FILE__)) return  ! bail out
    allocate(itemTypeList(itemCount), stat=stat)
    if (ESMF_LogFoundAllocError(statusToCheck=stat, &
      msg="Allocation of state item type memory failed.", &
      line=__LINE__, file=__FILE__)) return  ! bail out

    call ESMF_StateGet(state,itemNameList=itemNameList, &
      itemTypeList=itemTypeList,rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return

    if ( fill_type .eq. FLD_INIT_ZERO ) then
      do n=1, itemCount
        if ( itemTypeList(n) == ESMF_STATEITEM_FIELD) then
          call ESMF_StateGet(state, field=field, &
            itemName=itemNameList(n),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call ESMF_FieldFill(field, dataFillScheme="const", &
            const1=0.0_ESMF_KIND_R8, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call NUOPC_SetAttribute(field, name="Updated", value="true", rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      enddo
    else if ( fill_type .eq. FLD_INIT_FILLV ) then
      if (.not. present(fillValue)) then
        call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
          msg="Missing fillValue for FLD_INIT_FILLV.", &
          line=__LINE__,file=__FILE__,rcToReturn=rc)
        return  ! bail out
      end if
      do n=1, itemCount
        if ( itemTypeList(n) == ESMF_STATEITEM_FIELD) then
          call ESMF_StateGet(state, field=field, &
            itemName=itemNameList(n),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call ESMF_FieldFill(field, dataFillScheme="const", &
            const1=fillValue, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call NUOPC_SetAttribute(field, name="Updated", value="true", rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      enddo
    else if ( fill_type .eq. FLD_INIT_DEFAULT ) then
      if (.not. present(fieldList)) then
        call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
          msg="Missing fieldList for FLD_INIT_DEFAULT.", &
          line=__LINE__,file=__FILE__,rcToReturn=rc)
        return  ! bail out
      end if
      do n=1, itemCount
        if ( itemTypeList(n) == ESMF_STATEITEM_FIELD) then
          call ESMF_StateGet(state, field=field, &
            itemName=itemNameList(n),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call field_find_statename(fieldList, itemNameList(n), &
            defaultValue=defaultValue, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call ESMF_FieldFill(field, dataFillScheme="const", &
            const1=defaultValue, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return
          call NUOPC_SetAttribute(field, name="Updated", value="true", rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      enddo
    else if ( fill_type .eq. FLD_INIT_FILE ) then
      if (.not. present(filePrefix)) then
        call ESMF_LogSetError(ESMF_RC_ARG_BAD, &
          msg="Missing filePrefix for FLD_INIT_FILE.", &
          line=__LINE__,file=__FILE__,rcToReturn=rc)
        return  ! bail out
      end if
      do n=1, itemCount
        if ( itemTypeList(n) == ESMF_STATEITEM_FIELD) then
          call ESMF_StateGet(state,field=field, &
            itemName=itemNameList(n),rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return ! bail out
          call NUOPC_GetAttribute(field, name="StandardName", &
            value=fldName, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return ! bail out
          call ESMF_FieldRead(field, variableName=trim(fldName), &
            fileName=trim(filePrefix)//"_"//trim(itemNameList(n))//".nc", &
            iofmt=ESMF_IOFMT_NETCDF, rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return ! bail out
          call NUOPC_SetAttribute(field, name="Updated", value="true", rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      enddo
    else
      call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
        msg="Unsupported fill_type for field_fill_state", &
        line=__LINE__,file=__FILE__,rcToReturn=rc)
      return  ! bail out
    end if

    deallocate(itemNameList)
    deallocate(itemTypeList)

  end subroutine field_fill_state

  !-----------------------------------------------------------------------------

  subroutine field_prep_import(importState, nz, cplnz, cpldz, forcType, rc)
    type(ESMF_State), intent(in)    :: importState
    integer, intent(in)             :: nz
    integer, intent(in)             :: cplnz
    real, allocatable, intent(in)   :: cpldz(:)
    type(forcing_flag), intent(out) :: forcType
    integer, intent(out)            :: rc

    ! local variables
    integer :: s_flx, s_flx1, s_flx2, s_flx3, s_flx4
    type(ESMF_Field) :: fld_imp_flux
    type(ESMF_Field) :: fld_imp_flux1
    type(ESMF_Field) :: fld_imp_flux2
    type(ESMF_Field) :: fld_imp_flux3
    type(ESMF_Field) :: fld_imp_flux4
    type(ESMF_Field) :: fld_imp_pcpdrp
    type(ESMF_Field) :: fld_imp_edir
    type(ESMF_Field) :: fld_imp_et
    real(c_float), pointer :: ptr_imp_flux(:, :, :)
    real(c_float), pointer :: ptr_imp_flux1(:, :)
    real(c_float), pointer :: ptr_imp_flux2(:, :)
    real(c_float), pointer :: ptr_imp_flux3(:, :)
    real(c_float), pointer :: ptr_imp_flux4(:, :)
    real(c_float), pointer :: ptr_imp_pcpdrp(:, :)
    real(c_float), pointer :: ptr_imp_edir(:, :)
    real(c_float), pointer :: ptr_imp_et(:, :, :)
    type(ESMF_StateItem_Flag) :: itemType
    integer          :: i
    real, parameter  :: LVH2O = 2.501E+6 ! heat of vaporization
    real, parameter  :: CNVMH = (3600d0/1000d0) ! convert mm/s to m/h

    rc = ESMF_SUCCESS
    forcType = FORCING_ERROR

    ! check internal fields
    if(.not.associated(pf_flux%ptr)) then
      call ESMF_LogSetError(ESMF_RC_OBJ_INIT, msg="pf_flux missing", &
        line=__LINE__,file=__FILE__,rcToReturn=rc);  return  ! bail out
    endif

    ! search import for total water flux
    call ESMF_StateGet(importState, itemSearch="FLUX", &
      itemCount=s_flx, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    if (s_flx .gt. 0) then
      ! query import state for pf fields
      call ESMF_StateGet(importState, itemName="FLUX", &
        field=fld_imp_flux, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_FieldGet(fld_imp_flux, farrayPtr=ptr_imp_flux, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      do i=1, cplnz
        pf_flux%ptr(:,i,:) = ptr_imp_flux(:,i,:) * CNVMH / cpldz(i)
      enddo
      forcType = FORCING_WTRFLX3D
    else
      ! search import for total water flux (layers 1-4)
      call ESMF_StateGet(importState, itemSearch="FLUX1", &
        itemCount=s_flx1, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_StateGet(importState, itemSearch="FLUX2", &
        itemCount=s_flx2, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_StateGet(importState, itemSearch="FLUX3", &
        itemCount=s_flx3, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      call ESMF_StateGet(importState, itemSearch="FLUX4", &
        itemCount=s_flx4, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out
      if ((s_flx1.gt.0) .and. (s_flx2.gt.0) .and. &
          (s_flx3.gt.0) .and. (s_flx4.gt.0)) then
        if (cplnz.ne.4) then
          call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
            msg="Unsupported number of coupled soil layers.", &
            line=__LINE__,file=__FILE__,rcToReturn=rc)
          return  ! bail out
        endif
        ! query import state for pf fields
        call ESMF_StateGet(importState, itemName="FLUX1", &
          field=fld_imp_flux1, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_flux1, farrayPtr=ptr_imp_flux1, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_StateGet(importState, itemName="FLUX2", &
          field=fld_imp_flux2, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_flux2, farrayPtr=ptr_imp_flux2, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_StateGet(importState, itemName="FLUX3", &
          field=fld_imp_flux3, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_flux3, farrayPtr=ptr_imp_flux3, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_StateGet(importState, itemName="FLUX4", &
          field=fld_imp_flux4, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_flux4, farrayPtr=ptr_imp_flux4, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        pf_flux%ptr(:,1,:) = ptr_imp_flux1 * CNVMH / cpldz(1)
        pf_flux%ptr(:,2,:) = ptr_imp_flux2 * CNVMH / cpldz(2)
        pf_flux%ptr(:,3,:) = ptr_imp_flux3 * CNVMH / cpldz(3)
        pf_flux%ptr(:,4,:) = ptr_imp_flux4 * CNVMH / cpldz(4)
        forcType = FORCING_WTRFLX2D
      else ! calculate total water flux
        ! query import state for pf fields
        call ESMF_StateGet(importState, itemName="PCPDRP", &
          field=fld_imp_pcpdrp, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_pcpdrp, farrayPtr=ptr_imp_pcpdrp, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_StateGet(importState, itemName="EDIR", &
          field=fld_imp_edir, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_edir, farrayPtr=ptr_imp_edir, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_StateGet(importState, itemName="ET", &
          field=fld_imp_et, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        call ESMF_FieldGet(fld_imp_et, farrayPtr=ptr_imp_et, rc=rc)
        if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        ! calculate total water flux
        pf_flux%ptr(:,1,:) = (- ( (ptr_imp_edir(:,:) + ptr_imp_et(:,1,:)) &
                             / LVH2O ) - ptr_imp_pcpdrp(:,:) ) * &
                             CNVMH / cpldz(1)
        do i=2, cplnz
          pf_flux%ptr(:,i,:) = - (ptr_imp_et(:,i,:)/LVH2O) * CNVMH / cpldz(i)
        enddo
        forcType = FORCING_COMPOSITE
      endif
    endif
  end subroutine field_prep_import

  !-----------------------------------------------------------------------------

  subroutine field_prep_export(exportState, nz, cplnz, rc)
    type(ESMF_State), intent(inout) :: exportState
    integer, intent(in)             :: nz
    integer, intent(in)             :: cplnz
    integer, intent(out)            :: rc

    ! local variables
    type(ESMF_Field) :: fld_export
    real(c_float), pointer :: ptr_export2d(:, :)
    real(c_float), pointer :: ptr_export3d(:, :, :)
    integer                               :: stat
    integer                               :: itemCount
    integer                               :: iIndex
    character(len=64),allocatable         :: itemNameList(:)
    type(ESMF_StateItem_Flag),allocatable :: itemTypeList(:)
    integer                               :: i, j
    integer                               :: totalLBound(2)
    integer                               :: totalUBound(2)
    character(ESMF_MAXSTR)                :: logMsg

    rc = ESMF_SUCCESS

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

    call ESMF_StateGet(exportState, itemCount=itemCount, rc=rc)
    if (ESMF_STDERRORCHECK(rc)) return  ! bail out
    if (itemCount > 0 ) then

      allocate(itemNameList(itemCount), itemTypeList(itemCount), stat=stat)
      if (ESMF_LogFoundAllocError(statusToCheck=stat, &
        msg="Allocation of item list memory failed.", &
        CONTEXT, rcToReturn=rc)) return  ! bail out
      call ESMF_StateGet(exportState, itemNameList=itemNameList, &
        itemTypeList=itemTypeList, rc=rc)
      if (ESMF_STDERRORCHECK(rc)) return  ! bail out

      do iIndex=1, itemCount
        if (itemTypeList(iIndex) == ESMF_STATEITEM_FIELD) then
          call ESMF_StateGet(exportState, field=fld_export, &
            itemName=itemNameList(iIndex), rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
          select case (itemNameList(iIndex))
            case ('PRESSURE')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export3d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              do i=1, cplnz
                where (fld_mask(:,:) == 1)
                  ptr_export3d(:,i,:)=pf_pressure%ptr(:,i,:)
                elsewhere
                  ptr_export3d(:,i,:)=ESMF_DEFAULT_VALUE
                endwhere
              enddo
            case ('POROSITY')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export3d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              do i=1, cplnz
                where (fld_mask(:,:) == 1)
                  ptr_export3d(:,i,:)=pf_porosity%ptr(:,i,:)
                elsewhere
                  ptr_export3d(:,i,:)=ESMF_DEFAULT_VALUE
                endwhere
              enddo
            case ('SATURATION')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export3d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              do i=1, cplnz
                where (fld_mask(:,:) == 1)
                  ptr_export3d(:,i,:)=pf_saturation%ptr(:,i,:)
                elsewhere
                  ptr_export3d(:,i,:)=ESMF_DEFAULT_VALUE
                endwhere
              enddo
            case ('GWS')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export2d, &
                totalLBound=totalLBound, totalUBound=totalUBound, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              do j=totalLBound(2), totalUBound(2)
              do i=totalLBound(1), totalUBound(1)
                if (fld_mask(i,j) == 1) then
                  ptr_export2d(i,j)=sum(((pf_porosity%ptr(i,cplnz+1:nz,j) * &
                                          pf_saturation%ptr(i,cplnz+1:nz,j)) + &
                                         (pf_pressure%ptr(i,cplnz+1:nz,j) * &
                                          pf_saturation%ptr(i,cplnz+1:nz,j) * &
                                          pf_specific%ptr(i,cplnz+1:nz,j))) * &
                                        pf_zmult%ptr(i,cplnz+1:nz,j)) * &
                                    real(1000,ESMF_KIND_R4)
                else
                  ptr_export2d(i,j)=ESMF_DEFAULT_VALUE
                endif
              enddo
              enddo
            case ('SMOIS','SH2O')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export3d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              do i=1, cplnz
                where (fld_mask(:,:) == 1)
                  ptr_export3d(:,i,:)=pf_saturation%ptr(:,i,:) * &
                    pf_porosity%ptr(:,i,:)
                elsewhere
                  ptr_export3d(:,i,:)=ESMF_DEFAULT_VALUE
                endwhere
              enddo
            case ('SMOIS1','SH2O1')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export2d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              where (fld_mask(:,:) == 1)
                ptr_export2d=pf_saturation%ptr(:,1,:) * pf_porosity%ptr(:,1,:)
              elsewhere
                ptr_export2d(:,:)=ESMF_DEFAULT_VALUE
              endwhere
            case ('SMOIS2','SH2O2')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export2d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              where (fld_mask(:,:) == 1)
                ptr_export2d=pf_saturation%ptr(:,2,:) * pf_porosity%ptr(:,2,:)
              elsewhere
                ptr_export2d(:,:)=ESMF_DEFAULT_VALUE
              endwhere
            case ('SMOIS3','SH2O3')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export2d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              where (fld_mask(:,:) == 1)
                ptr_export2d=pf_saturation%ptr(:,3,:) * pf_porosity%ptr(:,3,:)
              elsewhere
                ptr_export2d(:,:)=ESMF_DEFAULT_VALUE
              endwhere
            case ('SMOIS4','SH2O4')
              call ESMF_FieldGet(fld_export, farrayPtr=ptr_export2d, rc=rc)
              if (ESMF_STDERRORCHECK(rc)) return  ! bail out
              where (fld_mask(:,:) == 1)
                ptr_export2d=pf_saturation%ptr(:,4,:) * pf_porosity%ptr(:,4,:)
              elsewhere
                ptr_export2d(:,:)=ESMF_DEFAULT_VALUE
              endwhere
            case default
              call ESMF_LogSetError(ESMF_RC_NOT_IMPL, &
                msg="Unsupported export field: "//trim(itemNameList(iIndex)), &
                line=__LINE__,file=__FILE__,rcToReturn=rc)
              return  ! bail out
          endselect
          call NUOPC_SetAttribute(fld_export, name="Updated", &
            value="true", rc=rc)
          if (ESMF_STDERRORCHECK(rc)) return  ! bail out
        endif
      enddo

      deallocate(itemNameList)
      deallocate(itemTypeList)

    endif

  end subroutine field_prep_export

  !-----------------------------------------------------------------------------

end module parflow_nuopc_fields
