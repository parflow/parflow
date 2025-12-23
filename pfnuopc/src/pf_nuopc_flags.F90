#include "pf_nuopc_macros.h"

module parflow_nuopc_flags

  use ESMF, only: ESMF_UtilStringUpperCase, ESMF_SUCCESS

  implicit none

  private

  type grid_coord_flag
    sequence
    private
      integer :: ctype
  end type grid_coord_flag

  type(grid_coord_flag),  parameter ::       &
    GRD_COORD_ERROR = grid_coord_flag(-1),   &
    GRD_COORD_NONE  = grid_coord_flag(0),    &
    GRD_COORD_CLMVEGTF = grid_coord_flag(1), &
    GRD_COORD_CARTESIAN = grid_coord_flag(2)

  type field_init_flag
    sequence
    private
      integer :: init
  end type field_init_flag

  type(field_init_flag), parameter ::       &
    FLD_INIT_ERROR   = field_init_flag(-1), &
    FLD_INIT_ZERO    = field_init_flag(0),  &
    FLD_INIT_FILLV   = field_init_flag(1),  &
    FLD_INIT_DEFAULT = field_init_flag(2),  &
    FLD_INIT_FILE    = field_init_flag(3),  &
    FLD_INIT_MODEL   = field_init_flag(4),  &
    FLD_INIT_IMPORT  = field_init_flag(5)

  type field_check_flag
    sequence
    private
      integer :: check
  end type field_check_flag

  type(field_check_flag), parameter ::      &
    FLD_CHECK_ERROR = field_check_flag(-1), &
    FLD_CHECK_CURRT = field_check_flag(0),  &
    FLD_CHECK_NEXTT = field_check_flag(1),  &
    FLD_CHECK_NONE  = field_check_flag(2)

  type geom_src_flag
    sequence
    private
      integer :: src
  end type geom_src_flag

  type(geom_src_flag), parameter ::   &
    GEOM_ERROR   = geom_src_flag(-1), &
    GEOM_PROVIDE = geom_src_flag(0),  &
    GEOM_ACCEPT  = geom_src_flag(1)

  type forcing_flag
    sequence
    private
      integer :: opt
  end type forcing_flag

  type(forcing_flag),  parameter ::       &
    FORCING_ERROR     = forcing_flag(-1), &
    FORCING_WTRFLX3D  = forcing_flag(0),  &
    FORCING_WTRFLX2D  = forcing_flag(1),  &
    FORCING_COMPOSITE = forcing_flag(2)

  public grid_coord_flag
  public field_init_flag
  public field_check_flag
  public geom_src_flag
  public forcing_flag
  public GRD_COORD_ERROR
  public GRD_COORD_NONE
  public GRD_COORD_CLMVEGTF
  public GRD_COORD_CARTESIAN
  public FLD_INIT_ERROR
  public FLD_INIT_ZERO
  public FLD_INIT_FILLV
  public FLD_INIT_DEFAULT
  public FLD_INIT_FILE
  public FLD_INIT_MODEL
  public FLD_INIT_IMPORT
  public FLD_CHECK_ERROR
  public FLD_CHECK_CURRT
  public FLD_CHECK_NEXTT
  public FLD_CHECK_NONE
  public GEOM_ERROR
  public GEOM_PROVIDE
  public GEOM_ACCEPT
  public FORCING_ERROR
  public FORCING_WTRFLX3D
  public FORCING_WTRFLX2D
  public FORCING_COMPOSITE

  public operator(==), assignment(=)

  interface operator (==)
    module procedure grid_ctype_eq
    module procedure field_init_eq
    module procedure field_check_eq
    module procedure geom_src_eq
    module procedure forcing_eq
  end interface

  interface assignment (=)
    module procedure grid_ctype_toString
    module procedure grid_ctype_frString
    module procedure field_init_toString
    module procedure field_init_frString
    module procedure field_check_toString
    module procedure field_check_frString
    module procedure geom_src_toString
    module procedure geom_src_frString
    module procedure forcing_toString
    module procedure forcing_frString
  end interface

  !-----------------------------------------------------------------------------
  contains
  !-----------------------------------------------------------------------------

  function grid_ctype_eq(val1, val2)
    logical grid_ctype_eq
    type(grid_coord_flag), intent(in) :: val1, val2
    grid_ctype_eq = (val1%ctype == val2%ctype)
  end function grid_ctype_eq

  !-----------------------------------------------------------------------------

  subroutine grid_ctype_toString(string, val)
    character(len=*), intent(out) :: string
    type(grid_coord_flag), intent(in) :: val
    if (val == GRD_COORD_NONE) then
      write(string,'(a)') 'GRD_COORD_NONE'
    elseif (val == GRD_COORD_CLMVEGTF) then
      write(string,'(a)') 'GRD_COORD_CLMVEGTF'
    elseif (val == GRD_COORD_CARTESIAN) then
      write(string,'(a)') 'GRD_COORD_CARTESIAN'
    else
      write(string,'(a)') 'GRD_COORD_ERROR'
    endif
  end subroutine grid_ctype_toString

  !-----------------------------------------------------------------------------

  subroutine grid_ctype_frString(val, string)
    type(grid_coord_flag), intent(out) :: val
    character(len=*), intent(in) :: string
    character(len=32) :: ustring
    integer :: rc
    ustring = ESMF_UtilStringUpperCase(string, rc=rc)
    if (rc .ne. ESMF_SUCCESS) then
      val = GRD_COORD_ERROR
    elseif (ustring .eq. 'GRD_COORD_NONE') then
      val = GRD_COORD_NONE
    elseif (ustring .eq. 'GRD_COORD_CLMVEGTF') then
      val = GRD_COORD_CLMVEGTF
    elseif (ustring .eq. 'GRD_COORD_CARTESIAN') then
      val = GRD_COORD_CARTESIAN
    else
      val = GRD_COORD_ERROR
    endif
  end subroutine grid_ctype_frString

  !-----------------------------------------------------------------------------

  function field_init_eq(val1, val2)
    logical field_init_eq
    type(field_init_flag), intent(in) :: val1, val2
    field_init_eq = (val1%init == val2%init)
  end function field_init_eq

  !-----------------------------------------------------------------------------

  subroutine field_init_toString(string, val)
    character(len=*), intent(out) :: string
    type(field_init_flag), intent(in) :: val
    if (val == FLD_INIT_ZERO) then
      write(string,'(a)') 'FLD_INIT_ZERO'
    elseif (val == FLD_INIT_FILLV) then
      write(string,'(a)') 'FLD_INIT_FILLV'
    elseif (val == FLD_INIT_DEFAULT) then
      write(string,'(a)') 'FLD_INIT_DEFAULT'
    elseif (val == FLD_INIT_FILE) then
      write(string,'(a)') 'FLD_INIT_FILE'
    elseif (val == FLD_INIT_MODEL) then
      write(string,'(a)') 'FLD_INIT_MODEL'
    elseif (val == FLD_INIT_IMPORT) then
      write(string,'(a)') 'FLD_INIT_IMPORT'
    else
      write(string,'(a)') 'FLD_INIT_ERROR'
    endif
  end subroutine field_init_toString

  !-----------------------------------------------------------------------------

  subroutine field_init_frString(val, string)
    type(field_init_flag), intent(out) :: val
    character(len=*), intent(in) :: string
    character(len=16) :: ustring
    integer :: rc
    ustring = ESMF_UtilStringUpperCase(string, rc=rc)
    if (rc .ne. ESMF_SUCCESS) then
      val = FLD_INIT_ERROR
    elseif (ustring .eq. 'FLD_INIT_ZERO') then
      val = FLD_INIT_ZERO
    elseif (ustring .eq. 'FLD_INIT_FILLV') then
      val = FLD_INIT_FILLV
    elseif (ustring .eq. 'FLD_INIT_MISSING') then
      val = FLD_INIT_FILLV
    elseif (ustring .eq. 'FLD_INIT_DEFAULT') then
      val = FLD_INIT_DEFAULT
    elseif (ustring .eq. 'FLD_INIT_FILE') then
      val = FLD_INIT_FILE
    elseif (ustring .eq. 'FLD_INIT_MODEL') then
      val = FLD_INIT_MODEL
    elseif (ustring .eq. 'FLD_INIT_IMPORT') then
      val = FLD_INIT_IMPORT
    else
      val = FLD_INIT_ERROR
    endif
  end subroutine field_init_frString

  !-----------------------------------------------------------------------------

  function field_check_eq(val1, val2)
    logical field_check_eq
    type(field_check_flag), intent(in) :: val1, val2
    field_check_eq = (val1%check == val2%check)
  end function field_check_eq

  !-----------------------------------------------------------------------------

  subroutine field_check_toString(string, val)
    character(len=*), intent(out) :: string
    type(field_check_flag), intent(in) :: val
    if (val == FLD_CHECK_CURRT) then
      write(string,'(a)') 'FLD_CHECK_CURRT'
    elseif (val == FLD_CHECK_NEXTT) then
      write(string,'(a)') 'FLD_CHECK_NEXTT'
    elseif (val == FLD_CHECK_NONE) then
      write(string,'(a)') 'FLD_CHECK_NONE'
    else
      write(string,'(a)') 'FLD_CHECK_ERROR'
    endif
  end subroutine field_check_toString

  !-----------------------------------------------------------------------------

  subroutine field_check_frString(val, string)
    type(field_check_flag), intent(out) :: val
    character(len=*), intent(in) :: string
    character(len=16) :: ustring
    integer :: rc
    ustring = ESMF_UtilStringUpperCase(string, rc=rc)
    if (rc .ne. ESMF_SUCCESS) then
      val = FLD_CHECK_ERROR
    elseif (ustring .eq. 'FLD_CHECK_CURRT') then
      val = FLD_CHECK_CURRT
    elseif (ustring .eq. 'FLD_CHECK_NEXTT') then
      val = FLD_CHECK_NEXTT
    elseif (ustring .eq. 'FLD_CHECK_NONE') then
      val = FLD_CHECK_NONE
    else
      val = FLD_CHECK_ERROR
    endif
  end subroutine field_check_frString

  !-----------------------------------------------------------------------------

  function geom_src_eq(val1, val2)
    logical geom_src_eq
    type(geom_src_flag), intent(in) :: val1, val2
    geom_src_eq = (val1%src == val2%src)
  end function geom_src_eq

  !-----------------------------------------------------------------------------

  subroutine geom_src_toString(string, val)
    character(len=*), intent(out) :: string
    type(geom_src_flag), intent(in) :: val
    if (val == GEOM_PROVIDE) then
      write(string,'(a)') 'GEOM_PROVIDE'
    elseif (val == GEOM_ACCEPT) then
      write(string,'(a)') 'GEOM_ACCEPT'
    else
      write(string,'(a)') 'GEOM_ERROR'
    endif
  end subroutine geom_src_toString

  !-----------------------------------------------------------------------------

  subroutine geom_src_frString(val, string)
    type(geom_src_flag), intent(out) :: val
    character(len=*), intent(in) :: string
    character(len=32) :: ustring
    integer :: rc
    ustring = ESMF_UtilStringUpperCase(string, rc=rc)
    if (rc .ne. ESMF_SUCCESS) then
      val = GEOM_ERROR
    elseif (ustring .eq. 'GEOM_PROVIDE') then
      val = GEOM_PROVIDE
    elseif (ustring .eq. 'GEOM_ACCEPT') then
      val = GEOM_ACCEPT
    else
      val = GEOM_ERROR
    endif
  end subroutine geom_src_frString

  !-----------------------------------------------------------------------------

  function forcing_eq(val1, val2)
    logical forcing_eq
    type(forcing_flag), intent(in) :: val1, val2
    forcing_eq = (val1%opt == val2%opt)
  end function forcing_eq

  !-----------------------------------------------------------------------------

  subroutine forcing_toString(string, val)
    character(len=*), intent(out) :: string
    type(forcing_flag), intent(in) :: val
    if (val == FORCING_WTRFLX3D) then
      write(string,'(a)') 'FORCING_WTRFLX3D'
    elseif (val == FORCING_WTRFLX2D) then
      write(string,'(a)') 'FORCING_WTRFLX2D'
    elseif (val == FORCING_COMPOSITE) then
      write(string,'(a)') 'FORCING_COMPOSITE'
    else
      write(string,'(a)') 'FORCING_ERROR'
    endif
  end subroutine forcing_toString

  !-----------------------------------------------------------------------------

  subroutine forcing_frString(val, string)
    type(forcing_flag), intent(out) :: val
    character(len=*), intent(in) :: string
    character(len=32) :: ustring
    integer :: rc
    ustring = ESMF_UtilStringUpperCase(string, rc=rc)
    if (rc .ne. ESMF_SUCCESS) then
      val = FORCING_ERROR
    elseif (ustring .eq. 'FORCING_WTRFLX3D') then
      val = FORCING_WTRFLX3D
    elseif (ustring .eq. 'FORCING_WTRFLX2D') then
      val = FORCING_WTRFLX2D
    elseif (ustring .eq. 'FORCING_COMPOSITE') then
      val = FORCING_COMPOSITE
    else
      val = FORCING_ERROR
    endif
  end subroutine forcing_frString

  !-----------------------------------------------------------------------------

end module
