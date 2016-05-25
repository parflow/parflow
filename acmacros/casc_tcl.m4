dnl Define a macro for supporting TCL

AC_DEFUN([CASC_SUPPORT_TCL],[

# Begin CASC_SUPPORT_TCL
# Defines tcl_PREFIX tcl_INCLUDES and tcl_LIBS if with-tcl is specified.
AC_ARG_WITH(tcl,
[ --with-tcl[=PATH]  Use TCL and optionally specify where TCL is installed.],
, with_tcl=no)

case "$with_tcl" in
  no|yes)
    # TCL install path was not specified.
    # Look in a couple of standard locations to probe if 
    # TCL header files are there.
    AC_MSG_CHECKING([for TCL installation])
    for dir in /usr /usr/local; do
      if test -f ${dir}/include/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include"
        break
      elif test -f ${dir}/include/tcl8.4/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include/tcl8.4"
        break
      elif test -f ${dir}/include/tcl8.5/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include/tcl8.5"
        break
      elif test -f ${dir}/include/tcl8.6/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include/tcl8.6"
        break
      else
        AC_MSG_ERROR([TCL not found in /usr or /usr/local])
      fi
    done
    AC_MSG_RESULT([$tcl_PREFIX])
  ;;
  *)
    # TCL install path was specified.
    AC_MSG_CHECKING([for TCL installation])
    for dir in $with_tcl; do
      if test -f ${dir}/include/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include"
        break
      elif test -f ${dir}/include/tcl8.4/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include/tcl8.4"
        break
      elif test -f ${dir}/include/tcl8.5/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include/tcl8.5"
        break
      elif test -f ${dir}/include/tcl8.6/tcl.h; then
        tcl_PREFIX=${dir}
	tcl_INCLUDES="-I${tcl_PREFIX}/include/tcl8.6"
        break
      else
        AC_MSG_RESULT([$tcl_PREFIX])
        AC_MSG_ERROR([TCL not found in $with_tcl])
      fi
    done
  ;;
esac

# Determine which TCL library is built
if test "${tcl_PREFIX+set}" = set; then
   AC_MSG_CHECKING([for TCL library])

   if test -f ${tcl_PREFIX}/lib64/libtcl8.6.so; then
      tcl_LIBS='-ltcl8.6'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib64/libtcl8.5.so; then
      tcl_LIBS='-ltcl8.5'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib64/libtcl8.4.so; then
      tcl_LIBS='-ltcl8.4'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib64/libtcl.so; then
      tcl_LIBS='-ltcl'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.6.so; then
      tcl_LIBS='-ltcl8.6'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.5.so; then
      tcl_LIBS='-ltcl8.5'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.4.so; then
      tcl_LIBS='-ltcl8.4'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl.so; then
      tcl_LIBS='-ltcl'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.6.dylib; then
      tcl_LIBS='-ltcl8.6'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.5.dylib; then
      tcl_LIBS='-ltcl8.5'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.4.dylib; then
      tcl_LIBS='-ltcl8.4'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl.dylib; then
      tcl_LIBS='-ltcl'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.6.a; then
      tcl_LIBS='-ltcl8.6'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.5.a; then
      tcl_LIBS='-ltcl8.5'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl8.4.a; then
      tcl_LIBS='-ltcl8.4'
      AC_MSG_RESULT([using $tcl_LIBS])
   elif test -f ${tcl_PREFIX}/lib/libtcl.a; then
      tcl_LIBS='-ltcl'
      AC_MSG_RESULT([using $tcl_LIBS])
   else
      AC_MSG_RESULT([could not find a tcl library...assuming not needed])
   fi
   
   if test -n "${tcl_LIBS}"; then
      tcl_LIBS="-L${tcl_PREFIX}/lib ${tcl_LIBS}"
   else
      tcl_LIBS=""
   fi
fi

# END CASC_SUPPORT_TCL

])dnl End definition of CASC_SUPPORT_TCL
