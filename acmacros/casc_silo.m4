dnl Define a macro for supporting SILO

AC_DEFUN([CASC_SUPPORT_SILO],[

# Begin CASC_SUPPORT_SILO
# Defines silo_PREFIX silo_INCLUDES and silo_LIBS if with-silo is specified.
AC_ARG_WITH(silo,
[  --with-silo[=PATH]        Use SILO and optionally specify 
                          where SILO is installed.],
, with_silo=no)

case "$with_silo" in
  no)
    AC_MSG_NOTICE([configuring without SILO support])
    : Do nothing
  ;;
  yes)
    # SILO install path was not specified.
    # Look in a couple of standard locations to probe if 
    # SILO header files are there.
    AC_MSG_CHECKING([for SILO installation])
    for dir in /usr /usr/local; do
      if test -f ${dir}/include/silo.h; then
        silo_PREFIX=${dir}
        break
      fi
    done
    AC_MSG_RESULT([$silo_PREFIX])
  ;;
  *)
    # SILO install path was specified.
    AC_MSG_CHECKING([for SILO installation])
    silo_PREFIX=$with_silo
    silo_INCLUDES="-I${silo_PREFIX}/include"
    if test -f ${silo_PREFIX}/include/silo.h; then
        AC_MSG_RESULT([$silo_PREFIX])
    else
        AC_MSG_RESULT([$silo_PREFIX])
        AC_MSG_ERROR([SILO not found in $with_silo])
    fi
  ;;
esac

# Determine which SILO library is built
if test "${silo_PREFIX+set}" = set; then
   AC_MSG_CHECKING([for SILO library])
   if test -f ${silo_PREFIX}/lib/libsiloxx.a; then
      silo_LIBS='-lsiloxx'
      AC_MSG_RESULT([using $silo_LIBS])
   elif test -f ${silo_PREFIX}/lib/libsiloh5.a; then
      silo_LIBS='-lsiloh5'
      AC_MSG_RESULT([using $silo_LIBS])
   elif test -f ${silo_PREFIX}/lib/x86_64-linux-gnu/libsiloh5.so; then
      # This is for Ubuntu  
      silo_LIBS='-lsiloh5'
      AC_MSG_RESULT([using $silo_LIBS])
   elif test -f ${silo_PREFIX}/lib/libsilo.a; then
      silo_LIBS='-lsilo'
      AC_MSG_RESULT([using $silo_LIBS])
   else
      AC_MSG_RESULT([using $silo_LIBS])
      AC_MSG_ERROR([Could not fine silo library in $silo_PREFIX])
   fi

   silo_LIBS="-L${silo_PREFIX}/lib ${silo_LIBS}"
fi

# END CASC_SUPPORT_SILO

])dnl End definition of CASC_SUPPORT_SILO
