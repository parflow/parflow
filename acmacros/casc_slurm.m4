dnl Define a macro for supporting SLURM

AC_DEFUN([CASC_SUPPORT_SLURM],[

# Begin CASC_SUPPORT_SLURM
# Defines slurm_PREFIX slurm_INCLUDES and slurm_LIBS if with-slurm is specified.
AC_ARG_WITH(slurm,
[  --with-slurm[=PATH]        Use SLURM and optionally specify 
                          where SLURM is installed.],
, with_slurm=no)

case "$with_slurm" in
  no)
    AC_MSG_NOTICE([configuring without SLURM support])
    : Do nothing
  ;;
  yes)
    # SLURM install path was not specified.
    # Look in a couple of standard locations to probe if 
    # SLURM header files are there.
    AC_MSG_CHECKING([for SLURM installation])
    for dir in /opt/slurm/default /usr /usr/local; do
      if test -f ${dir}/include/slurm/slurm.h; then
        slurm_PREFIX=${dir}
        break
      fi
    done
    AC_MSG_RESULT([$slurm_PREFIX])
  ;;
  *)
    # SLURM install path was specified.
    AC_MSG_CHECKING([for SLURM installation])
    slurm_PREFIX=$with_slurm
    slurm_INCLUDES="-I${slurm_PREFIX}/include"
    if test -f ${slurm_PREFIX}/include/slurm/slurm.h; then
        AC_MSG_RESULT([$slurm_PREFIX])
    else
        AC_MSG_RESULT([$slurm_PREFIX])
        AC_MSG_ERROR([SLURM not found in $with_slurm])
    fi
  ;;
esac

# Determine which SLURM library is built
if test "${slurm_PREFIX+set}" = set; then
   AC_MSG_CHECKING([for SLURM library])
   if test -f ${slurm_PREFIX}/lib64/libslurm.so; then
      slurm_LIBS='-lslurm'
      AC_MSG_RESULT([using $slurm_LIBS])
      slurm_LIBS="-L${slurm_PREFIX}/lib ${slurm_LIBS}"	
   elif test -f ${slurm_PREFIX}/lib64/libslurm.a; then
      slurm_LIBS='-lslurm'
      AC_MSG_RESULT([using $slurm_LIBS])
      slurm_LIBS="-L${slurm_PREFIX}/lib ${slurm_LIBS}"	
   elif test -f ${slurm_PREFIX}/lib/libslurm.so; then
      slurm_LIBS='-lslurm'
      AC_MSG_RESULT([using $slurm_LIBS])
      slurm_LIBS="-L${slurm_PREFIX}/lib ${slurm_LIBS}"	
   elif test -f ${slurm_PREFIX}/lib/libslurm.a; then
      slurm_LIBS='-lslurm'
      AC_MSG_RESULT([using $slurm_LIBS])
   else
      AC_MSG_RESULT([using $slurm_LIBS])
      AC_MSG_ERROR([Could not fine slurm library in $slurm_PREFIX])
   fi
fi

# END CASC_SUPPORT_SLURM

])dnl End definition of CASC_SUPPORT_SLURM
