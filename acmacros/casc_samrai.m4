AC_DEFUN([CASC_SUPPORT_SAMRAI],[

dnl Support using SAMRAI by
dnl - Setting the variables samrai_INCLUDES and samrai_LIBS.
dnl - Setting variables to help in coordinating Doxygen
dnl   with SAMRAI's Doxygen documentation.
dnl
dnl User will specify where SAMRAI was installed (--with-samrai)
dnl
dnl This macro takes these arguments:
dnl 1: Path to auxiliary directory, relative to the top source directory.
dnl    Usually, this path is "aux".  Utility scripts useful to
dnl    the SAMRAI application is placed here.
dnl
# Begin macro $0

# Set samrai_PREFIX to the SAMRAI install directory.

AC_MSG_CHECKING([for compiled SAMRAI library])

AC_ARG_WITH(samrai,
[  --with-samrai=PATH
                        Specify where SAMRAI is installed.],
if test "${with_samrai}" = yes; then
  AC_MSG_ERROR(SAMRAI install directory omitted.
Please specify using --with-samrai=STRING)
elif test "${with_samrai}" = no; then
  unset samrai_PREFIX
  AC_MSG_RESULT([no])
else
  samrai_PREFIX=$with_samrai
  AC_MSG_RESULT([$samrai_PREFIX])
fi,
  if test "${samrai_PREFIX+yes}" = yes; then
    AC_MSG_RESULT([$samrai_PREFIX])
  else
    AC_MSG_RESULT([no])
  fi
)

# samrai_PREFIX should be set to where SAMRAI is installed.
# If it is set to blank, we expect that the preprocessor -I flag
# and the load -L flag are not needed, but we still issue the
# load -l flag.

if test "${samrai_PREFIX+yes}" = yes; then

if test -z "${samrai_PREFIX}"; then
  AC_MSG_WARN([SAMRAI install directory omitted.
I will expect that the SAMRAI library has been installed
where the compiler will automatically look for it.])
fi

# If user specified the SAMRAI prefix, sanity check it.
if test -n "$samrai_PREFIX"; then
  if test ! -d "$samrai_PREFIX"; then
    AC_MSG_WARN([SAMRAI installation directory ($samrai_PREFIX) is not a directory.])
  fi
  if test ! -d "$samrai_PREFIX/include"; then
    AC_MSG_WARN([SAMRAI installation directory ($samrai_PREFIX/include) is not a directory.])
  fi
  if test ! -d "$samrai_PREFIX/lib"; then
    AC_MSG_WARN([SAMRAI installation directory ($samrai_PREFIX/lib) is not a directory.])
  fi
fi

if test "${samrai_PREFIX+yes}" = yes; then
AC_DEFINE([HAVE_SAMRAI], [1], ["Configured with SAMRAI."])
fi

if test -n "$samrai_PREFIX"; then
   samrai_INCLUDES="-I$samrai_PREFIX/include"
fi

# Determine the samrai library names.

samrai_libs_ls=' libSAMRAI_appu.a libSAMRAI_algs.a libSAMRAI_solv.a libSAMRAI_geom.a libSAMRAI_mesh.a libSAMRAI_math.a libSAMRAI_pdat.a libSAMRAI_xfer.a libSAMRAI_hier.a libSAMRAI_tbox.a '

# Build up SAMRAI_LIBS string using library names.
if test -n "$samrai_libs_ls"; then
  for i in $samrai_libs_ls; do
    j=`echo $i | sed 's/lib\(SAMRAI_....\)\.a/\1/'`
    samrai_LIBS="$samrai_LIBS -l$j"
  done
fi
if test -n "$samrai_PREFIX"; then
   samrai_LIBS="-L${samrai_PREFIX}/lib ${samrai_LIBS}"
fi

fi # samrai_PREFIX

dnl end The new stuff

# End macro $0
])


