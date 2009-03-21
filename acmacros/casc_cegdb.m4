dnl Define a macro for supporting CEGDB

AC_DEFUN([CASC_SUPPORT_CEGDB],[

# Begin CASC_SUPPORT_CEGDB
# Defines cegdb_PREFIX cegdb_INCLUDES and cegdb_LIBS if with-cegdb is specified.
AC_ARG_WITH(cegdb,
[  --with-cegdb[=PATH]        Use CEGDB and optionally specify 
                            where CEGDB is installed.],
, with_cegdb=no)

case "$with_cegdb" in
  no)
    AC_MSG_NOTICE([configuring without CEGDB support])
    : Do nothing
  ;;
  yes)
    # CEGDB install path was not specified.
    # Look in a couple of standard locations to probe if 
    # CEGDB header files are there.
    AC_MSG_CHECKING([for CEGDB installation])
    for dir in /usr /usr/local; do
      if test -f ${dir}/include/cegdb.h; then
        cegdb_PREFIX=${dir}
        break
      fi
    done
    AC_MSG_RESULT([$cegdb_PREFIX])
  ;;
  *)
    # CEGDB install path was specified.
    AC_MSG_CHECKING([for CEGDB installation])
    cegdb_PREFIX=$with_cegdb
    cegdb_INCLUDES="-I${cegdb_PREFIX}/include"
    if test -f ${cegdb_PREFIX}/include/cegdb.h; then
        AC_MSG_RESULT([$cegdb_PREFIX])
    else
        AC_MSG_RESULT([$cegdb_PREFIX])
        AC_MSG_ERROR([CEGDB not found in $with_cegdb])
    fi
  ;;
esac

# Determine which CEGDB library is built
if test "${cegdb_PREFIX+set}" = set; then
   AC_MSG_CHECKING([for CEGDB library])
   if test -f ${cegdb_PREFIX}/lib/libcegdb.a; then
      cegdb_LIBS='-lcegdb'
      AC_MSG_RESULT([using $cegdb_LIBS])
   else
      AC_MSG_RESULT([using $cegdb_LIBS])
      AC_MSG_ERROR([Could not fine cegdb library in $cegdb_PREFIX])
   fi

   cegdb_LIBS="-L${cegdb_PREFIX}/lib ${cegdb_LIBS}"
fi

# END CASC_SUPPORT_CEGDB

])dnl End definition of CASC_SUPPORT_CEGDB
