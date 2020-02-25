dnl Define macros for supporting HYPRE.

dnl Support hypre libraries by setting the variables
dnl hypre_PREFIX, hypre_INCLUDES, and hypre_LIBS.
dnl hypre_MPI
AC_DEFUN([CASC_SUPPORT_HYPRE],[

# Begin CASC_SUPPORT_SILO
# Defines hypre_PREFIX hypre_INCLUDES and hypre_LIBS if with-hypre is specified.
AC_ARG_WITH(hypre,
[  --with-hypre[=PATH]       Use HYPRE and optionally specify 
                          where HYPRE is installed.],
, with_hypre=no)

case "$with_hypre" in
  no)
    AC_MSG_NOTICE([configuring without HYPRE support])
    : Do nothing
  ;;
  yes)
    # HYPRE install path was not specified.
    # Look in a couple of standard locations to probe if 
    # HYPRE header files are there.
    AC_MSG_CHECKING([for HYPRE installation])
    for dir in /usr /usr/local; do
      if test -f ${dir}/include/HYPRE.h; then
        hypre_PREFIX=${dir}
        break
      fi
    done
    AC_MSG_RESULT([$hypre_PREFIX])
  ;;
  *)
    # HYPRE install path was specified.
    AC_MSG_CHECKING([for HYPRE installation])
    hypre_PREFIX=$with_hypre
  ;;
esac

if test "${hypre_PREFIX+set}" = set; then

    hypre_INCLUDES="-I${hypre_PREFIX}/include"

    #
    # Check for HYPRE libraries
    # 	
    hypre_LIBS="-L${hypre_PREFIX}/lib"

    HYPRE_OLD_LDFLAGS=$LDFLAGS
    HYPRE_OLD_CPPFLAGS=$LDFLAGS
    LDFLAGS="$LDFLAGS -L${hypre_PREFIX}/lib"
    CPPFLAGS="$CPPFLAGS -I${hypre_PREFIX}/include"

    if test "$with_mpi" = 'yes' ; then
       LDFLAGS="$LDFLAGS $MPILIBDIRS $MPILIBS "
       CPPFLAGS="$CPPFLAGS $MPIINCLUDE"
    fi 

    AC_LANG_SAVE
    AC_LANG_C

    AC_CHECK_LIB(HYPRE, HYPRE_StructPFMGSolve, [hypre_libhypre=yes], [hypre_libhypre=no])
    if test "$hypre_libhypre" = "yes"
    then
	hypre_LIBS="$hypre_LIBS -lHYPRE"
    else
        # Following two checks were needed on Ubuntu distributions.   libHYPRE.so was empty.
	AC_CHECK_LIB(HYPRE_struct_ls, HYPRE_StructPFMGSolve, [hypre_libhypre=yes], [hypre_libhypre=no])
	if test "$hypre_libhypre" = "yes"
	then
	   hypre_LIBS="$hypre_LIBS -lHYPRE_struct_ls"

	   AC_CHECK_LIB(HYPRE_struct_mv,hypre_StructMatvecCreate, [hypre_libhypre=yes], [hypre_libhypre=no])
	   if test "$hypre_libhypre" = "yes"
	   then
	      hypre_LIBS="$hypre_LIBS -lHYPRE_struct_mv"	
	   fi
	else
	   with_hypre='no'
	   hypre_LIBS=''
	fi

    fi
    AC_LANG_RESTORE

    #
    # Check if parallel version of HYPRE is being used
    #
    save_cppflags=$CPPFLAGS
    # Add hypre include flags to cpp so we can examine its header file.
    CPPFLAGS="$hypre_INCLUDES $CPPFLAGS"

    # Check if HYPRE header is ok.
    AC_CHECK_HEADER(HYPRE_config.h,[],AC_MSG_ERROR([HYPRE not found in $with_hypre]))

    # Check if HYPRE was compiled with parallelism.
    AC_MSG_CHECKING(if hypre is parallel)
    AC_EGREP_CPP([^HYPRE_SEQUENTIAL_IS_DEFINED$], [
#include <HYPRE_config.h>
#ifdef HYPRE_SEQUENTIAL
HYPRE_SEQUENTIAL_IS_DEFINED
#endif
       ],
       hypre_MPI=no,
       hypre_MPI=yes)

    AC_MSG_RESULT($hypre_MPI)

    # Reset cpp after checking hypre header file.
    CPPFLAGS=$save_cppflags
fi

# End macro 
])


