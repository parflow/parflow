dnl Define macros for supporting P4EST.

dnl Support p4est libraries by setting the variables
dnl p4est_PREFIX, p4est_INCLUDES, p4est_LIBS and p4est_MPI

AC_DEFUN([CASC_SUPPORT_P4EST],[

# Set p4est_PREFIX to the P4EST install directory.

unset p4est_PREFIX
AC_MSG_CHECKING([for using P4EST library])
AC_ARG_WITH([p4est],
    [AS_HELP_STRING([--with-p4est=PATH],
                    [Specify where P4EST is installed])],
    [if test "x$with_p4est" = xyes; then
        AC_MSG_ERROR([P4EST install directory omitted.
        Please specify --with-p4est=STRING])
    elif test "x$with_p4est" = xno; then
        AC_MSG_RESULT([no])
    else
        p4est_PREFIX="$with_p4est"
        AC_MSG_RESULT([$p4est_PREFIX])
    fi
    ],
    [   AC_MSG_RESULT([no])
    ]
)

if test "x$p4est_PREFIX" != x; then

    p4est_INCLUDES="-I${p4est_PREFIX}/include"
    p4est_LIBS="-L${p4est_PREFIX}/lib -lp4est -lsc"

    save_cppflags=$CPPFLAGS

    # Add p4est include flags to cpp to look in its header file.
    CPPFLAGS="$p4est_INCLUDES $CPPFLAGS"

    # Check if p4est header is ok.
    AC_CHECK_HEADER([p4est_config.h],[:],
    [AC_MSG_ERROR([p4est not found in $with_p4est])])

    # Check if HYPRE was compiled with parallelism.
    AC_MSG_CHECKING([p4est compiled serial or parallel])
    AC_EGREP_CPP([P4EST_COMPILED_WITH_MPI],
    [[
#include <p4est_config.h>
#ifdef P4EST_ENABLE_MPI
P4EST_COMPILED_WITH_MPI
#endif
    ]],
    [p4est_MPI=yes],
    [p4est_MPI=no])

    AC_MSG_RESULT([$p4est_MPI])

    # Reset cpp after checking p4est header file.
    CPPFLAGS="$save_cppflags"

fi

])
