dnl **********************************************************************
dnl * CASC_LANG_MPISWITCH(LANG)
dnl * 
dnl * Causes tests that would have been performed on a conventional
dnl * to be performed on the MPI wrapper version of that compiler.
dnl * The only acceptable arguments are "CC", "CXX", and "F77".  For
dnl * example, CASC_LANG_MPISWITCH(CXX) will temporarily set the variable
dnl * CXX to be equal to the value of the variable MPICXX.  The effect of
dnl * this macro is undone by the macro CASC_LANG_MPIUNSWITCH
dnl *
dnl * This macro is only for switching between the conventional compiler
dnl * and the MPI compiler for the same language.  To switch between
dnl * languages, see the builtin autoconf macros AC_LANG_C,
dnl * AC_LANG_CPLUSPLUS, and other related macros
dnl **********************************************************************

AC_DEFUN([CASC_LANG_MPISWITCH],
[
   if "$1" = "CC"; then
      CC_switch_save=$CC
      CC=$MPICC
   fi

   if "$1" = "CXX"
      CXX_switch_save=$CXX
      CXX=$MPICXX
   fi

   if "$1" = "F77"
      F77_switch_save=$F77
      F77=$MPIF77
   fi
])dnl

dnl *********************************************************************
dnl * CASC_LANG_MPIUNSWITCH(LANG)
dnl *
dnl * Undoes the results of a previous call to CASC_LANG_MPISWITCH.
dnl * The variable CC reverts to the value it had before 
dnl * CASC_LANG_MPISWITCH was called.  This macro should not be called
dnl * without a previous call to CASC_LANG_MPISWITCH with the same
dnl * argument.  Again, the only acceptable arguments are "CC", "CXX", and
dnl * "F77".
dnl *********************************************************************

AC_DEFUN([CASC_LANG_MPIUNSWITCH],
[
   if "$1" = "CC"; then
      $CC=$CC_switch_save
   fi

   if "$1" = "CXX"
      $CXX=$CXX_switch_save
   fi

   if "$1" = "F77"
      $F77=F77_switch_save
   fi
])dnl
