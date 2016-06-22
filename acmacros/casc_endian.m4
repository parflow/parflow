AC_DEFUN([CASC_CHECK_BIGENDIAN],[
  AC_CACHE_VAL([casc_cv_bigendian],
               [AC_C_BIGENDIAN( [casc_cv_bigendian=yes], [casc_cv_bigendian=no], [casc_cv_bigendian=no] )
  ])

  if test "x$casc_cv_bigendian" = xyes
  then
    AC_DEFINE(CASC_HAVE_BIGENDIAN, 1,
       [Define this if words are stored with the most significant byte first])
  fi
]) # end of AC_DEFUN of CASC_CHECK_BIGENDIAN


