AC_DEFUN([CASC_CHECK_GETTIMEOFDAY],[
  AC_MSG_CHECKING([for gettimeofday])
  AC_CACHE_VAL([casc_cv_gettimeofday],[
  AC_TRY_RUN([
  #include <sys/time.h>
  int main() 
  { 
     struct timeval r_time;
     return gettimeofday(&r_time, 0);
  }
  ],[casc_cv_gettimeofday=yes],[casc_cv_gettimeofday=no],[casc_cv_gettimeofday=no]
  ) # end of TRY_RUN]) # end of CACHE_VAL

  AC_MSG_RESULT([$casc_cv_gettimeofday])
  if test x$casc_cv_gettimeofday = xyes
  then
    AC_DEFINE(CASC_HAVE_GETTIMEOFDAY, 1,
       [Define this if BSD gettimeofday is available])
  fi
]) # end of AC_DEFUN of CASC_CHECK_GETTIMEOFDAY

