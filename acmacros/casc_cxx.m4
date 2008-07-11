dnl
dnl CASC C++ autoconf macros
dnl
dnl The following macros test various features of the C++ compiler, including:
dnl
dnl	- boolean types
dnl	- namespace construct
dnl	- template-based complex numbers
dnl	- sstream.h header file with class ostringstream
dnl	- new placement operator
dnl	- explicit template instantiation
dnl	- standard member function specialization
dnl	- standard static data instantiation
dnl	- standard static data specialization
dnl	- static data specialization via pragmas
dnl

dnl
dnl Check whether the C++ compiler supports the boolean built-in type.
dnl
dnl Variable:	casc_cv_cxx_have_bool = (yes|no)
dnl Defines:	(HAVE|LACKS)_BOOL
dnl

AC_DEFUN([CASC_CXX_BOOL], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports bool)

   AC_CACHE_VAL(casc_cv_cxx_have_bool, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
bool b = true;
         ], [/* empty */],
         casc_cv_cxx_have_bool=yes,
         casc_cv_cxx_have_bool=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_bool)

   if test "$casc_cv_cxx_have_bool" = yes; then
      AC_DEFINE(HAVE_BOOL)
   else
      AC_DEFINE(LACKS_BOOL)
   fi
])

dnl
dnl Check whether the C++ compiler supports the ANSI/ISO namespace construct.
dnl
dnl Variable:	casc_cv_cxx_have_namespace = (yes|no)
dnl Defines:	(HAVE|LACKS)_NAMESPACE
dnl

AC_DEFUN([CASC_CXX_NAMESPACE], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports namespace)

   AC_CACHE_VAL(casc_cv_cxx_have_namespace, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_LINK([
namespace test {
   int t();
   int t() { return 0; } 
}
using namespace test;
int foo() { int x = t(); x++; return x; }
         ], [/* empty */],
         casc_cv_cxx_have_namespace=yes,
         casc_cv_cxx_have_namespace=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_namespace)

   if test "$casc_cv_cxx_have_namespace" = yes; then
      AC_DEFINE(HAVE_NAMESPACE)
   else
      AC_DEFINE(LACKS_NAMESPACE)
   fi
])

dnl
dnl Check whether the C++ compiler supports cmath
dnl
dnl Variable:	casc_cv_cxx_have_cmath = (yes|no)
dnl Defines:	(HAVE|LACKS)_CMATH
dnl

AC_DEFUN([CASC_CXX_CMATH], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports cmath)

   AC_CACHE_VAL(casc_cv_cxx_have_cmath, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
         AC_TRY_COMPILE([
#include <cmath>
void foo() {
   double temp = std::sin(0.0);
}
            ], [/* empty */],
            casc_cv_cxx_have_cmath=yes,
            casc_cv_cxx_have_cmath=no)
         AC_LANG_RESTORE
      ])

   AC_MSG_RESULT($casc_cv_cxx_have_cmath)

   if test "$casc_cv_cxx_have_cmath" = yes; then
      AC_DEFINE(HAVE_CMATH)
   else
      AC_DEFINE(LACKS_CMATH)
   fi
])

dnl
dnl Check whether the C++ compiler supports template-based complex numbers.
dnl
dnl Variable:	casc_cv_cxx_have_template_comlex = (yes|no)
dnl Defines:	(HAVE|LACKS)_TEMPLATE_COMPLEX
dnl

AC_DEFUN([CASC_CXX_TEMPLATE_COMPLEX], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports template-based complex numbers)

   AC_CACHE_VAL(casc_cv_cxx_have_template_complex, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
#include <complex.h>
void foo() {
   complex<double> a(0.0, 1.0);
}
         ], [/* empty */],
         casc_cv_cxx_have_template_complex=yes,
         casc_cv_cxx_have_template_complex=no)
      AC_LANG_RESTORE
   ])

   AC_MSG_RESULT($casc_cv_cxx_have_template_complex)

   if test "$casc_cv_cxx_have_template_complex" = yes; then
      AC_DEFINE(HAVE_TEMPLATE_COMPLEX)
   else
      AC_MSG_CHECKING(whether ${CXX} supports ISO template-based complex numbers)
      AC_CACHE_VAL(casc_cv_cxx_have_template_complex_std, [
         AC_LANG_SAVE
         AC_LANG_CPLUSPLUS
         AC_TRY_COMPILE([
#include <complex>
void foo() {
   std::complex<double> a(0.0, 1.0);
}
            ], [/* empty */],
            casc_cv_cxx_have_template_complex_std=yes,
            casc_cv_cxx_have_template_complex_std=no)
         AC_LANG_RESTORE
      ])

      AC_MSG_RESULT($casc_cv_cxx_have_template_complex_std)

      if test "$casc_cv_cxx_have_template_complex_std" = yes; then
         AC_DEFINE(HAVE_TEMPLATE_COMPLEX)
      else
         AC_DEFINE(LACKS_TEMPLATE_COMPLEX)
      fi
   fi
])

dnl
dnl Check whether the C++ compiler supports sstream.h and class ostringstream.
dnl
dnl Variable:	casc_cv_cxx_have_sstream = (yes|no)
dnl Defines:	(HAVE|LACKS)_SSTREAM
dnl             NOTE: Also defines (HAVE|LACKS)_ISO_SSTREAM if compiler 
dnl                   supports or does not support std ISO header file 
dnl                   <sstream>.  This can be used determine a certain 
dnl                   level of compiler support for std ISO header files. 
dnl                   It is not intended to apply to all compilers.
dnl

AC_DEFUN([CASC_CXX_SSTREAM], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports sstream.h and class ostringstream)

   AC_CACHE_VAL(casc_cv_cxx_have_sstream, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
#include <sstream.h>
void foo() {
//   char buffer[100];
//   std::ostringstream os(buffer, 100);
//   os << "hello, world...";
   std::ostringstream out;
   double f = 5.0;
   int    g = 1;
   out << "(" << f << "<some string>" << g << ")";
}
         ], [/* empty */],
         casc_cv_cxx_have_sstream=yes,
         casc_cv_cxx_have_sstream=no)
      AC_LANG_RESTORE
   ])

   AC_MSG_RESULT($casc_cv_cxx_have_sstream)

   if test "$casc_cv_cxx_have_sstream" = yes; then
      AC_DEFINE(HAVE_SSTREAM)
   else
        AC_MSG_CHECKING(whether ${CXX} supports sstream and class ostringstream)
	AC_CACHE_VAL(casc_cv_cxx_have_sstream_std, [
        AC_LANG_SAVE
        AC_LANG_CPLUSPLUS
        AC_TRY_COMPILE([
#include <sstream>
void foo() {
//   char buffer[100];
//   std::ostringstream os(buffer, 100);
//   os << "hello, world...";
   std::ostringstream out;
   double f = 5.0;
   int    g = 1;
   out << "(" << f << "<some string>" << g << ")";
}
         ], [/* empty */],
        casc_cv_cxx_have_sstream_std=yes,
        casc_cv_cxx_have_sstream_std=no)
        AC_LANG_RESTORE
        ])

      AC_MSG_RESULT($casc_cv_cxx_have_sstream_std)

      if test "$casc_cv_cxx_have_sstream_std" = yes; then
         AC_DEFINE(HAVE_SSTREAM)
         AC_DEFINE(HAVE_ISO_SSTREAM)
      else
         AC_DEFINE(LACKS_SSTREAM)
         AC_DEFINE(LACK_ISO_SSTREAM)
      fi
   fi
])

dnl
dnl Check if left is supported
dnl
dnl Variable:	casc_cv_cxx_have_iomanip_left = (yes|no)
dnl Defines:	(HAVE|LACKS)_IOMANIP_LEFT

AC_DEFUN([CASC_CXX_IOMANIP_LEFT], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} defines the iomanip left operator)

   AC_CACHE_VAL(casc_cv_cxx_have_iomanip_left, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_LINK([
#include <iostream>
#include <iomanip>
using namespace std;

void foo() 
{
   cout << left << 12.1;
}
         ], [/* empty */],
         casc_cv_cxx_have_iomanip_left=yes,
         casc_cv_cxx_have_iomanip_left=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_iomanip_left)

   if test "$casc_cv_cxx_have_iomanip_left" = yes; then
      AC_DEFINE(HAVE_IOMANIP_LEFT)
   else
      AC_DEFINE(LACKS_IOMANIP_LEFT)
   fi
])


dnl
dnl Check whether the C++ compiler defines the new placement operator.
dnl
dnl Variable:	casc_cv_cxx_have_new_placement_operator = (yes|no)
dnl Defines:	(HAVE|LACKS)_NEW_PLACEMENT_OPERATOR
dnl

AC_DEFUN([CASC_CXX_NEW_PLACEMENT_OPERATOR], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} defines the new placement operator)

   AC_CACHE_VAL(casc_cv_cxx_have_new_placement_operator, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_LINK([
#include <new>
void trynew() {
   void *ptr = 0;
   double *data = new (ptr) double;
}
         ], [/* empty */],
         casc_cv_cxx_have_new_placement_operator=yes,
         casc_cv_cxx_have_new_placement_operator=no)
      AC_LANG_RESTORE
   ])

   AC_MSG_RESULT($casc_cv_cxx_have_new_placement_operator)

   if test "$casc_cv_cxx_have_new_placement_operator" = yes; then
      AC_DEFINE(HAVE_NEW_PLACEMENT_OPERATOR)
   else
      AC_DEFINE(LACKS_NEW_PLACEMENT_OPERATOR)
   fi
])

dnl
dnl Check whether the C++ compiler supports explicit template instantiation.
dnl
dnl Variable:	casc_cv_cxx_have_explicit_template_instantiation = (yes|no)
dnl Defines:	(HAVE|LACKS)_EXPLICIT_TEMPLATE_INSTANTIATION
dnl
dnl The explicit template instantiation syntax forces the compiler to
dnl instantiate a template of the specified type.  For example,
dnl
dnl	template <class T> class Pointer {
dnl	   public: T *value;
dnl	};
dnl	#ifndef LACKS_EXPLICIT_TEMPLATE_INSTANTIATION
dnl	template class Pointer<int>;
dnl	#endif
dnl
dnl will create the code for a Pointer of type int.  If this syntax is
dnl not allowed, then the compiler must define some other mechanism for
dnl automatically generating template code.
dnl

AC_DEFUN([CASC_CXX_EXPLICIT_TEMPLATE_INSTANTIATION], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports explicit template instantiation)
   AC_CACHE_VAL(casc_cv_cxx_have_explicit_template_instantiation, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
template <class T> class Pointer { public: T *value; };
template class Pointer<int>;
         ], [/* empty */],
         casc_cv_cxx_have_explicit_template_instantiation=yes,
         casc_cv_cxx_have_explicit_template_instantiation=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_explicit_template_instantiation)

   if test "$casc_cv_cxx_have_explicit_template_instantiation" = yes; then
      AC_DEFINE(HAVE_EXPLICIT_TEMPLATE_INSTANTIATION)
   else
      AC_DEFINE(LACKS_EXPLICIT_TEMPLATE_INSTANTIATION)
   fi
])

dnl
dnl Check whether the C++ compiler supports member function specialization.
dnl
dnl Variable:	casc_cv_cxx_have_member_function_specialization = (yes|no)
dnl Defines:	(HAVE|LACKS)_MEMBER_FUNCTION_SPECIALIZATION
dnl
dnl The ANSI/ISO member function specialization syntax is used when defining
dnl a specialized member function of a template class.  For example:
dnl
dnl	template <class T> class Pointer {
dnl	   public: void foo();
dnl	};
dnl	#ifndef LACKS_MEMBER_FUNCTION_SPECIALIZATION
dnl	template <>
dnl	#endif
dnl     void Pointer<int>::foo() { }
dnl
dnl will define the specialized version of Pointer<int>::foo().  Some
dnl compilers such as GNU g++ cannot parse the template <> syntax.
dnl

AC_DEFUN([CASC_CXX_MEMBER_FUNCTION_SPECIALIZATION], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports member function specialization)
   AC_CACHE_VAL(casc_cv_cxx_have_member_function_specialization, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
template <class T> class Pointer { public: void foo(); };
template <> void Pointer<int>::foo();
template <> void Pointer<int>::foo() { }
         ], [/* empty */],
         casc_cv_cxx_have_member_function_specialization=yes,
         casc_cv_cxx_have_member_function_specialization=no)
      AC_LANG_RESTORE
   ])

   dnl ASCI Red compiles but does not generate the code so manually
   dnl set this
   case $ARCH in
      ipsc2)
         casc_cv_cxx_have_member_function_specialization=no
         ;;
   esac

   AC_MSG_RESULT($casc_cv_cxx_have_member_function_specialization)
   if test "$casc_cv_cxx_have_member_function_specialization" = yes; then
      AC_DEFINE(HAVE_MEMBER_FUNCTION_SPECIALIZATION)
   else
      AC_DEFINE(LACKS_MEMBER_FUNCTION_SPECIALIZATION)
   fi

])

dnl
dnl Check whether the C++ compiler supports static data instantiation.
dnl
dnl Variable:	casc_cv_cxx_have_static_data_instantiation = (yes|no)
dnl Defines:	(HAVE|LACKS)_STATIC_DATA_INSTANTIATION
dnl
dnl The ANSI/ISO specifies that the default values of the static data members
dnl of a template class may be defined as follows:
dnl
dnl	template <class T> class Pointer {
dnl	   public: static T *s_test;
dnl	};
dnl	#ifndef LACKS_STATIC_DATA_INSTANTIATION
dnl	template <class T> T* Pointer<T>::s_test = (T*) 0;
dnl	#endif
dnl
dnl Some compilers such as GNU g++ cannot parse the generic static data member
dnl instantiation syntax and require that static data members for type T be
dnl explicitly specified to instantiate the data member.

AC_DEFUN([CASC_CXX_STATIC_DATA_INSTANTIATION], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports static data instantiation)
   AC_CACHE_VAL(casc_cv_cxx_have_static_data_instantiation, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
template <class T> class Pointer { public: void foo(); };
template <> void Pointer<int>::foo() { }
         ], [/* empty */],
         casc_cv_cxx_have_static_data_instantiation=yes,
         casc_cv_cxx_have_static_data_instantiation=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_static_data_instantiation)
   if test "$casc_cv_cxx_have_static_data_instantiation" = yes; then
      AC_DEFINE(HAVE_STATIC_DATA_INSTANTIATION)
   else
      AC_DEFINE(LACKS_STATIC_DATA_INSTANTIATION)
   fi
])

dnl
dnl Check whether the C++ compiler supports standard static data specialization.
dnl
dnl Variable:	casc_cv_cxx_have_standard_static_data_specialization = (yes|no)
dnl Defines:	(HAVE|LACKS)_STANDARD_STATIC_DATA_SPECIALIZATION
dnl
dnl The ANSI/ISO specifies that static data members of a template class may
dnl be specialized as follows:
dnl
dnl	template <class T> class Pointer {
dnl	   public: static T *s_test;
dnl	};
dnl	template <> int *Pointer<int>::s_test;
dnl	template <> int *Pointer<int>::s_test = (int*) 0;
dnl	template class Pointer<int>;
dnl
dnl Some compilers such as GNU g++ and older versions of KCC cannot parse
dnl this syntax and use other methods (such as pragmas or different syntax).
dnl

AC_DEFUN([CASC_CXX_STANDARD_STATIC_DATA_SPECIALIZATION], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports standard static data specialization)
   AC_CACHE_VAL(casc_cv_cxx_have_standard_static_data_specialization, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
template <class T> class Pointer { public: static T *s_test; };
template <> int *Pointer<int>::s_test;
int test() { Pointer<int> P; return(*P.s_test); }
template <> int *Pointer<int>::s_test = (int*) 0;
template class Pointer<int>;
         ], [/* empty */],
         casc_cv_cxx_have_standard_static_data_specialization=yes,
         casc_cv_cxx_have_standard_static_data_specialization=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_standard_static_data_specialization)
   if test "$casc_cv_cxx_have_standard_static_data_specialization" = yes; then
      AC_DEFINE(HAVE_STANDARD_STATIC_DATA_SPECIALIZATION)
   else
      AC_DEFINE(LACKS_STANDARD_STATIC_DATA_SPECIALIZATION)
   fi
])

dnl
dnl Check whether the C++ compiler supports pragma static data specialization.
dnl
dnl Variable:	casc_cv_cxx_have_pragma_static_data_specialization = (yes|no)
dnl Defines:	(HAVE|LACKS)_PRAGMA_STATIC_DATA_SPECIALIZATION
dnl
dnl Some compilers support the specialization of a static data member of a
dnl template class using the following syntax:
dnl
dnl	template <class T> class Pointer {
dnl	   public: static T *s_test;
dnl	};
dnl	#pragma do_not_instantiate int *Pointer<int>::s_test
dnl	template <> int *Pointer<int>::s_test = (int*) 0;
dnl	template class Pointer<int>;
dnl
dnl This syntax is supported by older versions of KCC.  Note that this
dnl macro should be used ONLY if the standard static data specialization
dnl syntax fails.
dnl

AC_DEFUN([CASC_CXX_PRAGMA_STATIC_DATA_SPECIALIZATION], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports pragma static data specialization)
   AC_CACHE_VAL(casc_cv_cxx_have_pragma_static_data_specialization, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([
template <class T> class Pointer { public: static T *s_test; };
#pragma do_not_instantiate int *Pointer<int>::s_test
int test() { Pointer<int> P; return(*P.s_test); }
template <> int *Pointer<int>::s_test = (int*) 0;
template class Pointer<int>;
         ], [/* empty */],
         casc_cv_cxx_have_pragma_static_data_specialization=yes,
         casc_cv_cxx_have_pragma_static_data_specialization=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_pragma_static_data_specialization)
   if test "$casc_cv_cxx_have_pragma_static_data_specialization" = yes; then
      AC_DEFINE(HAVE_PRAGMA_STATIC_DATA_SPECIALIZATION)
   else
      AC_DEFINE(LACKS_PRAGMA_STATIC_DATA_SPECIALIZATION)
   fi
])

dnl
dnl Check whether the C++ compiler supports exception handling.
dnl
dnl Variable:	casc_cv_cxx_have_exception_handling = (yes|no)
dnl Defines:	(HAVE|LACKS)_EXCEPTION_HANDLING
dnl
dnl Compilers that support exception handling will support the following  
dnl operation: 
dnl
dnl static void byebye(int error) {
dnl    fprintf(stderr, "floating point exception\n");   abort(); 
dnl }
dnl int main(int argc, char** argv) {
dnl    unsigned short fpu_flags = _FPU_DEFAULT;
dnl    fpu_flags &= ~_FPU_MASK_IM;  /* Execption on Invalid operation */
dnl    fpu_flags &= ~_FPU_MASK_ZM;  /* Execption on Division by zero  */
dnl    fpu_flags &= ~_FPU_MASK_OM;  /* Execption on Overflow */
dnl    _FPU_SETCW(fpu_flags);
dnl    signal(SIGFPE, byebye);  /* Invoke byebye when above occurs */
dnl }
dnl
dnl
AC_DEFUN([CASC_CXX_EXCEPTION_HANDLING], [
    AC_REQUIRE([AC_PROG_CXX])
    AC_MSG_CHECKING(whether ${CXX} supports exception handling)
    AC_CACHE_VAL(casc_cxx_have_exception_handling, [
       AC_LANG_SAVE
       AC_LANG_CPLUSPLUS
       AC_TRY_COMPILE([
#include <fpu_control.h>
#include <signal.h>
static void byebye(int error) { }
void foo() {
   unsigned short fpu_flags = _FPU_DEFAULT;
   fpu_flags &= ~_FPU_MASK_IM;  /* Execption on Invalid operation */
   fpu_flags &= ~_FPU_MASK_ZM;  /* Execption on Division by zero  */
   fpu_flags &= ~_FPU_MASK_OM;  /* Execption on Overflow */
   _FPU_SETCW(fpu_flags);
   signal(SIGFPE, byebye);
}
         ], [/* empty */],
         casc_cxx_have_exception_handling=yes,
         casc_cxx_have_exception_handling=no)
       AC_LANG_RESTORE
    ])
    AC_MSG_RESULT($casc_cxx_have_exception_handling)
    if test "$casc_cxx_have_exception_handling" = yes; then
       AC_DEFINE(HAVE_EXCEPTION_HANDLING)
    else
       AC_DEFINE(LACKS_EXCEPTION_HANDLING)
    fi
])


dnl
dnl Determines which form of isnan is present
dnl 
dnl Defines:	(HAVE|LACKS)_CMATH_ISNAN
dnl             (HAVE|LACKS)_ISNAN
dnl  	        (HAVE|LACKS)_ISNAND
dnl  	        (HAVE|LACKS)_INLINE_ISNAND
dnl
dnl isnan is part of C99 spec and not necessarily available under
dnl ISO C++.  Test for some other possible functions.
dnl
AC_DEFUN([CASC_CXX_ISNAN], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(checking for isnan in cmath)

   AC_LANG_SAVE
   AC_LANG_CPLUSPLUS
   AC_TRY_COMPILE([ #include <cmath> ], 
      [ int test = std::isnan(0.0); ],
      casc_cv_cxx_have_isnan=yes,
      casc_cv_cxx_have_isnan=no)
   AC_LANG_RESTORE

   AC_MSG_RESULT($casc_cv_cxx_have_isnan)

   if test "$casc_cv_cxx_have_isnan" = yes; then
      AC_DEFINE(HAVE_CMATH_ISNAN)
   else
      AC_DEFINE(LACKS_CMATH_ISNAN)

      AC_MSG_CHECKING(checking for isnan in math.h)

      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([#include <math.h>], 
         [int test = isnan(0.0);],
         casc_cv_cxx_have_isnan=yes,
         casc_cv_cxx_have_isnan=no)
      AC_LANG_RESTORE

      AC_MSG_RESULT($casc_cv_cxx_have_isnan)

      if test "$casc_cv_cxx_have_isnan" = yes; then
         AC_DEFINE(HAVE_ISNAN)
      else
         AC_DEFINE(LACKS_ISNAN)

         AC_MSG_CHECKING(checking for __isnand)

         AC_LANG_SAVE
         AC_LANG_CPLUSPLUS
         AC_TRY_COMPILE([#include <math.h>],
            [int test = __isnand(0.0);],
            casc_cv_cxx_have_isnand=yes,
            casc_cv_cxx_have_isnand=no)
         AC_LANG_RESTORE
  
         AC_MSG_RESULT($casc_cv_cxx_have_isnand)
         if test "$casc_cv_cxx_have_isnand" = yes; then
            AC_DEFINE(HAVE_ISNAND)
         else
            AC_DEFINE(LACKS_ISNAND)

            AC_MSG_CHECKING(checking for __inline_isnand)

            AC_LANG_SAVE
            AC_LANG_CPLUSPLUS
            AC_TRY_COMPILE([#include <math.h>],
                 [int test = __inline_isnand(0.0);],
                casc_cv_cxx_have_inline_isnan=yes,
                casc_cv_cxx_have_inline_isnan=no)
              AC_LANG_RESTORE

            AC_MSG_RESULT($casc_cv_cxx_have_inline_isnan)
            if test "$casc_cv_cxx_have_inline_isnan" = yes; then
               AC_DEFINE(HAVE_INLINE_ISNAND)
            else
               AC_DEFINE(LACKS_INLINE_ISNAND)
           fi
	fi
      fi
   fi
])


dnl
dnl Check whether the GNU C++ compiler needs float NAN templates
dnl
dnl Variable:	casc_cv_cxx_have_isnan_template = (yes|no)
dnl Defines:	(HAVE|LACKS)_ISNAN_TEMPLATE
dnl

AC_DEFUN([CASC_CXX_ISNAN_TEMPLATE], [
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} needs isnan templates)

   AC_CACHE_VAL(casc_cv_cxx_have_isnan_template, [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS
      AC_TRY_COMPILE([

#include <complex>

template int __gnu_cxx::isnan<float>(float);
template int __gnu_cxx::__capture_isnan<float>(float);
         ], [/* empty */],
         casc_cv_cxx_have_isnan_template=yes,
         casc_cv_cxx_have_isnan_template=no)
      AC_LANG_RESTORE
   ])
   AC_MSG_RESULT($casc_cv_cxx_have_isnan_template)

   if test "$casc_cv_cxx_have_isnan_template" = yes; then
      AC_DEFINE(HAVE_ISNAN_TEMPLATE)
   else
      AC_DEFINE(LACKS_ISNAN_TEMPLATE)
   fi
])
