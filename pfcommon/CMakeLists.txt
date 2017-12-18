set (SRC_FILES messages.c)

FIND_LIBRARY(FCA_LIBRARY fca)

add_library(pfcommon ${SRC_FILES})

install (TARGETS pfcommon DESTINATION lib)

set (BUILD_PYTHON_ANALYZER_API True CACHE BOOL
  "Build Python Analyzer API to build analyzer modules in python")

IF( BUILD_PYTHON_ANALYZER_API )

  FIND_PACKAGE(SWIG)
  FIND_PACKAGE(PythonLibs)
  FIND_PACKAGE(NumPy)


  IF( PYTHONLIBS_FOUND AND SWIG_FOUND AND NUMPY_FOUND)
    SET(NUMPY_I_PATH ${NUMPY_INCLUDE_DIRS}/../../../instant/swig CACHE PATH "Path to numpy.i")
    IF(NOT EXISTS ${NUMPY_I_PATH}/numpy.i)
      Message(SEND_ERROR "Cannot find numpy.i in ${NUMPY_I} which needed for the Python Analyzer API")
    ENDIF(NOT EXISTS ${NUMPY_I_PATH}/numpy.i)


    set_source_files_properties(${SRC_FILES} PROPERTIES COMPILE_FLAGS -fPIC)
    INCLUDE(UseSWIG)

    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${FCA_INCLUDE_DIRECTORY}/..
      ${NUMPY_I_PATH}
      )

    SET(CMAKE_SWIG_FLAGS "")
    SWIG_ADD_LIBRARY(pypfcommon LANGUAGE python SOURCES pypfcommon.i pypfcommon.c)
    SWIG_LINK_LIBRARIES(pypfcommon ${PYTHON_LIBRARIES} pfcommon ${FCA_LIBRARY})
    # low: performance better if we compile wit ${PYTHON_LIBRARIES} and if we do not use
    # the previously linked version of pfcommon but include ${SRC_FILES}

    #execute_process ( COMMAND
      #python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
      #OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
    #INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/pypfcommon.py DESTINATION ${PYTHON_SITE_PACKAGES})

    STRING(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.[0-9]+" "\\1;\\2"
      RESULT ${PYTHONLIBS_VERSION_STRING})

    list(GET RESULT 0 PYTHON_MAJOR_VERSION)
    list(GET RESULT 1 PYTHON_MINOR_VERSION)

    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/_pypfcommon.so
      ${CMAKE_CURRENT_BINARY_DIR}/pypfcommon.py
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})

  ENDIF( PYTHONLIBS_FOUND  AND SWIG_FOUND AND NUMPY_FOUND )

ENDIF( BUILD_PYTHON_ANALYZER_API )
