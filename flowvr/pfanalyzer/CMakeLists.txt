set (SRC_FILES messages.c)

FIND_LIBRARY(FCA_LIBRARY fca)

add_library(pfanalyzer ${SRC_FILES})

install (TARGETS pfanalyzer DESTINATION lib)

set (BUILD_PYTHON_ANALYZER_API True CACHE BOOL
  "Build Python Analyzer API to build analyzer modules in python")

IF( BUILD_PYTHON_ANALYZER_API )

  FIND_PACKAGE(SWIG)
  FIND_PACKAGE(PythonLibs)
  FIND_PACKAGE(NumPy)


  IF( PYTHONLIBS_FOUND AND SWIG_FOUND AND NUMPY_FOUND)
    SET(NUMPY_I_PATH ${NUMPY_INCLUDE_DIRS}/../../../instant/swig CACHE PATH "Path to numpy.i")
    IF(NOT EXISTS ${NUMPY_I_PATH}/numpy.i)
      Message(SEND_ERROR "Cannot find numpy.i in ${NUMPY_I} which is needed for the Python Analyzer API")
    ENDIF(NOT EXISTS ${NUMPY_I_PATH}/numpy.i)


    set_source_files_properties(${SRC_FILES} PROPERTIES COMPILE_FLAGS -fPIC)
    INCLUDE(UseSWIG)

    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${fca_INCLUDE_DIRECTORY}
      ${NUMPY_I_PATH}
      )

    SET(CMAKE_SWIG_FLAGS "")
    SWIG_ADD_LIBRARY(pypfAnalyzer LANGUAGE python SOURCES pypfAnalyzer.i pypfAnalyzer.c)
    SWIG_LINK_LIBRARIES(pypfAnalyzer ${PYTHON_LIBRARIES} pfanalyzer ${FCA_LIBRARY})

    STRING(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.[0-9]+" "\\1;\\2"
      RESULT ${PYTHONLIBS_VERSION_STRING})

    list(GET RESULT 0 PYTHON_MAJOR_VERSION)
    list(GET RESULT 1 PYTHON_MINOR_VERSION)

    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/_pypfAnalyzer.so
      ${CMAKE_CURRENT_BINARY_DIR}/pypfAnalyzer.py
      DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION})

  ENDIF( PYTHONLIBS_FOUND  AND SWIG_FOUND AND NUMPY_FOUND )

ENDIF( BUILD_PYTHON_ANALYZER_API )
