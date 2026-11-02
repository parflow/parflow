# -----------------------------
# AMPS CMakeLists.txt
# -----------------------------

# Common AMPS files
set(COMMON_SRC_FILES
    amps_abort.c
    amps_clear.c
    amps_clock.c
    amps_exchange.c
    amps_ffopen.c
    amps_find_powers.c
    amps_fopen.c
    amps_invoice.c
    amps_io.c
    amps_newhandle.c
    amps_newpackage.c
    amps_sfbcast.c
    amps_sfclose.c
    amps_sfopen.c
    amps_wait.c
    signal.c
    amps_print.c
)

# Add RMM / Umpire wrappers if enabled
if(${PARFLOW_HAVE_RMM})
    list(APPEND COMMON_SRC_FILES amps_rmm_wrapper.cpp)
endif()

if(${PARFLOW_HAVE_UMPIRE})
    list(APPEND COMMON_SRC_FILES amps_umpire_wrapper.cpp)
endif()

# Prepend paths for common and AMPS transport layer
string(
    REGEX REPLACE "([^;]+)"
    "common/\\1"
    COMMON_SRC_FILES
    "${COMMON_SRC_FILES}"
)
include("${PARFLOW_AMPS_LAYER}/CMakeLists.cmake")
string(
    REGEX REPLACE "([^;]+)"
    "${PARFLOW_AMPS_LAYER}/\\1"
    AMPS_SRC_FILES
    "${AMPS_SRC_FILES}"
)

# Set LANGUAGE CXX if building with C++
if(${PARFLOW_BUILD_WITH_CPP})
    set_source_files_properties(
        ${COMMON_SRC_FILES}
        ${AMPS_SRC_FILES}
        PROPERTIES LANGUAGE CXX
    )
endif()

# -----------------------------
# Create AMPS static library
# -----------------------------
add_library(amps STATIC ${COMMON_SRC_FILES} ${AMPS_SRC_FILES})

# -----------------------------
# Determine if C++17 is needed for rmm or umpire
# -----------------------------
# Initialize flags
set(AMPS_NEEDS_CXX17 OFF)
set(AMPS_CXX17_REASON "")

# --- Check RMM ---
if(PARFLOW_HAVE_RMM)
    # RMM >= 25.06.00 requires C++17
    if(rmm_VERSION VERSION_GREATER_EQUAL 25.06.00)
        set(AMPS_NEEDS_CXX17 ON)
        set(AMPS_CXX17_REASON "RMM >= 25.06.00 requires C++17")
    endif()
endif()

# --- Check Umpire ---
if(PARFLOW_HAVE_UMPIRE)
    # Umpire >= 2025.09.0 requires C++17
    if(umpire_VERSION VERSION_GREATER_EQUAL 2025.09.0)
        set(AMPS_NEEDS_CXX17 ON)
        set(AMPS_CXX17_REASON "Umpire >= 2025.09.0 requires C++17")
    endif()
endif()

# --- Apply C++17 only if needed ---
if(AMPS_NEEDS_CXX17)
    message(STATUS "Setting amps C++17 because: ${AMPS_CXX17_REASON}")
    target_compile_features(amps PRIVATE cxx_std_17)
endif()

# -----------------------------
# Include directories
# -----------------------------
if(${PARFLOW_HAVE_CUDA})
    target_include_directories(
        amps
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endif()

if(${PARFLOW_HAVE_MPI})
    target_include_directories(amps PUBLIC "${MPI_C_INCLUDE_PATH}")
endif()

# -----------------------------
# Link dependencies
# -----------------------------
if(${PARFLOW_HAVE_RMM})
    target_link_libraries(amps PUBLIC rmm::rmm)
endif()

if(${PARFLOW_HAVE_UMPIRE})
    target_link_libraries(amps PUBLIC umpire)
endif()

if(${PARFLOW_HAVE_KOKKOS})
    target_link_libraries(amps PUBLIC Kokkos::kokkos)
    target_include_directories(amps PUBLIC ${KOKKOS_INCLUDE})
endif()

if(${PARFLOW_HAVE_OAS3})
    target_link_libraries(amps PUBLIC OASIS3MCT::OASIS3MCT)
endif()

# -----------------------------
# Install targets and scripts
# -----------------------------
install(TARGETS amps DESTINATION lib)

# Install the AMPS transport layer scripts
set(AMPS_SCRIPTS run)
string(
    REGEX REPLACE "([^;]+)"
    "${PARFLOW_AMPS_LAYER}/\\1"
    AMPS_SCRIPTS
    "${AMPS_SCRIPTS}"
)
install(FILES ${AMPS_SCRIPTS} DESTINATION bin)

# Add subdirectory for tests
add_subdirectory(test/src)
