# Project Name
PROJECT(IndySAExamples)

cmake_minimum_required(VERSION 2.8)

# Find all our packages
FIND_PACKAGE(CUDA REQUIRED)
# FIND_PACKAGE(Boost)
FIND_PACKAGE(Doxygen)
FIND_PACKAGE(OpenMP REQUIRED)

# Needed to enable the proper compilation of extern defined global values.
set(CUDA_SEPARABLE_COMPILATION ON)

# Easiest to set your compilation variables all in one place.
SET(CUDACOMPILEOPS --ptxas-options=-v -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_20,code=sm_20 -lineinfo)

# We can easily define any additional flags required for debugging versus release
IF(CMAKE_BUILD_TYPE MATCHES Debug)
  SET(CUDACOMPILEOPS ${CUDACOMPILEOPS} -g -G)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -fpermissive")
ELSE()
  SET(CUDACOMPILEOPS "${CUDACOMPILEOPS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fpermissive")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

# How we must handle the addition of OpenMP parallelization 
IF(OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(OPENMP_FOUND)

# Doxygen requires some extra configuration after installationH
IF(DOXYGEN_FOUND)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  ADD_CUSTOM_TARGET(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generated API documentation with Doxygen" VERBATIM
  )
ENDIF(DOXYGEN_FOUND)

# This makes sure we can use some of the additions in the C99 standard for our host code
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

# Source code
ADD_SUBDIRECTORY(examples)
