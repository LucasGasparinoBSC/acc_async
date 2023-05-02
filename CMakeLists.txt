cmake_minimum_required(VERSION 3.15)

set(ENV{FC} "nvfortran")

project(acc_async LANGUAGES Fortran)

set(CMAKE_Fortran_FLAGS "-cpp -fast -gpu=lineinfo,ptxinfo -cuda -acc -Minfo=all -D_USE_NVTX -lnvToolsExt")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.f90)
set(SOURCES_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

add_executable(${PROJECT_NAME} ${SOURCES})