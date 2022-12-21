cmake_minimum_required(VERSION 3.6)
project(mpi_matrix_multi)

set(CMAKE_C_COMPILER "/usr/bin/mpicc")
# добавить MPI Package
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(mpi_matrix_multi main.cpp)
target_link_libraries(mpi_matrix_multi ${MPI_C_LIBRARIES})
