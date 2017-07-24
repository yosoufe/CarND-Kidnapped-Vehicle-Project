project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=gnu++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

find_package(CUDA QUIET REQUIRED)
ADD_DEFINITIONS(-DWITH_GPU)

set(sourcesCPP src/main.cpp src/map.h src/helper_functions.h)
set(sourcesCUDA src/particle_filter.cu src/particle_filter.cuh)


set(EXTRA_CXX_FLAGS "--std=c++11")
set(CUDA_NVCC_FLAGS ${EXTRA_CXX_FLAGS})
#cuda_compile(CUDA_OBJECTS ${sourcesCUDA})
CUDA_ADD_EXECUTABLE(particle_filter ${sourcesCPP} ${sourcesCUDA})



target_link_libraries(particle_filter z ssl uv uWS)
