cmake_minimum_required(VERSION 3.1)

project(OclExample CXX C)

SET(CMAKE_CXX_FLAGS "-std=c++23 -O3 -mavx2 -ftree-vectorize -s -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")

include_directories(.
${CMAKE_BINARY_DIR}/src/extern/compute/opencl/external/OpenCL-Headers
)

add_executable(OclExample 
                    #    src/Ocl.cpp
                    #    src/Ocl2.cpp
                       src/Ocl3.cpp
                   #    src/Ocl4.cpp
                        )

    
target_link_directories(OclExample PUBLIC 
${CMAKE_BINARY_DIR}/extern/compute/opencl/external/OpenCL-ICD-Loader
)

target_link_libraries(OclExample
    OpenCL
    )
