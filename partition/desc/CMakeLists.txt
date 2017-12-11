# Graph for Cut Pursuit
# author: Loic Landrieu
# date: 2017

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(LIBDESC)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

##############################
### Find required packages ###
##############################

find_package(Boost 1.58.0 COMPONENTS system filesystem thread serialization REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(Eigen3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
LINK_DIRECTORIES(${EIGEN3_LIBRARY_DIRS})

SET(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython2.7.so)
SET(PYTHON_INCLUDE_DIRS /usr/include/python2.7)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

##############################
###  Build target library  ###
##############################

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -fopenmp -O3 -Wall")

add_library(desc SHARED desc.cpp)
target_link_libraries(desc
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    boost_python
    )
