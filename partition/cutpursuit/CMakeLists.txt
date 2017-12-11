# Graph for Cut Pursuit
# author: Loic Landrieu
# date: 2017

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(CUT_PURSUIT_SEG)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -Wall -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")
##############################
### Find required packages ###
##############################

include_directories(include)

find_package(Boost 1.58.0 COMPONENTS graph REQUIRED) #system filesystem thread serialization
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

SET(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython2.7.so)
SET(PYTHON_INCLUDE_DIRS /usr/include/python2.7)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

file(GLOB CP_HEADERS include/*.h)

add_subdirectory(src)
