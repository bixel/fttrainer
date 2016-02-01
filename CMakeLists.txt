cmake_minimum_required (VERSION 2.8)
project (fttrainer)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

set(XGB $ENV{XGB_REPO})

set(XGB_INCLUDE_PATH ${XGB}/wrapper ${XGB}/src/io)
set(XGB_LIBRARY_PATH ${XGB}/wrapper)

#link_directories(${XGB_LIBRARY_PATH})
add_library(xgboostwrapper SHARED IMPORTED)
set_property(TARGET xgboostwrapper PROPERTY
    IMPORTED_LOCATION /${XGB_LIBRARY_PATH}/libxgboostwrapper.so)

add_executable(tag tag.cxx)
target_include_directories(tag PRIVATE ${XGB_INCLUDE_PATH})
target_link_libraries(tag xgboostwrapper)
