cmake_minimum_required (VERSION 2.8)
project (fttrainer)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

set(XGB $ENV{XGB})

#link_directories(${XGB_LIBRARY_PATH})
add_library(xgboostwrapper SHARED IMPORTED)
set_property(TARGET xgboostwrapper PROPERTY
    IMPORTED_LOCATION /${XGB}/lib/libxgboost.so)

add_executable(tag tag.cxx)
target_include_directories(tag PRIVATE
    ${XGB}/include/
    ${XGB}/dmlc-core/include/)
target_link_libraries(tag xgboostwrapper)
