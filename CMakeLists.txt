cmake_minimum_required (VERSION 2.8)
project (fttrainer)

set(CMAKE_CXX_COMPILER g++-5)
set(CMAKE_C_COMPILER gcc-5)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

set(XGB_INCLUDE_DIRS
    /Users/kheinicke/repos/xgboost/wrapper/
    /Users/kheinicke/repos/xgboost/src/io/
    )
set(XGB_LIBRARY_DIRS "/Users/kheinicke/repos/xgboost/wrapper/")

#link_directories(${XGB_LIBRARY_DIRS})
add_library(xgboostwrapper SHARED IMPORTED)
set_property(TARGET xgboostwrapper PROPERTY IMPORTED_LOCATION ${XGB_LIBRARY_DIRS}/libxgboostwrapper.so)

add_executable(tag tag.cxx)
target_include_directories(tag PRIVATE ${XGB_INCLUDE_DIRS})
target_link_libraries(tag xgboostwrapper)
