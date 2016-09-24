cmake_minimum_required (VERSION 2.8.5)
project (fttrainer)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_SKIP_RPATH 1)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++14)

set(XGB $ENV{XGB})

#link_directories(${XGB_LIBRARY_PATH})
add_library(xgboostwrapper SHARED IMPORTED)
set_property(TARGET xgboostwrapper PROPERTY
    IMPORTED_LOCATION ${XGB}/wrapper/libxgboostwrapper.so)

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_executable(tag tag.cxx)
target_include_directories(tag PRIVATE
    ${XGB}/wrapper/
    )
target_link_libraries(tag xgboostwrapper ${Boost_LIBRARIES})
