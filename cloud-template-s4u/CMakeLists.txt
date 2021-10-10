cmake_minimum_required(VERSION 3.1)

# core

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(SimGridCloudTemplate) # TODO: give a real name to your project here
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(SimGrid 3.22 REQUIRED) # This template requires SimGrid v3.22
include_directories(${SimGrid_INCLUDE_DIR})

# json

include(FetchContent)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.7.3)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(cloud cloud.cpp)
target_link_libraries(cloud PRIVATE ${SimGrid_LIBRARY})
target_link_libraries(cloud PRIVATE nlohmann_json::nlohmann_json)
