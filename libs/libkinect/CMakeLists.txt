cmake_minimum_required(VERSION 3.20.4)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(kinectSim)
set(PROJECT_NAME kinectSim)
set(CMAKE_BUILD_TYPE Release)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#set the path for project includes
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

execute_process(COMMAND lsb_release -sc OUTPUT_VARIABLE _distro OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_${_distro}")
message("Compiling for Ubuntu version ${_distro}")

find_package(Eigen3 REQUIRED)
if (EIGEN3_FOUND)
  message("Eigen3 found.")
  message("Include dir: ${EIGEN3_INCLUDE_DIR}")
  message("Libs: ${EIGEN3_LIBRARIES}")
  include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_FOUND)

find_package(CGAL REQUIRED)
if(CGAL_FOUND)
  message("CGAL found.")
  message("Include dir: ${CGAL_INCLUDE_DIRS}")
  message("Libs: ${CGAL_LIBRARIES}")
  include_directories(${CGAL_INCLUDE_DIRS})
endif(CGAL_FOUND)

find_package(OpenMP)
if(OPENMP_FOUND)
  message("OpenMP found.")
  message("Include dir: ${OpenMP_INCLUDE_DIRS}")
  message("Libs: ${OpenMP_LIBRARIES}")
  include_directories(${OpenMP_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${OpenMP_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DHAVE_OMP")
endif(OPENMP_FOUND)

find_package(OpenCV REQUIRED COMPONENTS core highgui)
if(OpenCV_FOUND)
  message("OpenCV found.")
  message("Include dir: ${OpenCV_INCLUDE_DIRS}")
  message("Libs: ${OpenCV_LIBS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(OpenCV_LIBS ${OpenCV_LIBS} opencv_imgcodecs)
  set(LIBS ${LIBS} ${OpenCV_LIBS})
endif(OpenCV_FOUND)

include_directories(lib/libnoise/include)

add_library(${PROJECT_NAME} SHARED src/kinectSimulator.cpp src/noiseutils.cpp src/simulate.cpp)
target_link_libraries(${PROJECT_NAME} assimp CGAL noise ${LIBS})
target_link_directories(${PROJECT_NAME} PRIVATE lib/libnoise/bin)

add_executable(render_object src/main.cpp)
target_link_libraries(render_object ${PROJECT_NAME})
target_link_directories(render_object PRIVATE lib/libnoise/bin)
